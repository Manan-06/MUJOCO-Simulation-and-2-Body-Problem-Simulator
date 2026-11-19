#!/usr/bin/env python3
"""
Unitree G1 apple toss in MuJoCo.

The G1 reaches for the apple on the table, takes it in its right hand, turns
the palm up, then tosses the apple back and forth between its palms, and
finally puts it back on the table.

Folder layout expected (same as your scene):
    main.py
    scene.xml          <- your scene (includes g1/g1.xml)
    g1/g1.xml          <- Unitree G1 model + its mesh assets

Run:
    python main.py                    # interactive viewer
    python main.py --speed 0.3        # slow motion
    python main.py --tosses 0         # toss forever
    python main.py --headless         # no window, prints a log (quick check)
    python main.py --flip-palm        # only if the apple shows on the BACK of the hand

How it works (short version):
  * The pelvis is held fixed in space (like a robot on a test stand), so the
    script does not need a walking/balance controller.
  * Every robot joint is driven through the model's own actuators with a
    computed-torque controller (mass matrix + gravity/Coriolis compensation +
    PD), so the arms track the planned motion closely.
  * Each hand is controlled in position AND palm direction (5 constraints on a
    7-joint arm) with damped least-squares inverse kinematics every control
    step. The palm faces the apple from the side to pick it up, faces up while
    holding, throwing and catching, and faces sideways again to set it down.
  * The G1 rubber hand cannot close around a sphere, so holding is a "virtual
    grasp": while held, the apple is fixed to the palm (it moves and rotates
    with the hand). Once released it is a free physical body: gravity does the
    flight and the floor/table still collide with it. The apple does not
    collide with the robot, so it can't snag on the arm.
  * The throw velocity is solved exactly for the simulator's integrator, and
    the catching palm is planned to the apple's predicted position at the
    moment of release, using the apple's real release state.
"""

import argparse
import math
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import mujoco

# ----------------------------------------------------------------------------
# Tunables (lengths in metres, times in seconds).
# Directions are in the ROBOT frame: x forward, y to the robot's left, z up.
# ----------------------------------------------------------------------------
FLIGHT_TIME = 0.50                          # airtime of each toss
STATION = np.array([-0.08, 0.18, 0.22])     # hold point vs apple start (y mirrored per hand)
PALM_GAP = 0.055                            # hand centre to apple centre (apple radius + half hand thickness)
PREGRASP = np.array([0.0, -0.07, 0.095])    # apple-space approach point for the right hand
GRASP_RAISE = 0.015                         # grip the apple slightly above its centre (clears the table)
THROW_STROKE = 0.18                         # length of the accelerating throw stroke
FOLLOW_THROUGH = 0.15                       # hand keeps moving this far after release
CATCH_STROKE = 0.08                         # catching hand accelerates along the apple's path
CUSHION = 0.08                              # and brakes over this distance after the catch
CATCH_SPEED_RATIO = 0.4                     # hand speed at catch / apple speed at catch
GRASP_TOL = 0.05                            # max apple-to-palm error allowed to grasp
CATCH_TOL = 0.05                            # max apple-to-palm error allowed to catch
PAUSE = 0.35                                # rest between tosses

# Palm normal in the hand (wrist_yaw_link) frame. From the G1 hand inertia the
# hand is a flat plate whose normal is the link's y axis; the palms face the
# body's centre line, so left = -y, right = +y. --flip-palm reverses both.
PALM_NORMAL_LOCAL = {"left": np.array([0.0, -1.0, 0.0]), "right": np.array([0.0, 1.0, 0.0])}

KP, KD = 400.0, 40.0                        # joint-space gains (1/s^2, 1/s), critically damped
K_NULL = 0.5                                # pull toward the home arm posture (IK null space)
IK_DAMPING = 0.03
W_ORI = 0.15                                # metres per radian: weight of palm direction vs position
QDD_MAX = 500.0

JNT_FREE = int(mujoco.mjtJoint.mjJNT_FREE)
JNT_HINGE = int(mujoco.mjtJoint.mjJNT_HINGE)
JNT_SLIDE = int(mujoco.mjtJoint.mjJNT_SLIDE)


# ----------------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------------

def find_robot_file(scene_path):
    root = ET.parse(scene_path).getroot()
    for inc in root.iter("include"):
        f = inc.get("file")
        if f:
            return (scene_path.parent / f).resolve(), f
    return None, None


def load_model(scene_path):
    """Load the scene. If that fails (typical causes: the robot file's meshdir
    is resolved relative to the scene folder, or the robot's keyframe has the
    wrong size once the apple is added), retry with a temporary copy of the
    scene placed next to the robot file and the robot keyframe stripped."""
    try:
        return mujoco.MjModel.from_xml_path(str(scene_path))
    except Exception as first_err:
        robot_path, inc = find_robot_file(scene_path)
        if robot_path is None or not robot_path.exists():
            raise
        print(f"[load] direct load failed ({str(first_err).strip().splitlines()[0]})")
        print("[load] retrying with a temporary scene next to the robot file ...")
        rdir = robot_path.parent
        tmp_robot = rdir / "_apple_toss_robot.xml"
        tmp_scene = rdir / "_apple_toss_scene.xml"
        robot_txt = robot_path.read_text(encoding="utf-8")
        robot_txt = re.sub(r"<keyframe\b.*?</keyframe>", "", robot_txt, flags=re.S)
        robot_txt = re.sub(r"<keyframe\b[^>]*/>", "", robot_txt)
        scene_txt = scene_path.read_text(encoding="utf-8")
        scene_txt = re.sub(r'(<include\s+file\s*=\s*)(["\'])' + re.escape(inc) + r"\2",
                           r'\g<1>"' + tmp_robot.name + '"', scene_txt, count=1)
        try:
            tmp_robot.write_text(robot_txt, encoding="utf-8")
            tmp_scene.write_text(scene_txt, encoding="utf-8")
            m = mujoco.MjModel.from_xml_path(str(tmp_scene))
            print("[load] OK")
            return m
        except Exception as second_err:
            raise RuntimeError(f"Could not load the scene.\n  direct load: {first_err}\n"
                               f"  retry:       {second_err}") from None
        finally:
            for f in (tmp_robot, tmp_scene):
                try:
                    f.unlink()
                except OSError:
                    pass


def robot_keyframe_pose(robot_path):
    """Read the robot's standing keyframe (if any) from the robot file alone.
    Returns {joint_name: qpos values}; key '' is the unnamed free joint."""
    if robot_path is None or not robot_path.exists():
        return {}, None
    try:
        rm = mujoco.MjModel.from_xml_path(str(robot_path))
    except Exception:
        return {}, None
    if rm.nkey == 0:
        return {}, None
    names = [mujoco.mj_id2name(rm, mujoco.mjtObj.mjOBJ_KEY, k) or "" for k in range(rm.nkey)]
    k = 0
    for pref in ("stand", "home", "standing", "default"):
        if pref in names:
            k = names.index(pref)
            break
    key_qpos = rm.key_qpos[k]
    pose = {}
    for j in range(rm.njnt):
        n = mujoco.mj_id2name(rm, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        size = {JNT_FREE: 7, int(mujoco.mjtJoint.mjJNT_BALL): 4}.get(int(rm.jnt_type[j]), 1)
        a = rm.jnt_qposadr[j]
        pose[n] = key_qpos[a:a + size].copy()
    return pose, (names[k] or f"#{k}")


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def jname(m, j):
    return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""


def body_depth(m, b):
    n = 0
    while b != 0:
        b = m.body_parentid[b]
        n += 1
    return n


def dls(J, lam):
    """Damped least-squares pseudo-inverse."""
    return J.T @ np.linalg.solve(J @ J.T + (lam * lam) * np.eye(J.shape[0]), np.eye(J.shape[0]))


def flight_terms(m, T):
    """Steps, exact duration, and gravity displacement of a ballistic flight
    as integrated by MuJoCo (semi-implicit Euler unless RK4)."""
    dt = m.opt.timestep
    n = max(1, int(round(T / dt)))
    Tn = n * dt
    g = np.array(m.opt.gravity, dtype=float)
    if int(m.opt.integrator) == int(mujoco.mjtIntegrator.mjINT_RK4):
        gterm = 0.5 * g * Tn * Tn
    else:
        gterm = g * dt * dt * n * (n + 1) / 2.0
    return n, Tn, gterm, g


# ----------------------------------------------------------------------------
# Trajectory segments (positions are "apple-space": where the apple centre is)
# ----------------------------------------------------------------------------
class MinJerk:
    def __init__(self, p0, p1, t0, dur):
        self.p0 = np.array(p0, float)
        self.p1 = np.array(p1, float)
        self.t0, self.dur = t0, max(dur, 1e-3)
        self.t1 = t0 + self.dur

    def __call__(self, t):
        s = min(max((t - self.t0) / self.dur, 0.0), 1.0)
        d = self.p1 - self.p0
        pos = 10 * s**3 - 15 * s**4 + 6 * s**5
        vel = (30 * s**2 - 60 * s**3 + 30 * s**4) / self.dur
        return self.p0 + d * pos, d * vel


class ConstAccel:
    def __init__(self, p0, v0, a, t0, dur):
        self.p0, self.v0, self.a = np.array(p0, float), np.array(v0, float), np.array(a, float)
        self.t0, self.dur = t0, dur
        self.t1 = t0 + dur

    def __call__(self, t):
        tau = min(max(t - self.t0, 0.0), self.dur)
        return self.p0 + self.v0 * tau + 0.5 * self.a * tau * tau, self.v0 + self.a * tau


class HandPath:
    """Append-only queue of segments; holds the last point when idle."""

    def __init__(self, p):
        self.segs = []
        self.hold = np.array(p, float)

    def add(self, seg):
        self.segs.append(seg)
        return seg

    def end(self, t_now):
        if self.segs:
            s = self.segs[-1]
            return max(t_now, s.t1), s(s.t1)[0]
        return t_now, self.hold.copy()

    def __call__(self, t):
        while self.segs and t > self.segs[0].t1 + 1e-9:
            s = self.segs.pop(0)
            self.hold = s(s.t1)[0]
        if self.segs and t >= self.segs[0].t0 - 1e-9:
            return self.segs[0](t)
        return self.hold.copy(), np.zeros(3)


# ----------------------------------------------------------------------------
# Arms and inverse kinematics
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Arms and inverse kinematics (hand position + palm direction)
# ----------------------------------------------------------------------------
def unit(v):
    return np.asarray(v, float) / np.linalg.norm(v)


def angle_between(a, b):
    return float(np.arccos(np.clip(np.dot(unit(a), unit(b)), -1.0, 1.0)))


class Arm:
    def __init__(self, m, side, joints, palm_local):
        self.side, self.joints = side, joints
        self.qadr = np.array([m.jnt_qposadr[j] for j in joints])
        self.dadr = np.array([m.jnt_dofadr[j] for j in joints])
        lo, hi = [], []
        for j in joints:
            if m.jnt_limited[j]:
                lo.append(m.jnt_range[j, 0] + 0.01)
                hi.append(m.jnt_range[j, 1] - 0.01)
            else:
                lo.append(-np.inf)
                hi.append(np.inf)
        self.lo, self.hi = np.array(lo), np.array(hi)
        self.body = max((int(m.jnt_bodyid[j]) for j in joints), key=lambda b: body_depth(m, b))
        self.local = np.zeros(3)
        self.palm_local = unit(palm_local)
        self.jacp = np.zeros((3, m.nv))
        self.jacr = np.zeros((3, m.nv))

    def set_hand_point(self, d):
        # Hand centre = centre of mass of the whole hand subtree, fixed in the hand frame.
        R = d.xmat[self.body].reshape(3, 3)
        self.local = R.T @ (d.subtree_com[self.body] - d.xpos[self.body])

    def R(self, d):
        return d.xmat[self.body].reshape(3, 3)

    def point(self, d):
        return d.xpos[self.body] + self.R(d) @ self.local

    def normal(self, d):
        return self.R(d) @ self.palm_local

    def jacs(self, m, d, p):
        mujoco.mj_jac(m, d, self.jacp, self.jacr, np.ascontiguousarray(p, dtype=np.float64), self.body)
        return self.jacp[:, self.dadr], self.jacr[:, self.dadr]

    def twist(self, m, d, p):
        """Linear velocity of point p on the hand, and the hand's angular velocity (world)."""
        mujoco.mj_jac(m, d, self.jacp, self.jacr, np.ascontiguousarray(p, dtype=np.float64), self.body)
        return self.jacp @ d.qvel, self.jacr @ d.qvel


def arm_task(m, d, a, tgt):
    """Stacked task: 3 position rows + 3 palm-direction rows (rank 2, spin about
    the palm normal is left free). Returns J, error, feed-forward velocity."""
    p_des, v_des, n_des, nd_des = tgt
    p = a.point(d)
    n = a.normal(d)
    Jp, Jr = a.jacs(m, d, p)
    P = np.eye(3) - np.outer(n, n)
    J = np.vstack([Jp, W_ORI * (P @ Jr)])
    e = np.concatenate([p_des - p, W_ORI * np.cross(n, n_des)])
    w_des = np.cross(n_des, nd_des)                  # angular velocity of the target normal
    vel = np.concatenate([v_des, W_ORI * (P @ w_des)])
    return J, e, vel, float(np.linalg.norm(p_des - p)), angle_between(n, n_des)


class DualArmIK:
    """Closed-loop differential IK for both hands. Produces q, qdot, qddot references."""

    def __init__(self, m, qpos_home, arms):
        self.m = m
        self.d = mujoco.MjData(m)
        self.d.qpos[:] = qpos_home
        self.arms = {a.side: a for a in arms}
        self.home = {a.side: qpos_home[a.qadr].copy() for a in arms}
        self.q = {s: q.copy() for s, q in self.home.items()}
        self.qd = {a.side: np.zeros(len(a.qadr)) for a in arms}
        self.max_err = 0.0
        self.max_ang = 0.0

    def _fk(self):
        for s, a in self.arms.items():
            self.d.qpos[a.qadr] = self.q[s]
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)

    def step(self, targets, dt, iters=4, tol=2e-4, tol_ang=2e-3):
        tasks = {}
        for it in range(iters + 1):
            self._fk()
            worst, worst_ang = 0.0, 0.0
            for s, a in self.arms.items():
                tasks[s] = arm_task(self.m, self.d, a, targets[s])
                worst = max(worst, tasks[s][3])
                worst_ang = max(worst_ang, tasks[s][4])
            if (worst < tol and worst_ang < tol_ang) or it == iters:
                break
            for s, a in self.arms.items():
                J, e = tasks[s][0], tasks[s][1]
                dq = dls(J, IK_DAMPING) @ e
                n = np.linalg.norm(dq)
                if n > 0.3:
                    dq *= 0.3 / n
                self.q[s] = np.clip(self.q[s] + dq, a.lo, a.hi)
        self.max_err = max(self.max_err, worst)
        self.max_ang = max(self.max_ang, worst_ang)

        out = {}
        for s, a in self.arms.items():
            J, e, vel = tasks[s][0], tasks[s][1], tasks[s][2]
            Jp = dls(J, IK_DAMPING)
            N = np.eye(len(a.qadr)) - Jp @ J
            qd = Jp @ (vel + 20.0 * e) + N @ (K_NULL * (self.home[s] - self.q[s]))
            qdd = np.clip((qd - self.qd[s]) / dt, -QDD_MAX, QDD_MAX)
            out[s] = (self.q[s].copy(), qd, qdd)
            self.qd[s] = qd
            self.q[s] = np.clip(self.q[s] + qd * dt, a.lo, a.hi)   # prediction for next step
        return out

    def reach_error(self, side, p_target, n_target, iters=400):
        """Offline check: how close can this hand get to a pose from home?
        Returns (position error in m, palm angle error in rad)."""
        a = self.arms[side]
        saved = self.q[side].copy()
        q = self.home[side].copy()
        tgt = (p_target, np.zeros(3), unit(n_target), np.zeros(3))
        for _ in range(iters):
            self.q[side] = q
            self._fk()
            J, e, _, pe, ae = arm_task(self.m, self.d, a, tgt)
            if pe < 1e-4 and ae < 1e-3:
                break
            dq = dls(J, 0.05) @ e
            n = np.linalg.norm(dq)
            if n > 0.2:
                dq *= 0.2 / n
            q = np.clip(q + dq, a.lo, a.hi)
        self.q[side] = q
        self._fk()
        _, _, _, pe, ae = arm_task(self.m, self.d, a, tgt)
        self.q[side] = saved
        return pe, ae



def find_arm_joints(m, side, robot_root):
    short = "l_" if side == "left" else "r_"
    parts = ("shoulder", "elbow", "wrist")
    bad = ("hand", "finger", "thumb", "index", "middle", "ring", "pinky", "little")
    out = []
    for j in range(m.njnt):
        if int(m.jnt_type[j]) != JNT_HINGE or m.body_rootid[m.jnt_bodyid[j]] != robot_root:
            continue
        n = jname(m, j).lower()
        if not (side in n or n.startswith(short)):
            continue
        if not any(p in n for p in parts) or any(b in n for b in bad):
            continue
        out.append(j)
    out.sort(key=lambda j: body_depth(m, m.jnt_bodyid[j]))
    return out



class Controller:
    def __init__(self, m, robot_joints):
        self.m = m
        self.rq = np.array([m.jnt_qposadr[j] for j in robot_joints])
        self.rd = np.array([m.jnt_dofadr[j] for j in robot_joints])
        self.M = np.zeros((m.nv, m.nv))
        dof_to_k = {int(dof): k for k, dof in enumerate(self.rd)}
        jnt_set = set(robot_joints)

        self.acts = []           # (actuator id, k, gear, gain, b0, b1, b2, limited, lo, hi)
        driven = set()
        skipped = []
        for i in range(m.nu):
            if int(m.actuator_trntype[i]) != int(mujoco.mjtTrn.mjTRN_JOINT):
                continue
            j = int(m.actuator_trnid[i, 0])
            if j not in jnt_set:
                continue
            k = dof_to_k[int(m.jnt_dofadr[j])]
            ok = (int(m.actuator_dyntype[i]) == int(mujoco.mjtDyn.mjDYN_NONE)
                  and int(m.actuator_gaintype[i]) == int(mujoco.mjtGain.mjGAIN_FIXED)
                  and int(m.actuator_biastype[i]) in (int(mujoco.mjtBias.mjBIAS_NONE),
                                                      int(mujoco.mjtBias.mjBIAS_AFFINE))
                  and abs(m.actuator_gainprm[i, 0]) > 1e-9
                  and abs(m.actuator_gear[i, 0]) > 1e-9
                  and k not in driven)
            if not ok:
                skipped.append(i)
                continue
            driven.add(k)
            b = m.actuator_biasprm[i] if int(m.actuator_biastype[i]) == int(mujoco.mjtBias.mjBIAS_AFFINE) \
                else np.zeros(3)
            self.acts.append((i, k, float(m.actuator_gear[i, 0]), float(m.actuator_gainprm[i, 0]),
                              float(b[0]), float(b[1]), float(b[2]),
                              bool(m.actuator_ctrllimited[i]),
                              float(m.actuator_ctrlrange[i, 0]), float(m.actuator_ctrlrange[i, 1])))
        self.unact = np.array([k for k in range(len(self.rd)) if k not in driven], dtype=int)
        self.skipped = skipped
        self.n_driven = len(driven)
        self.sat_steps = 0
        self.steps = 0
        self._mass_method = None

    # Mass matrix, robust across MuJoCo versions: older ones store it in d.qM,
    # newer ones in d.M. The first call picks a method and verifies it against
    # mj_mulM (available in every version); if nothing verifies, it builds the
    # robot block column by column with mj_mulM.
    def _mass_candidates(self, d):
        m = self.m
        for attr in ("qM", "M"):
            if hasattr(d, attr):
                def f(d, attr=attr):
                    mujoco.mj_fullM(m, self.M, getattr(d, attr))
                    return self.M[np.ix_(self.rd, self.rd)].copy()
                yield attr, f
        yield "mj_mulM", self._mass_by_mulM

    def _mass_by_mulM(self, d):
        m = self.m
        e = np.zeros(m.nv)
        col = np.zeros(m.nv)
        out = np.empty((len(self.rd), len(self.rd)))
        for c, dof in enumerate(self.rd):
            e[:] = 0.0
            e[dof] = 1.0
            mujoco.mj_mulM(m, d, col, e)
            out[:, c] = col[self.rd]
        return out

    def robot_mass_matrix(self, d):
        if self._mass_method is None:
            ref = self._mass_by_mulM(d)
            for name, f in self._mass_candidates(d):
                try:
                    Mr = f(d)
                except Exception:
                    continue
                if Mr.shape == ref.shape and np.allclose(Mr, ref, rtol=1e-6, atol=1e-9):
                    self._mass_method = f
                    print(f"[robot] mass matrix via {name}")
                    break
        return self._mass_method(d)

    def apply(self, d, q_des, qd_des, qdd_des):
        m = self.m
        Mr = self.robot_mass_matrix(d)
        e = q_des - d.qpos[self.rq]
        ed = qd_des - d.qvel[self.rd]
        tau = Mr @ (qdd_des + KP * e + KD * ed) + d.qfrc_bias[self.rd] - d.qfrc_passive[self.rd]

        d.qfrc_applied[self.rd] = 0.0
        sat = False
        for (i, k, gear, gain, b0, b1, b2, lim, lo, hi) in self.acts:
            force = tau[k] / gear
            c = (force - b0 - b1 * d.actuator_length[i] - b2 * d.actuator_velocity[i]) / gain
            if lim:
                cc = min(max(c, lo), hi)
                sat |= cc != c
                c = cc
            d.ctrl[i] = c
        if len(self.unact):
            d.qfrc_applied[self.rd[self.unact]] = tau[self.unact]
        self.steps += 1
        self.sat_steps += int(sat)


# ----------------------------------------------------------------------------
# The show: state machine + simulation step
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# The show: state machine + simulation step
# ----------------------------------------------------------------------------
class AppleToss:
    def __init__(self, scene_path, n_tosses, flight_time, flip_palm=False, verbose=True):
        self.log_on = verbose
        self.n_target = n_tosses
        self.flight_time = flight_time
        self.gap = PALM_GAP

        m = load_model(scene_path)
        d = mujoco.MjData(m)
        self.m, self.d = m, d
        self.dt = m.opt.timestep
        if self.dt > 0.003:
            print(f"[warn] timestep is {self.dt*1000:.1f} ms; 1-2 ms is recommended for the fast throw.")

        # --- apple -----------------------------------------------------------
        self.apple = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "apple")
        if self.apple < 0:
            sys.exit("No body named 'apple' in the scene.")
        aj = m.body_jntadr[self.apple]
        if aj < 0 or int(m.jnt_type[aj]) != JNT_FREE:
            sys.exit("The apple needs a free joint.")
        self.aq, self.ad = m.jnt_qposadr[aj], m.jnt_dofadr[aj]

        # --- robot base and joints -------------------------------------------
        self.base_j = None
        for j in range(m.njnt):
            if int(m.jnt_type[j]) == JNT_FREE and m.jnt_bodyid[j] != self.apple:
                self.base_j = j
                break
        if self.base_j is not None:
            self.pelvis = int(m.jnt_bodyid[self.base_j])
        else:
            pid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
            self.pelvis = pid if pid >= 0 else 1
        self.robot_root = int(m.body_rootid[self.pelvis])
        robot_joints = [j for j in range(m.njnt)
                        if int(m.jnt_type[j]) in (JNT_HINGE, JNT_SLIDE)
                        and m.body_rootid[m.jnt_bodyid[j]] == self.robot_root]

        sign = -1.0 if flip_palm else 1.0
        arms = []
        for side in ("right", "left"):
            js = find_arm_joints(m, side, self.robot_root)
            if len(js) < 4:
                names = ", ".join(jname(m, j) for j in robot_joints)
                sys.exit(f"Could not find the {side} arm joints (found {len(js)}). Robot joints: {names}")
            arms.append(Arm(m, side, js, sign * PALM_NORMAL_LOCAL[side]))
        self.arms = {a.side: a for a in arms}

        # --- initial pose: robot keyframe from the robot file (if it has one) --
        mujoco.mj_resetData(m, d)
        robot_path, _ = find_robot_file(scene_path)
        pose, keyname = robot_keyframe_pose(robot_path)
        applied = 0
        for j in range(m.njnt):
            if m.body_rootid[m.jnt_bodyid[j]] != self.robot_root:
                continue
            n = jname(m, j)
            val = pose.get(n)
            if val is None and j == self.base_j and "" in pose and len(pose[""]) == 7:
                val = pose[""]
            if val is None:
                continue
            size = 7 if int(m.jnt_type[j]) == JNT_FREE else 1
            if len(val) == size:
                d.qpos[m.jnt_qposadr[j]:m.jnt_qposadr[j] + size] = val
                applied += 1
        mujoco.mj_forward(m, d)
        self.qpos_home = d.qpos.copy()
        if self.base_j is not None:
            qa, da = m.jnt_qposadr[self.base_j], m.jnt_dofadr[self.base_j]
            self.base_slice_q = slice(qa, qa + 7)
            self.base_slice_v = slice(da, da + 6)
            self.base_q = d.qpos[self.base_slice_q].copy()

        for a in arms:
            a.set_hand_point(d)

        # --- contacts: apple touches floor/table, not the robot ---------------
        self._setup_contacts()

        # --- robot frame (yaw of the pelvis) and key points --------------------
        R = d.xmat[self.pelvis].reshape(3, 3)
        yaw = math.atan2(R[1, 0], R[0, 0])
        c, s = math.cos(yaw), math.sin(yaw)
        self.Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        self.yaw = yaw
        W = lambda v: self.Rz @ np.asarray(v, float)

        self.A = d.xpos[self.apple].copy()
        self.up = np.array([0.0, 0.0, 1.0])
        self.station = {"left": self.A + W(STATION),
                        "right": self.A + W(STATION * np.array([1, -1, 1]))}
        self.pregrasp = self.A + W(PREGRASP)
        # palm directions used to pick up / put down: facing the apple from the outside
        self.side_normal = {"right": W([0, 1, 0]), "left": W([0, -1, 0])}

        fwd = (self.A - d.xpos[self.pelvis]) @ W([1, 0, 0])
        if fwd <= 0:
            print("[warn] the apple is not in front of the robot; the arms may not reach it.")

        # --- controller + IK ---------------------------------------------------
        self.ctrl = Controller(m, robot_joints)
        self.q_des = self.qpos_home[self.ctrl.rq].copy()
        self.k_of_qadr = {int(a): k for k, a in enumerate(self.ctrl.rq)}
        self.ik = DualArmIK(m, self.qpos_home, arms)

        if self.log_on:
            print(f"[robot] {len(robot_joints)} joints, {self.ctrl.n_driven} driven by actuators, "
                  f"{len(self.ctrl.unact)} by direct joint torque; pelvis fixed: {self.base_j is not None}")
            print(f"[robot] start pose: keyframe '{keyname}' ({applied} joints)" if keyname
                  else "[robot] start pose: model default (no keyframe found)")
            for a in arms:
                print(f"[robot] {a.side:5s} arm: " + ", ".join(jname(m, j) for j in a.joints)
                      + f"  | hand body: {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, a.body)}"
                      + f"  | palm normal (hand frame): {np.round(a.palm_local, 2)}")
            if self.ctrl.skipped:
                print(f"[warn] {len(self.ctrl.skipped)} actuator(s) of an unsupported type left at ctrl=0")
        self._reach_check()

        # --- state -------------------------------------------------------------
        # paths[side]  : where the apple centre would be if held by that hand
        # npaths[side] : desired palm normal (world)
        self.paths, self.npaths = {}, {}
        for side, a in self.arms.items():
            n0 = a.normal(d)
            self.paths[side] = HandPath(a.point(d) + n0 * self.gap)
            self.npaths[side] = HandPath(n0)
        self.held = None
        self.hold_local = np.zeros(3)
        self.hold_qrel = np.array([1.0, 0, 0, 0])
        self.settle = None
        self.holder = None
        self.tosses = 0
        self.state = "reach"
        self.max_track = 0.0
        self.max_palm = 0.0
        self.track_where = ""
        self.palm_where = ""
        self.phase_err = {}                     # state -> worst hand error (m)
        self.force_sat = {}                     # joint name -> steps at its force limit
        self.touch = {}                         # "arm-table" etc -> steps in contact
        self._diag_ok = True
        self.drop_reported = False

        # opening moves: right hand to the apple (palm facing it), left hand to its station (palm up)
        self.move("right", self.pregrasp, 1.6, n_to=self.side_normal["right"], t0=0.3)
        s = self.move("right", self.A + self.up * GRASP_RAISE, 0.9)
        self.move("left", self.station["left"], 1.6, n_to=self.up, t0=0.3)
        self.t_event = s.t1 + 0.15

    # ------------------------------------------------------------------ setup
    def _setup_contacts(self):
        """Collision bits:
             apple      : collides with the world (floor, table) only
             arm links  : collide with the world only, not with the robot's own body.
                          MuJoCo collides meshes through their convex hulls, and the G1
                          torso/arm hulls overlap where the arms naturally move, which
                          otherwise pins the arms against the body.
             rest of the robot (legs, torso) and the world: unchanged."""
        m = self.m
        arm_bodies = {int(m.jnt_bodyid[j]) for a in self.arms.values() for j in a.joints}
        for g in range(m.ngeom):
            b = int(m.geom_bodyid[g])
            collides = bool(m.geom_contype[g] or m.geom_conaffinity[g])
            if b == self.apple:
                m.geom_contype[g] = 2
                m.geom_conaffinity[g] = 2
            elif b in arm_bodies:
                if collides:
                    m.geom_contype[g] = 4          # (arm.contype & world.conaffinity) -> touches world
                    m.geom_conaffinity[g] = 0      # nothing on the robot has bit 4 in contype
            elif m.body_rootid[b] == self.robot_root and b != 0:
                m.geom_contype[g] &= ~6
                m.geom_conaffinity[g] &= ~6
            elif collides:
                m.geom_contype[g] |= 2
                m.geom_conaffinity[g] |= 2 | 4

    def _throw_geometry(self, src, dst):
        n, Tn, gterm, g = flight_terms(self.m, self.flight_time)
        S_f, S_t = self.station[src], self.station[dst]
        v = (S_t - S_f - gterm) / Tn
        vh = v / np.linalg.norm(v)
        return v, vh, S_f - vh * THROW_STROKE, S_f + vh * FOLLOW_THROUGH

    def _reach_check(self):
        up = self.up
        put = lambda side, h: self.A + up * h
        pts = {"right": [("pre-grasp", self.pregrasp, self.side_normal["right"]),
                         ("grasp", put("right", GRASP_RAISE), self.side_normal["right"]),
                         ("station", self.station["right"], up),
                         ("set-down", put("right", 0.012), self.side_normal["right"])],
               "left": [("station", self.station["left"], up),
                        ("set-down", put("left", 0.012), self.side_normal["left"])]}
        for src, dst in (("right", "left"), ("left", "right")):
            _, _, dip, peak = self._throw_geometry(src, dst)
            pts[src] += [("throw dip", dip, up), ("follow-through", peak, up)]
        bad = []
        for side, lst in pts.items():
            for name, p, n in lst:
                pe, ae = self.ik.reach_error(side, p - n * self.gap, n)
                if pe > 0.01 or ae > math.radians(10):
                    bad.append(f"{side} {name}: {pe*100:.1f} cm short, palm off by {math.degrees(ae):.0f} deg")
        if bad:
            print("[warn] some poses can't be reached exactly (the motion still runs, "
                  "but may look off there):\n   " + "\n   ".join(bad))
        elif self.log_on:
            print("[check] every planned hand pose (position + palm direction) is reachable")

    # ---------------------------------------------------------------- helpers
    def log(self, msg):
        if self.log_on:
            print(f"[t={self.d.time:6.2f}s] {msg}")

    def other(self, side):
        return "left" if side == "right" else "right"

    def move(self, side, p_to, dur, n_to=None, t0=None):
        """Minimum-jerk move of the (apple-space) hand target, optionally turning
        the palm to n_to over the same interval. Starts when the previous move ends."""
        pp, npth = self.paths[side], self.npaths[side]
        t_now = self.d.time if t0 is None else t0
        t_start, p_from = pp.end(t_now)
        t_start = max(t_start, t_now)
        seg = pp.add(MinJerk(p_from, p_to, t_start, dur))
        if n_to is not None:
            _, n_from = npth.end(t_start)
            npth.add(MinJerk(unit(n_from), unit(n_to), t_start, dur))
        return seg

    def attach_apple(self):
        """Apple fixed to the palm: pose and velocity follow the hand exactly."""
        a = self.arms[self.held]
        d = self.d
        p = a.point(d)
        v, w = a.twist(self.m, d, p)
        local = self.hold_local
        if self.settle is not None:                  # apple slides into the palm centre after pickup
            t0, dur, l0, l1 = self.settle
            sf = min(max((d.time - t0) / dur, 0.0), 1.0)
            local = l0 + (l1 - l0) * (10 * sf**3 - 15 * sf**4 + 6 * sf**5)
            if sf >= 1.0:
                self.hold_local, self.settle = l1, None
        r = a.R(d) @ local
        q = np.zeros(4)
        mujoco.mju_mulQuat(q, d.xquat[a.body], self.hold_qrel)
        q /= np.linalg.norm(q)
        Ra = np.zeros(9)
        mujoco.mju_quat2Mat(Ra, q)
        d.qpos[self.aq:self.aq + 3] = p + r
        d.qpos[self.aq + 3:self.aq + 7] = q
        d.qvel[self.ad:self.ad + 3] = v + np.cross(w, r)
        d.qvel[self.ad + 3:self.ad + 6] = Ra.reshape(3, 3).T @ w     # free joint: local angular velocity

    def try_grab(self, side, tol, what, settle=0.0):
        d = self.d
        a = self.arms[side]
        apple = d.qpos[self.aq:self.aq + 3].copy()
        p = a.point(d)
        err = float(np.linalg.norm(apple - (p + a.normal(d) * self.gap)))
        if err > tol:
            self.log(f"{what} FAILED with the {side} hand: apple is {err*100:.1f} cm from the palm")
            return False
        self.held = side
        self.holder = side
        self.hold_local = a.R(d).T @ (apple - p)
        qa = d.qpos[self.aq + 3:self.aq + 7].copy()
        qa /= np.linalg.norm(qa)
        qh_inv = np.zeros(4)
        mujoco.mju_negQuat(qh_inv, d.xquat[a.body])
        self.hold_qrel = np.zeros(4)
        mujoco.mju_mulQuat(self.hold_qrel, qh_inv, qa)
        self.settle = None
        if settle > 0:
            self.settle = (d.time, settle, self.hold_local.copy(), a.palm_local * self.gap)
        self.log(f"{what} with the {side} hand (error {err*1000:.1f} mm)")
        return True

    # --------------------------------------------------------------- planning
    def plan_carry(self, t):
        # lift with the palm still facing the apple, then turn the palm up on the way to the station
        self.move("right", self.A + self.up * 0.10, 0.5, t0=t)
        s = self.move("right", self.station["right"], 0.9, n_to=self.up)
        self.t_event = s.t1 + PAUSE

    def plan_throw(self, t):
        src = self.holder
        dst = self.other(src)
        v, vh, dip, peak = self._throw_geometry(src, dst)
        speed = float(np.linalg.norm(v))
        S_f = self.station[src]
        path = self.paths[src]
        t0, _ = path.end(t)
        s1 = path.add(MinJerk(S_f, dip, t0, 0.45))
        s2 = path.add(MinJerk(dip, dip, s1.t1, 0.10))
        acc = speed**2 / (2 * THROW_STROKE)
        s3 = path.add(ConstAccel(dip, np.zeros(3), vh * acc, s2.t1, 2 * THROW_STROKE / speed))
        dec = speed**2 / (2 * FOLLOW_THROUGH)
        s4 = path.add(ConstAccel(S_f, v, -vh * dec, s3.t1, 2 * FOLLOW_THROUGH / speed))
        path.add(MinJerk(peak, S_f, s4.t1, 0.55))
        self.t_release = s3.t1
        self.v_throw = v
        self.thrower, self.catcher = src, dst

    def release_throw(self, t):
        self.attach_apple()                       # exact pose at this instant
        self.d.qvel[self.ad:self.ad + 3] = self.v_throw
        self.d.qvel[self.ad + 3:self.ad + 6] = (2.0, 0.0, 0.0)   # a little spin, looks natural
        self.held = None
        P = self.d.qpos[self.aq:self.aq + 3].copy()
        n, Tn, gterm, g = flight_terms(self.m, self.flight_time)
        C = P + self.v_throw * Tn + gterm          # where the apple will be at catch time
        v_in = self.v_throw + g * Tn               # its velocity then
        self.t_catch = t + Tn
        self.log(f"release from the {self.thrower} hand at {np.linalg.norm(self.v_throw):.2f} m/s")

        dst = self.catcher
        u = CATCH_SPEED_RATIO * v_in
        us = float(np.linalg.norm(u))
        uh = u / us
        t_acc = 2 * CATCH_STROKE / us
        C_pre = C - uh * CATCH_STROKE
        path = self.paths[dst]
        t0, start = path.end(t)
        move_dur = (self.t_catch - t_acc) - t0 - 0.03
        if move_dur < 0.08:
            self.log("[warn] catching hand has very little time; increase --flight-time")
        s1 = path.add(MinJerk(start, C_pre, t0, max(move_dur, 0.05)))
        s2 = path.add(MinJerk(C_pre, C_pre, s1.t1, (self.t_catch - t_acc) - s1.t1))
        s3 = path.add(ConstAccel(C_pre, np.zeros(3), uh * us**2 / (2 * CATCH_STROKE), s2.t1, t_acc))
        s4 = path.add(ConstAccel(C, u, -uh * us**2 / (2 * CUSHION), s3.t1, 2 * CUSHION / us))
        path.add(MinJerk(C + uh * CUSHION, self.station[dst], s4.t1, 0.6))

    def plan_place(self, t):
        side = self.holder
        n_place = self.side_normal[side]
        self.move(side, self.A + self.up * 0.08, 1.0, n_to=n_place, t0=t)
        s = self.move(side, self.A + self.up * 0.012, 0.6)
        self.t_event = s.t1 + 0.1

    def place_release(self, t):
        self.attach_apple()
        self.d.qvel[self.ad:self.ad + 6] = 0.0
        side = self.held
        self.held = None
        self.log(f"apple placed back on the table by the {side} hand")
        _, p_now = self.paths[side].end(t)
        back = p_now - self.side_normal[side] * 0.07 + self.up * 0.08
        self.move(side, back, 0.6, t0=t)
        s = self.move(side, self.station[side], 0.8, n_to=self.up)
        self.t_event = s.t1

    # --------------------------------------------------------------- director
    def director(self, t):
        st = self.state
        if st == "reach" and t >= self.t_event:
            if self.try_grab("right", GRASP_TOL, "grasp", settle=0.8):
                self.plan_carry(t)
                self.state = "hold"
            else:
                self.state = "failed"
        elif st == "hold" and t >= self.t_event:
            if self.n_target == 0 or self.tosses < self.n_target:
                self.plan_throw(t)
                self.state = "throw"
            else:
                self.plan_place(t)
                self.state = "place"
        elif st == "throw" and t >= self.t_release - 0.5 * self.dt:
            self.release_throw(t)
            self.state = "flight"
        elif st == "flight" and t >= self.t_catch - 0.5 * self.dt:
            if self.try_grab(self.catcher, CATCH_TOL, f"catch #{self.tosses + 1}"):
                self.tosses += 1
                t_end, _ = self.paths[self.catcher].end(t)
                self.t_event = t_end + PAUSE
                self.state = "hold"
            else:
                self.state = "dropped"
        elif st == "place" and t >= self.t_event:
            self.place_release(t)
            self.state = "done"

        if self.held is None and not self.drop_reported and self.state in ("flight", "dropped", "failed"):
            if self.d.qpos[self.aq + 2] < 0.3:
                self.log("the apple hit the floor")
                self.drop_reported = True

    # ------------------------------------------------------------------- step
    def step(self):
        m, d = self.m, self.d
        t = d.time
        if self.base_j is not None:
            d.qpos[self.base_slice_q] = self.base_q
            d.qvel[self.base_slice_v] = 0.0
        mujoco.mj_forward(m, d)

        self.director(t)
        if self.held is not None:
            self.attach_apple()

        targets = {}
        for side, a in self.arms.items():
            p, v = self.paths[side](t)
            n_raw, nd_raw = self.npaths[side](t)
            nn = float(np.linalg.norm(n_raw))
            n = n_raw / nn
            nd = (nd_raw - n * float(n @ nd_raw)) / nn
            targets[side] = (p - n * self.gap, v - nd * self.gap, n, nd)
            te = float(np.linalg.norm(a.point(d) - targets[side][0]))
            pe = angle_between(a.normal(d), n)
            if te > self.max_track:
                self.max_track, self.track_where = te, f"t={t:.2f}s, {side} hand, during '{self.state}'"
            if pe > self.max_palm:
                self.max_palm, self.palm_where = pe, f"t={t:.2f}s, {side} hand, during '{self.state}'"
            self.phase_err[self.state] = max(self.phase_err.get(self.state, 0.0), te)
        ref = self.ik.step(targets, self.dt)

        q_des = self.q_des.copy()
        qd_des = np.zeros_like(q_des)
        qdd_des = np.zeros_like(q_des)
        for side, a in self.arms.items():
            q, qd, qdd = ref[side]
            idx = [self.k_of_qadr[int(x)] for x in a.qadr]
            q_des[idx], qd_des[idx], qdd_des[idx] = q, qd, qdd
        self.ctrl.apply(d, q_des, qd_des, qdd_des)
        mujoco.mj_step(m, d)
        if self._diag_ok:
            try:
                self._diagnostics()
            except Exception as exc:            # diagnostics must never stop the simulation
                print(f"[diag] disabled ({exc})")
                self._diag_ok = False

    def _diagnostics(self):
        m, d = self.m, self.d
        # joints pushing against their actuator force limit (forcerange)
        for (i, k, *_rest) in self.ctrl.acts:
            if m.actuator_forcelimited[i]:
                lo, hi = m.actuator_forcerange[i]
                f = d.actuator_force[i]
                if f <= lo + 1e-6 or f >= hi - 1e-6:
                    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"actuator {i}"
                    self.force_sat[name] = self.force_sat.get(name, 0) + 1
        # unwanted arm contacts: with the table/world, or with the robot's own body
        arm_bodies = getattr(self, "_arm_bodies", None)
        if arm_bodies is None:
            arm_bodies = set()
            for a in self.arms.values():
                for j in a.joints:
                    arm_bodies.add(int(m.jnt_bodyid[j]))
            self._arm_bodies = arm_bodies
        seen = set()
        ncon = int(d.ncon)
        if ncon == 0:
            return
        cl = d.contact
        if hasattr(cl, "geom"):
            pairs = np.asarray(cl.geom).reshape(-1, 2)[:ncon]
        else:
            pairs = np.stack([np.asarray(cl.geom1), np.asarray(cl.geom2)], axis=1)[:ncon]
        for g1, g2 in pairs:
            b1, b2 = int(m.geom_bodyid[g1]), int(m.geom_bodyid[g2])
            for ba, bb in ((b1, b2), (b2, b1)):
                if ba in arm_bodies:
                    na = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, ba) or f"body {ba}"
                    if m.body_rootid[bb] == self.robot_root and bb != 0:
                        nb = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bb) or f"body {bb}"
                        seen.add(f"{na} vs {nb}")
                    elif bb != self.apple:
                        gname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(g2 if ba == b1 else g1))
                        seen.add(f"{na} vs {gname or 'world'}")
        for key in seen:
            self.touch[key] = self.touch.get(key, 0) + 1

    @property
    def finished(self):
        if self.state == "done":
            return self.d.time > self.t_event + 1.0
        return self.state in ("dropped", "failed")

    def summary(self):
        print("\n=== summary ===")
        print(f"tosses caught      : {self.tosses}")
        print(f"final state        : {self.state}")
        print(f"max hand tracking  : {self.max_track*100:.2f} cm (actual hand vs plan) at {self.track_where}")
        print(f"max palm tilt error: {math.degrees(self.max_palm):.1f} deg (actual palm vs plan) at {self.palm_where}")
        print("worst hand error by phase: " + ", ".join(f"{k} {v*100:.1f} cm" for k, v in self.phase_err.items()))
        print(f"max IK residual    : {self.ik.max_err*1000:.2f} mm, {math.degrees(self.ik.max_ang):.2f} deg")
        if self.ctrl.steps:
            print(f"torque saturation  : {100.0*self.ctrl.sat_steps/self.ctrl.steps:.1f}% of steps (commands clipped)")
            if self.force_sat:
                dt = self.dt
                print("joints at force limit: " + ", ".join(
                    f"{n} {c*dt:.2f}s" for n, c in sorted(self.force_sat.items(), key=lambda x: -x[1])[:6]))
            else:
                print("joints at force limit: none")
            if self.touch:
                print("unwanted contacts  : " + ", ".join(
                    f"{k} {c*self.dt:.2f}s" for k, c in sorted(self.touch.items(), key=lambda x: -x[1])[:6]))
            else:
                print("unwanted contacts  : none")


# ----------------------------------------------------------------------------

def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Unitree G1 tosses an apple between its hands (MuJoCo).")
    ap.add_argument("--scene", default=str(here / "scene.xml"), help="scene XML (default: scene.xml next to main.py)")
    ap.add_argument("--tosses", type=int, default=8, help="number of tosses, 0 = forever (default 8)")
    ap.add_argument("--flight-time", type=float, default=FLIGHT_TIME, help="airtime per toss in s (default 0.5)")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed, e.g. 0.3 for slow motion")
    ap.add_argument("--flip-palm", action="store_true",
                    help="use if the apple sits on the BACK of the hand instead of the palm")
    ap.add_argument("--headless", action="store_true", help="run without a window and print a summary")
    ap.add_argument("--max-time", type=float, default=120.0, help="headless: stop after this sim time")
    args = ap.parse_args()

    scene = Path(args.scene).resolve()
    if not scene.exists():
        sys.exit(f"Scene file not found: {scene}")
    if args.flight_time < 0.4:
        sys.exit("--flight-time must be at least 0.4 s (the catching hand needs time to get there)")

    show = AppleToss(scene, args.tosses, args.flight_time, flip_palm=args.flip_palm)

    if args.headless:
        t0 = time.perf_counter()
        while show.d.time < args.max_time:
            show.step()
            if show.finished:
                break
        print(f"\nsimulated {show.d.time:.1f} s in {time.perf_counter()-t0:.1f} s wall time")
        show.summary()
        return

    import mujoco.viewer
    m, d = show.m, show.d
    with mujoco.viewer.launch_passive(m, d) as viewer:
        mid = 0.5 * (show.station["left"] + show.station["right"])
        viewer.cam.lookat[:] = mid + np.array([0.0, 0.0, 0.05])
        viewer.cam.distance = 1.9
        viewer.cam.azimuth = math.degrees(show.yaw) + 180.0 - 30.0
        viewer.cam.elevation = -12.0

        frame = 1.0 / 60.0
        max_steps = max(1, int(4 * frame / show.dt))
        wall0, sim0 = time.perf_counter(), d.time
        summary_done = False
        while viewer.is_running():
            f0 = time.perf_counter()
            target = sim0 + (f0 - wall0) * args.speed
            n = 0
            with viewer.lock():
                while d.time < target and n < max_steps:
                    show.step()
                    n += 1
            if n == max_steps:                 # running slower than real time: resync clock
                wall0, sim0 = time.perf_counter(), d.time
            viewer.sync()
            if show.finished and not summary_done:
                show.summary()
                print("(close the viewer window to exit)")
                summary_done = True
            rest = frame - (time.perf_counter() - f0)
            if rest > 0:
                time.sleep(rest)


if __name__ == "__main__":
    main()