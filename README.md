# Interactive 2-Body & N-Body Orbital Mechanics Simulator

An interactive orbital mechanics simulator built from scratch in Python and Pygame for the **ME206** assignment. It demonstrates Newtonian gravity, orbital maneuvers, gravitational assists, and binary systems in real time.

## Physics & Features

* **Gravitational Engine:** Computes pairwise Newtonian gravitational forces ($F = G \frac{m_1 m_2}{r^2}$) with radius-based softening to prevent infinite acceleration singularities.
* **Numerical Integration:** Uses Semi-Implicit Euler integration with **dynamic sub-stepping** (up to 150 sub-steps per frame), preserving orbital stability even at **50x time acceleration**.
* **Perfectly Inelastic Collisions:** Merging bodies conserve linear momentum ($m_1 \vec{v}_1 + m_2 \vec{v}_2 = (m_1 + m_2)\vec{v}_{\text{new}}$) and update position to their combined center of mass.
* **Dynamic N-Body Trajectory Prediction:** Clones the active system state to project real-time ghost trajectories, accounting for moving central stars and predicted collisions.
* **Precision Maneuver Tools:** Vector projection snapping (`CTRL`) locks velocity impulses to exact prograde or retrograde directions for executing Hohmann transfers.
* **Interactive Camera & Telemetry:** World-space zooming (cursor-anchored), camera panning, and a persistent telemetry HUD for tracking mass, speed, and net force.

## Software Requirements

* **Python:** 3.8 or higher
* **Dependencies:** `pygame` (see `requirements.txt`)

## How to Run the Simulation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/](https://github.com/)<your-username>/orbital-mechanics-simulator.git
   cd orbital-mechanics-simulator
   ```

2. **(Optional) Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the simulator:**
   ```bash
   python simulator.py
   ```

## Controls

| Input | Action |
| :--- | :--- |
| **Left-Click & Drag (Empty Space)** | Spawn a new body with slingshot velocity |
| **Shift + Left-Click & Drag (Body)** | Apply a velocity impulse to an existing body |
| **Shift + Ctrl + Drag (Body)** | Snap impulse strictly to **Prograde / Retrograde** |
| **Left-Click (Body)** | Pin live telemetry (mass, speed, force) to HUD |
| **Right-Click (Body)** | Toggle body between Fixed and Dynamic |
| **Mouse Wheel (Empty Space)** | Zoom in / out (anchored to cursor) |
| **Mouse Wheel (Hover / Creating)** | Increase or decrease body mass |
| **Middle-Click Drag / WASD / Arrows** | Pan camera |
| **Delete / Backspace (Hover)** | Delete individual body |
| **Spacebar** | Pause / Resume simulation |
| **Time Scale Slider** | Adjust simulation speed (`0.0x` to `50.0x`) |
| **T** | Clear orbital trails |
| **C** | Reset scene |

## Demonstrated Concepts

The simulator is used to reproduce and visualize the following orbital mechanics phenomena:
* **Trochoidal Trajectories:** Absolute path traced by a planet orbiting a moving star.
* **Prograde & Retrograde Gravity Assists:** Momentum exchange speeding up or braking an incoming body.
* **Binary Systems:** Two equal-mass bodies orbiting a shared barycenter.
* **Gravitational Deflection:** Hyperbolic trajectory bending of a rogue body.
* **Elliptical Orbits & Hohmann Transfers:** Two-burn orbital raising using prograde-locked impulses.

# Unitree G1 Apple Toss (MuJoCo)

A MuJoCo simulation of a Unitree G1 humanoid that picks an apple up from a table, turns its palm up, tosses the apple back and forth between its hands, and puts it back on the table.

Once thrown, the apple is a free rigid body that follows a parabolic path under gravity. At the moment of release the code predicts where the apple will be when it lands, and the other hand is moved there in time to catch it.

---

## 1. Requirements

| Software | Version |
|---|---|
| Python | 3.10 or newer |
| MuJoCo (Python bindings) | 3.x (tested with a recent 3.x release) |
| NumPy | 1.24 or newer |

Install the dependencies:

```bash
pip install mujoco numpy
```

Works on Windows, Linux and macOS. On macOS, the interactive viewer must be started with `mjpython` instead of `python` (see section 4).

---

## 2. Repository structure

```
.
├── main.py          # simulation, planning and control (run this)
├── scene.xml        # scene: includes the G1, adds floor, light, table and apple
├── g1/
│   ├── g1.xml       # Unitree G1 model (29 DOF, rev 1.0)
│   └── assets/      # G1 mesh files (*.STL) used by g1.xml
└── README.md
```

`scene.xml` includes the robot with `<include file="g1/g1.xml"/>`, so keep the `g1` folder next to it. `main.py` looks for `scene.xml` in its own folder by default.

---

## 3. How to run

From the repository folder:

```bash
# Interactive 3D viewer (recommended)
python main.py

# Slow motion (30% speed), useful for watching the throw and catch
python main.py --speed 0.3

# No window: runs the full sequence and prints an event log and summary
python main.py --headless
```

### Command-line options

| Option | Default | Meaning |
|---|---|---|
| `--scene PATH` | `scene.xml` next to `main.py` | Scene file to load |
| `--tosses N` | `8` | Number of tosses; `0` = toss forever |
| `--flight-time T` | `0.5` | Time the apple spends in the air per toss, in seconds (minimum 0.4) |
| `--speed S` | `1.0` | Playback speed in the viewer |
| `--headless` | off | Run without a window and print a summary |
| `--max-time T` | `120` | Headless only: stop after this much simulated time |
| `--flip-palm` | off | Use only if the apple appears on the back of the hand |

---

## 4. Expected output

`python main.py --headless` prints the robot setup, each event, and a summary. A successful run looks like this:

```
[robot] 29 joints, 29 driven by actuators, 0 by direct joint torque; pelvis fixed: True
[robot] start pose: keyframe 'stand' (30 joints)
[check] every planned hand pose (position + palm direction) is reachable
[t=  2.95s] grasp with the right hand
[t=  5.39s] release from the right hand at 2.57 m/s
[t=  5.89s] catch #1 with the left hand
...
[t= 24.79s] apple placed back on the table by the right hand

=== summary ===
tosses caught      : 8
final state        : done
max hand tracking  : 2.82 cm (actual hand vs plan)
torque saturation  : 0.0% of steps (commands clipped)
joints at force limit: none
unwanted contacts  : none
```

In the viewer, the sequence plays once and the summary prints when it finishes. Close the window to exit.

macOS: run the viewer with `mjpython main.py` (installed with the `mujoco` package). `--headless` works with plain `python`.

---

## 5. How it works

### Sequence
1. **Reach and grasp:** the right hand approaches the apple from the side, palm facing it.
2. **Carry:** the hand lifts the apple and turns the palm up.
3. **Toss:** the hand dips, then accelerates upward and across and releases the apple at the planned velocity.
4. **Flight:** the apple flies as a free body under gravity alone.
5. **Catch:** the other palm meets the apple at its predicted position and moves with it briefly to soften the catch.
6. Steps 3 to 5 repeat, alternating hands; the apple is then put back on the table.

### Physics of the toss
The flight is projectile motion. For a hold point $S_f$ on the throwing side, a hold point $S_t$ on the catching side, and a flight time $T$:

$$S_t = S_f + v_0 T + \tfrac{1}{2} g T^2 \;\;\Rightarrow\;\; v_0 = \frac{S_t - S_f - \tfrac{1}{2} g T^2}{T}$$

The code uses the same equation in the discrete form that matches MuJoCo's time integration, so the predicted landing point agrees with the simulated one to within numerical precision. At release, the catch point is recalculated from the apple's actual position and velocity:

$$C = P_{release} + v_0 T + \tfrac{1}{2} g T^2, \qquad v_{catch} = v_0 + gT$$

The throw itself is a constant-acceleration stroke: the hand accelerates over 0.18 m to reach the release speed (about 2.6 m/s, roughly 18 m/s² of acceleration), releases, then decelerates over a short follow-through.

### Motion planning
Each hand follows a path in Cartesian space:
- **Minimum-jerk segments** for smooth reaching, carrying and returning.
- **Constant-acceleration segments** for the throw stroke and the catch.

### Inverse kinematics
Each 7-joint arm is controlled for hand **position** (3 values) and **palm direction** (2 values) every control step, using damped least-squares inverse kinematics:

$$\dot q = J^{+}(\dot x_{des} + K e) + (I - J^{+}J)\,k_{null}(q_{home} - q)$$

The extra joint freedom (the null space) gently pulls the arm toward its natural standing posture, and joint limits are enforced.

### Joint control (computed torque)
The joint targets from IK are tracked with a computed-torque controller:

$$\tau = M(q)\,\big(\ddot q_{des} + K_p (q_{des} - q) + K_d (\dot q_{des} - \dot q)\big) + C(q,\dot q) + G(q)$$

where $M$ is the mass matrix and $C + G$ are the Coriolis and gravity terms, all taken from MuJoCo. The torques are sent through the G1 model's own actuators, respecting their force limits. Gains: $K_p = 400$, $K_d = 40$ (critically damped).

### Modelling simplifications
- **Fixed pelvis:** the robot's base is held still, like a robot on a test stand, so no balance controller is needed.
- **Virtual grasp:** the G1 rubber hand cannot close around a sphere, so while held the apple is attached to the palm. After release it is a normal physical body that collides with the table and floor.
- **Collisions:** the arms do not collide with the robot's own body. MuJoCo collides meshes using their convex hulls, and the G1's torso hull would otherwise trap the arms. The arms still collide with the table and floor.

---

## 6. Tunable parameters

At the top of `main.py`:

| Parameter | Default | Effect |
|---|---|---|
| `FLIGHT_TIME` | 0.50 s | Air time per toss (higher = higher toss) |
| `STATION` | (-0.08, 0.18, 0.22) m | Where each hand holds the apple, relative to its starting position on the table |
| `THROW_STROKE` | 0.18 m | Length of the throwing stroke |
| `FOLLOW_THROUGH` | 0.15 m | Braking distance after release |
| `KP`, `KD` | 400, 40 | Joint controller gains (keep KD at about 2·√KP) |

---

## 7. Troubleshooting

| Problem | Fix |
|---|---|
| `Scene file not found` | Run from the repository folder, or pass `--scene path/to/scene.xml` |
| Mesh files not found | Make sure `g1/assets/` contains the `.STL` files referenced in `g1.xml` |
| Apple sits on the back of the hand | Run with `--flip-palm` |
| `launch_passive` error on macOS | Use `mjpython main.py` |
| A catch fails | Increase `--flight-time` (e.g. `0.6`) to give the catching hand more time |