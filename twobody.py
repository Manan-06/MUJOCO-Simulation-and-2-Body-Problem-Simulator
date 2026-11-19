import math
import pygame

# ==========================================
# 1. PHYSICS ENGINE & CLASSES
# ==========================================

class Body:
    def __init__(self, mass, x, y, radius, color=(100, 200, 255), fixed=False):
        self.mass = mass
        self.position = [x, y]
        self.velocity = [0.0, 0.0]
        self.radius = radius
        self.color = color
        self.trail = []
        self.fixed = fixed
        self.last_force = [0.0, 0.0]

def calculate_gravity_force(body_a, body_b, G):
    dx = body_b.position[0] - body_a.position[0]
    dy = body_b.position[1] - body_a.position[1]
    
    act_dist = math.sqrt(dx**2 + dy**2)
    if act_dist == 0:
        return [0.0, 0.0]
        
    calc_dist = max(act_dist, body_a.radius + body_b.radius)
    gravforce = (G * body_a.mass * body_b.mass) / (calc_dist**2)
    
    fx = gravforce * (dx / act_dist)
    fy = gravforce * (dy / act_dist)
    return [fx, fy]

def update_body(body, force, dt):
    if body.fixed:
        body.velocity = [0.0, 0.0]
        return

    acceleration_x = force[0] / body.mass
    acceleration_y = force[1] / body.mass

    body.velocity[0] += acceleration_x * dt
    body.velocity[1] += acceleration_y * dt
    body.position[0] += body.velocity[0] * dt
    body.position[1] += body.velocity[1] * dt

def handle_collisions(bodies):
    merged = set()
    new_bodies = []
    
    for i in range(len(bodies)):
        if i in merged: continue
        for j in range(i + 1, len(bodies)):
            if j in merged: continue
            
            b1, b2 = bodies[i], bodies[j]
            dist = math.sqrt((b1.position[0]-b2.position[0])**2 + (b1.position[1]-b2.position[1])**2)
            
            if dist < (b1.radius + b2.radius):
                merged.add(i)
                merged.add(j)
                
                total_mass = b1.mass + b2.mass
                new_vx = (b1.mass * b1.velocity[0] + b2.mass * b2.velocity[0]) / total_mass
                new_vy = (b1.mass * b1.velocity[1] + b2.mass * b2.velocity[1]) / total_mass
                new_x = (b1.mass * b1.position[0] + b2.mass * b2.position[0]) / total_mass
                new_y = (b1.mass * b1.position[1] + b2.mass * b2.position[1]) / total_mass
                
                new_radius = max(5, int(math.sqrt(total_mass)))
                heavier = b1 if b1.mass >= b2.mass else b2
                new_fixed = b1.fixed or b2.fixed 
                
                if new_fixed:
                    new_vx, new_vy = 0.0, 0.0
                    fixed_body = b1 if b1.fixed else b2
                    new_x, new_y = fixed_body.position
                        
                new_b = Body(total_mass, new_x, new_y, new_radius, color=heavier.color, fixed=new_fixed)
                new_b.velocity = [new_vx, new_vy]
                new_b.trail = heavier.trail.copy() 
                
                new_bodies.append(new_b)
                break 
                
        if i not in merged:
            new_bodies.append(bodies[i])
            
    return new_bodies

def get_dynamic_trajectory(start_pos, velocity, mass, current_bodies, G, steps=250, dt=0.5, ignore_body=None):
    temp_bodies = []
    ghost = None
    
    for b in current_bodies:
        if b == ignore_body:
            ghost = Body(mass, start_pos[0], start_pos[1], max(5, int(math.sqrt(mass))), fixed=b.fixed)
            ghost.velocity = list(velocity)
            temp_bodies.append(ghost)
        else:
            tb = Body(b.mass, b.position[0], b.position[1], b.radius, fixed=b.fixed)
            tb.velocity = list(b.velocity)
            temp_bodies.append(tb)
            
    if not ghost and not ignore_body:
        ghost = Body(mass, start_pos[0], start_pos[1], max(5, int(math.sqrt(mass))))
        ghost.velocity = list(velocity)
        temp_bodies.append(ghost)
        
    points = []
    for _ in range(steps):
        force_net = [[0.0, 0.0] for _ in temp_bodies]
        for i in range(len(temp_bodies)):
            for j in range(len(temp_bodies)):
                if i != j:
                    f = calculate_gravity_force(temp_bodies[i], temp_bodies[j], G)
                    force_net[i][0] += f[0]
                    force_net[i][1] += f[1]
                    
        for i in range(len(temp_bodies)):
            update_body(temp_bodies[i], force_net[i], dt)
            
        ghost_survived = True
        for b in temp_bodies:
            if b != ghost:
                dist = math.sqrt((b.position[0]-ghost.position[0])**2 + (b.position[1]-ghost.position[1])**2)
                if dist < (b.radius + ghost.radius):
                    ghost_survived = False
                    break
                    
        if not ghost_survived: break
        points.append((ghost.position[0], ghost.position[1]))
        
    return points

def calculate_added_velocity(drag_start, world_mouse, slingshot_mult, body, keys):
    vx_add = (drag_start[0] - world_mouse[0]) * slingshot_mult
    vy_add = (drag_start[1] - world_mouse[1]) * slingshot_mult
    
    if keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]:
        bx, by = body.velocity
        v_mag_sq = bx**2 + by**2
        if v_mag_sq > 0:
            dot_product = (vx_add * bx) + (vy_add * by)
            vx_add = (dot_product / v_mag_sq) * bx
            vy_add = (dot_product / v_mag_sq) * by
        else:
            if abs(vx_add) > abs(vy_add): vy_add = 0.0
            else: vx_add = 0.0
            
    return vx_add, vy_add

# ==========================================
# 2. UI WIDGETS & CAMERA HELPERS
# ==========================================

def world_to_screen(wx, wy, cx, cy, z, w, h):
    return (int((wx - cx) * z + w / 2), int((wy - cy) * z + h / 2))

def screen_to_world(sx, sy, cx, cy, z, w, h):
    return ((sx - w / 2) / z + cx, (sy - h / 2) / z + cy)

class Slider:
    def __init__(self, x, y, w, h, min_val, max_val, start_val, label):
        self.rect = pygame.Rect(x, y, w, h)
        self.min_val = min_val
        self.max_val = max_val
        self.val = start_val
        self.label = label
        self.knob_rect = pygame.Rect(x, y - 5, 10, h + 10)
        self.update_knob()
        self.dragging = False

    def update_knob(self):
        ratio = (self.val - self.min_val) / (self.max_val - self.min_val)
        self.knob_rect.centerx = self.rect.x + (self.rect.w * ratio)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.knob_rect.collidepoint(event.pos) or self.rect.collidepoint(event.pos):
                self.dragging = True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            mouse_x = max(self.rect.left, min(event.pos[0], self.rect.right))
            ratio = (mouse_x - self.rect.left) / self.rect.w
            self.val = self.min_val + ratio * (self.max_val - self.min_val)
            self.update_knob()
            return True
        return False

    def draw(self, surface, font):
        pygame.draw.rect(surface, (100, 100, 100), self.rect, border_radius=3)
        pygame.draw.rect(surface, (200, 200, 200), self.knob_rect, border_radius=5)
        text = font.render(f"{self.label}: {self.val:.2f}x", True, (255, 255, 255))
        surface.blit(text, (self.rect.x, self.rect.y - 20))

# ==========================================
# 3. PYGAME INITIALIZATION & LOOP
# ==========================================

pygame.init()
pygame.font.init()
WIDTH, HEIGHT = 1200, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Advanced Orbital Mechanics Simulator")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Consolas", 14)
title_font = pygame.font.SysFont("Consolas", 18, bold=True)

G_CONSTANT = 0.5 
BASE_DT = 0.1
bodies = []

# Camera & Viewport System
camera_x, camera_y = 0.0, 0.0
zoom = 1.0
panning_camera = False

# FIX: Expanded UI panel and moved slider down to prevent text overlap
time_slider = Slider(20, 285, 200, 8, 0.0, 50.0, 1.0, "Time Scale")
ui_panel = pygame.Surface((310, 320))
ui_panel.set_alpha(200)
ui_panel.fill((30, 30, 40))

# Interaction States
creating_body = False
modifying_body = None
pinned_body = None  
hovered_body = None
drag_start_pos_world = (0, 0)
new_body_mass = 50
slingshot_multiplier = 0.05

def reset_scene():
    global bodies, camera_x, camera_y, zoom, pinned_body
    camera_x, camera_y = 0.0, 0.0
    zoom = 1.0
    pinned_body = None
    bodies = [Body(mass=5000, x=0, y=0, radius=30, color=(255, 200, 50), fixed=True)]

def draw_grid(surface, cam_x, cam_y, z):
    grid_size = 50 * z
    if grid_size < 10: return 
    
    origin_screen_x = (-cam_x * z) + WIDTH / 2
    origin_screen_y = (-cam_y * z) + HEIGHT / 2
    
    offset_x = origin_screen_x % grid_size
    offset_y = origin_screen_y % grid_size
    
    for x in range(int(offset_x), WIDTH, int(grid_size)):
        pygame.draw.line(surface, (25, 25, 30), (x, 0), (x, HEIGHT))
    for y in range(int(offset_y), HEIGHT, int(grid_size)):
        pygame.draw.line(surface, (25, 25, 30), (0, y), (WIDTH, y))

reset_scene()
running = True

while running:
    mouse_x, mouse_y = pygame.mouse.get_pos()
    world_mouse_x, world_mouse_y = screen_to_world(mouse_x, mouse_y, camera_x, camera_y, zoom, WIDTH, HEIGHT)
    
    hovered_body = None
    for b in bodies:
        dist_world = math.sqrt((world_mouse_x - b.position[0])**2 + (world_mouse_y - b.position[1])**2)
        if dist_world <= b.radius + (10 / zoom):
            hovered_body = b
            break
    
    keys = pygame.key.get_pressed()
    cam_speed = 10 / zoom 
    if keys[pygame.K_w] or keys[pygame.K_UP]: camera_y -= cam_speed
    if keys[pygame.K_s] or keys[pygame.K_DOWN]: camera_y += cam_speed
    if keys[pygame.K_a] or keys[pygame.K_LEFT]: camera_x -= cam_speed
    if keys[pygame.K_d] or keys[pygame.K_RIGHT]: camera_x += cam_speed
    
    # 1. HANDLE EVENTS
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            
        if time_slider.handle_event(event):
            continue 
            
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                time_slider.val = 0.0 if time_slider.val > 0 else 1.0
                time_slider.update_knob()
            elif event.key == pygame.K_c:
                reset_scene()
            elif event.key == pygame.K_t:      
                for b in bodies: b.trail.clear()
            elif event.key == pygame.K_DELETE or event.key == pygame.K_BACKSPACE:
                if hovered_body:
                    if hovered_body == pinned_body:
                        pinned_body = None
                    bodies.remove(hovered_body)
                    hovered_body = None
                
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1: 
                if hovered_body:
                    if keys[pygame.K_LSHIFT]:
                        modifying_body = hovered_body
                        drag_start_pos_world = (world_mouse_x, world_mouse_y)
                    else:
                        pinned_body = hovered_body  
                elif not ui_panel.get_rect().collidepoint(event.pos):
                    pinned_body = None              
                    creating_body = True
                    drag_start_pos_world = (world_mouse_x, world_mouse_y)
                    
            elif event.button == 3: 
                if hovered_body:
                    hovered_body.fixed = not hovered_body.fixed
                    hovered_body.velocity = [0.0, 0.0]
                    
            elif event.button == 2: 
                panning_camera = True
                        
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                if creating_body:
                    creating_body = False
                    vx = (drag_start_pos_world[0] - world_mouse_x) * slingshot_multiplier
                    vy = (drag_start_pos_world[1] - world_mouse_y) * slingshot_multiplier
                    radius = max(5, int(math.sqrt(new_body_mass)))
                    new_b = Body(mass=new_body_mass, x=drag_start_pos_world[0], y=drag_start_pos_world[1], radius=radius)
                    new_b.velocity = [vx, vy]
                    bodies.append(new_b)
                elif modifying_body:
                    vx_add, vy_add = calculate_added_velocity(drag_start_pos_world, (world_mouse_x, world_mouse_y), slingshot_multiplier, modifying_body, keys)
                    modifying_body.velocity[0] += vx_add
                    modifying_body.velocity[1] += vy_add
                    modifying_body = None
                    
            elif event.button == 2:
                panning_camera = False
                
        elif event.type == pygame.MOUSEMOTION:
            if panning_camera:
                camera_x -= event.rel[0] / zoom
                camera_y -= event.rel[1] / zoom
                
        elif event.type == pygame.MOUSEWHEEL:
            if hovered_body:
                hovered_body.mass = max(10, hovered_body.mass + event.y * 50)
                hovered_body.radius = max(5, int(math.sqrt(hovered_body.mass)))
            elif creating_body:
                new_body_mass = max(10, new_body_mass + event.y * 50)
            else:
                zoom_factor = 1.1 if event.y > 0 else (1 / 1.1)
                new_zoom = max(0.01, min(50.0, zoom * zoom_factor))
                
                camera_x = world_mouse_x - (mouse_x - WIDTH / 2) / new_zoom
                camera_y = world_mouse_y - (mouse_y - HEIGHT / 2) / new_zoom
                zoom = new_zoom

    # 2. UPDATE PHYSICS
    if time_slider.val > 0 and not modifying_body:
        
        # DYNAMIC SUB-STEPPING: 
        # When time speed increases, we must calculate more physics steps per visual frame
        # otherwise orbits break geometry and teleport into space.
        dynamic_steps = max(5, int(time_slider.val * 3))
        dynamic_steps = min(150, dynamic_steps) # Cap at 150 to prevent game freezing
        
        dt = (BASE_DT * time_slider.val) / dynamic_steps
        
        for step_idx in range(dynamic_steps):
            force_net = [[0.0, 0.0] for _ in bodies]
            
            for i in range(len(bodies)):
                for j in range(len(bodies)):
                    if i != j: 
                        f = calculate_gravity_force(bodies[i], bodies[j], G_CONSTANT)
                        force_net[i][0] += f[0]
                        force_net[i][1] += f[1]
            
            for i in range(len(bodies)):
                bodies[i].last_force = force_net[i]
                update_body(bodies[i], force_net[i], dt)
                
            bodies = handle_collisions(bodies)
                
            # Only record the trail once per visual frame to keep the game running fast
            if step_idx == 0:
                for b in bodies:
                    if not b.fixed:
                        b.trail.append((int(b.position[0]), int(b.position[1])))
                        if len(b.trail) > 200:
                            b.trail.pop(0)

    if pinned_body and pinned_body not in bodies:
        pinned_body = None

    # 3. DRAW TO SCREEN
    screen.fill((15, 15, 20)) 
    draw_grid(screen, camera_x, camera_y, zoom)

    for b in bodies:
        if len(b.trail) > 1:
            screen_trail = [world_to_screen(px, py, camera_x, camera_y, zoom, WIDTH, HEIGHT) for px, py in b.trail]
            pygame.draw.lines(screen, b.color, False, screen_trail, 1)
            
        x_draw, y_draw = world_to_screen(b.position[0], b.position[1], camera_x, camera_y, zoom, WIDTH, HEIGHT)
        radius_draw = max(1, int(b.radius * zoom))
        
        if b.fixed:
            pygame.draw.circle(screen, (255, 255, 255), (x_draw, y_draw), radius_draw + 2, 1)
        pygame.draw.circle(screen, b.color, (x_draw, y_draw), radius_draw)

    # Draw Creation Slingshot
    if creating_body:
        vx = (drag_start_pos_world[0] - world_mouse_x) * slingshot_multiplier
        vy = (drag_start_pos_world[1] - world_mouse_y) * slingshot_multiplier
        start_screen = world_to_screen(drag_start_pos_world[0], drag_start_pos_world[1], camera_x, camera_y, zoom, WIDTH, HEIGHT)
        pygame.draw.line(screen, (150, 150, 150), start_screen, (mouse_x, mouse_y), 2)
        
        proj = get_dynamic_trajectory(drag_start_pos_world, [vx, vy], new_body_mass, bodies, G_CONSTANT)
        if len(proj) > 1:
            screen_proj = [world_to_screen(px, py, camera_x, camera_y, zoom, WIDTH, HEIGHT) for px, py in proj]
            pygame.draw.lines(screen, (100, 255, 100), False, screen_proj, 1)
            
        radius_draw = max(1, int(max(5, int(math.sqrt(new_body_mass))) * zoom))
        pygame.draw.circle(screen, (100, 255, 100), start_screen, radius_draw, 1)
        screen.blit(font.render(f"Mass: {new_body_mass}", True, (255,255,255)), (start_screen[0] + 15, start_screen[1] - 15))

    # Draw Modification Slingshot
    if modifying_body:
        vx_add, vy_add = calculate_added_velocity(drag_start_pos_world, (world_mouse_x, world_mouse_y), slingshot_multiplier, modifying_body, keys)
        total_vx = modifying_body.velocity[0] + vx_add
        total_vy = modifying_body.velocity[1] + vy_add
        
        end_world_x = drag_start_pos_world[0] - (vx_add / slingshot_multiplier)
        end_world_y = drag_start_pos_world[1] - (vy_add / slingshot_multiplier)
        start_screen = world_to_screen(drag_start_pos_world[0], drag_start_pos_world[1], camera_x, camera_y, zoom, WIDTH, HEIGHT)
        end_screen = world_to_screen(end_world_x, end_world_y, camera_x, camera_y, zoom, WIDTH, HEIGHT)
        
        pygame.draw.line(screen, (255, 150, 50), start_screen, end_screen, 2)
        
        proj = get_dynamic_trajectory(modifying_body.position, [total_vx, total_vy], modifying_body.mass, bodies, G_CONSTANT, ignore_body=modifying_body)
        if len(proj) > 1:
            screen_proj = [world_to_screen(px, py, camera_x, camera_y, zoom, WIDTH, HEIGHT) for px, py in proj]
            pygame.draw.lines(screen, (255, 200, 100), False, screen_proj, 1)

    # 4. DRAW HUD ELEMENTS
    screen.blit(ui_panel, (10, 10))
    screen.blit(title_font.render("SIMULATOR CONTROLS", True, (255, 255, 255)), (20, 20))
    controls = [
        "DRAG SPACE : Add Body",
        "SHIFT+DRAG : Add Velocity",
        " + CTRL    : Snap Prograde",
        "LEFT-CLICK : Pin Telemetry",
        "RIGHT-CLICK: Fix in Place",
        "MID-CLICK  : Pan Camera",
        "SCROLL     : Zoom / Mass",
        "DEL/BACKSPC: Delete Body",
        "SPACEBAR   : Pause Time",
        "T KEY      : Clear Trails",
        "C KEY      : Clear Screen"
    ]
    for i, text in enumerate(controls):
        screen.blit(font.render(text, True, (200, 200, 200)), (20, 45 + (i * 18)))
    time_slider.draw(screen, font)

    # Top Right Pinned Target HUD
    if pinned_body:
        bx, by = pinned_body.velocity
        fx, fy = pinned_body.last_force
        speed = math.sqrt(bx**2 + by**2)
        force_mag = math.sqrt(fx**2 + fy**2)
        
        panel_w, panel_h = 220, 110
        hud_surface = pygame.Surface((panel_w, panel_h))
        hud_surface.set_alpha(200)
        hud_surface.fill((30, 30, 40))
        screen.blit(hud_surface, (WIDTH - panel_w - 10, 10))
        
        stats = [
            "--- PINNED TARGET ---",
            f"Mass : {pinned_body.mass:.0f}",
            f"Speed: {speed:.2f}",
            f"Force: {force_mag:.2f}",
            f"State: {'FIXED' if pinned_body.fixed else 'Dynamic'}"
        ]
        for i, text in enumerate(stats):
            color = (100, 255, 100) if i == 0 else (255, 255, 255)
            screen.blit(font.render(text, True, color), (WIDTH - panel_w, 20 + (i * 18)))

    # Hover Vectors 
    if hovered_body and not pinned_body:
        bx, by = world_to_screen(hovered_body.position[0], hovered_body.position[1], camera_x, camera_y, zoom, WIDTH, HEIGHT)
        vx, vy = hovered_body.velocity
        pygame.draw.line(screen, (50, 255, 50), (bx, by), (bx + vx*10*zoom, by + vy*10*zoom), 2)
        
        r_draw = max(1, int(hovered_body.radius * zoom))
        screen.blit(font.render(f"Mass: {hovered_body.mass:.0f}", True, (255, 255, 255)), (bx + r_draw + 10, by - 20))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()