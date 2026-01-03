import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy.special import sph_harm, genlaguerre

sim_params = {
    'SPEED_SCALE': 2.5,   
    'CURVATURE_K': 8.0,
    'GM_STRENGTH': 3.5,
    'ADAPTIVE_K': 0.01
}

STEPS_PER_FRAME = 30
MAX_POINTS = 4000

def hopf_projection_scaled(q, scale):
    norm = np.linalg.norm(q)
    if norm < 1e-9: return np.zeros(3)
    q = q / norm
    x = 2*(q[0]*q[2] + q[1]*q[3])
    y = 2*(q[1]*q[2] - q[0]*q[3])
    z = q[0]**2 + q[1]**2 - q[2]**2 - q[3]**2
    return np.array([x, y, z]) * scale

def get_derivatives(state, params):
    d_state = np.zeros_like(state)
    k_spring = params['CURVATURE_K']
    gm_k = params['GM_STRENGTH']
    
    q = state[0:4]; v = state[4:8]
    
    f_spring = -k_spring * q
    f_spin = np.zeros(4)
    f_spin[0] = gm_k * (v[1]*q[2] - v[2]*q[1])
    f_spin[1] = gm_k * (v[2]*q[0] - v[0]*q[2])
    f_spin[2] = gm_k * (v[0]*q[1] - v[1]*q[0])
    
    total_force = f_spring + f_spin
    v_sq = np.dot(v, v); q_dot_f = np.dot(q, total_force)
    constraint = -(v_sq + q_dot_f) * q
    
    d_state[0:4] = v
    d_state[4:8] = total_force + constraint
    return d_state, np.dot(q, q)

def step_physics(state, params):
    _, dist_sq = get_derivatives(state, params)
    dt = params['ADAPTIVE_K'] * np.sqrt(dist_sq)
    dt = max(1e-4, min(0.05, dt))
    k1, _ = get_derivatives(state, params)
    k2, _ = get_derivatives(state + 0.5*dt*k1, params)
    k3, _ = get_derivatives(state + 0.5*dt*k2, params)
    k4, _ = get_derivatives(state + dt*k3, params)
    new_state = state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    new_state[0:4] /= np.linalg.norm(new_state[0:4])
    return new_state

def hydrogen_wavefunction(n, l, m, x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2) + 1e-9
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    
    rho = 2 * r / (n * 1.5)
    
    L_val = genlaguerre(n - l - 1, 2 * l + 1)(rho)
    R_nl = np.exp(-rho / 2) * (rho ** l) * L_val
    Y_lm = sph_harm(m, l, phi, theta)
    return np.abs(R_nl * Y_lm)**2

def generate_quantum_cloud(n, l, m, num_points=3000):
    points_x, points_y, points_z = [], [], []
    batch_size = 50000 
    search_box = 25.0
    scan = np.linspace(1, 15, 100)
    peak_scan = hydrogen_wavefunction(n, l, m, scan, scan, scan*0.5)
    max_prob = np.max(peak_scan)
    limit_low = 0.20 * max_prob
    limit_high = 0.30 * max_prob
    
    while len(points_x) < num_points:
        x = np.random.uniform(-search_box, search_box, batch_size)
        y = np.random.uniform(-search_box, search_box, batch_size)
        z = np.random.uniform(-search_box, search_box, batch_size)
        
        probs = hydrogen_wavefunction(n, l, m, x, y, z)
        mask = (probs > limit_low) & (probs < limit_high)
        
        points_x.extend(x[mask])
        points_y.extend(y[mask])
        points_z.extend(z[mask])
        
    return points_x[:num_points], points_y[:num_points], points_z[:num_points]

state_4d = np.array([1.0, 0.0, 0.0, 0.0,  0.0, 1.8, 1.2, 0.5])
state_4d[0:4] /= np.linalg.norm(state_4d[0:4])

qm_x, qm_y, qm_z = generate_quantum_cloud(n=3, l=2, m=1, num_points=MAX_POINTS)

fig = plt.figure(figsize=(16, 8), facecolor='#101010')

ax1 = fig.add_subplot(121, projection='3d', facecolor='#101010')
ax1.set_title("Theory (4D Springs)", color='cyan', fontsize=14)
ax1.set_xlim(-4, 4); ax1.set_ylim(-4, 4); ax1.set_zlim(-4, 4); ax1.axis('off')
sim_scat = ax1.scatter([], [], [], s=2, c='cyan', alpha=0.5)

ax2 = fig.add_subplot(122, projection='3d', facecolor='#101010')
ax2.set_title("SCHRÖDINGER (Real Isosurface)", color='lime', fontsize=14)
limit_qm = 20.0 
ax2.set_xlim(-limit_qm, limit_qm); ax2.set_ylim(-limit_qm, limit_qm); ax2.set_zlim(-limit_qm, limit_qm)
ax2.axis('off')

ax2.scatter(qm_x, qm_y, qm_z, s=1, c='lime', alpha=0.1)
ax2.text2D(0.05, 0.95, "n=3, l=2 (Casca)", transform=ax2.transAxes, color='lime')

sim_x, sim_y, sim_z = [], [], []

def animate(frame):
    global state_4d, sim_x, sim_y, sim_z
    
    for _ in range(STEPS_PER_FRAME):
        state_4d = step_physics(state_4d, sim_params)
        if np.random.rand() > 0.6:
            p3 = hopf_projection_scaled(state_4d[0:4], sim_params['SPEED_SCALE'])
            sim_x.append(p3[0])
            sim_y.append(p3[1])
            sim_z.append(p3[2])
    
    if len(sim_x) > MAX_POINTS:
        sim_x = sim_x[-MAX_POINTS:]
        sim_y = sim_y[-MAX_POINTS:]
        sim_z = sim_z[-MAX_POINTS:]
        
    sim_scat._offsets3d = (sim_x, sim_y, sim_z)
    
    ax1.view_init(elev=30, azim=frame*0.5)
    ax2.view_init(elev=30, azim=frame*0.5 + 45)
    
    return [sim_scat]

ani = FuncAnimation(fig, animate, interval=20, cache_frame_data=False)
plt.show()