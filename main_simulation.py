import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

sim_params = {
    'SPEED_SCALE': 2.0,   
    'CURVATURE_K': 12.98,
    'GM_STRENGTH': 5.0,
    'ADAPTIVE_K': 0.01,
    'PLANCK_H': 0.058
}

STEPS_PER_FRAME = 50
MAX_CLOUD_POINTS = 6000

def get_derivatives(state, params):
    d_state = np.zeros(9)
    q = state[0:4]; v = state[4:8]
    
    f_spring = -params['CURVATURE_K'] * q
    gm = params['GM_STRENGTH']
    f_spin = np.array([
        gm * (v[1]*q[2] - v[2]*q[1]),
        gm * (v[2]*q[0] - v[0]*q[2]),
        gm * (v[0]*q[1] - v[1]*q[0]),
        0.0 
    ])
    
    total_force = f_spring + f_spin
    v_sq = np.dot(v, v)
    q_dot_f = np.dot(q, total_force)
    constraint = -(v_sq + q_dot_f) * q
    
    d_state[0:4] = v
    d_state[4:8] = total_force + constraint
    d_state[8] = (0.5 * v_sq) / params['PLANCK_H']
    
    return d_state

def step_physics(state, params):
    if len(state) < 9: state = np.append(state, 0.0)
    
    d1 = get_derivatives(state, params)
    dt = params['ADAPTIVE_K'] * np.linalg.norm(state[0:4])
    dt = max(1e-4, min(0.02, dt))

    state_new = state + d1 * dt
    
    q = state_new[0:4]
    q /= np.linalg.norm(q)
    state_new[0:4] = q
    
    v = state_new[4:8]
    v = v - np.dot(v, q) * q
    state_new[4:8] = v
    
    return state_new

def hopf_projection(q):
    x = 2*(q[0]*q[2] + q[1]*q[3])
    y = 2*(q[1]*q[2] - q[0]*q[3])
    z = q[0]**2 + q[1]**2 - q[2]**2 - q[3]**2
    return np.array([x, y, z])

state = np.array([1.0, 0.0, 0.0, 0.0,  0.0, 2.0, 1.5, 0.5, 0.0])
state[0:4] /= np.linalg.norm(state[0:4])

fig = plt.figure(figsize=(10, 10), facecolor='black')
ax = fig.add_subplot(111, projection='3d', facecolor='black')
ax.set_title(f"Geometric Hidrogen\nK={sim_params['CURVATURE_K']} | Virial=0.500", color='lime', fontsize=14)
limit = 2.0
ax.set_xlim(-limit, limit); ax.set_ylim(-limit, limit); ax.set_zlim(-limit, limit)
ax.axis('off')

scat = ax.scatter([], [], [], s=1, c='lime', alpha=0.6)
cloud_x, cloud_y, cloud_z = [], [], []

def animate(frame):
    global state, cloud_x, cloud_y, cloud_z
    
    for _ in range(STEPS_PER_FRAME):
        state = step_physics(state, sim_params)
        
        phase = state[8]
        if np.cos(phase)**4 > 0.1:
            p = hopf_projection(state[0:4]) * sim_params['SPEED_SCALE']
            cloud_x.append(p[0])
            cloud_y.append(p[1])
            cloud_z.append(p[2])
            
    if len(cloud_x) > MAX_CLOUD_POINTS:
        cloud_x = cloud_x[-MAX_CLOUD_POINTS:]
        cloud_y = cloud_y[-MAX_CLOUD_POINTS:]
        cloud_z = cloud_z[-MAX_CLOUD_POINTS:]
        
    scat._offsets3d = (cloud_x, cloud_y, cloud_z)
    ax.view_init(elev=30, azim=frame*0.4)
    return [scat]

ani = FuncAnimation(fig, animate, interval=1, cache_frame_data=False)
plt.show()