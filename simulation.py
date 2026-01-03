import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Button
from collections import deque
from matplotlib.animation import FuncAnimation

# --- PARÂMETROS GLOBAIS ---
sim_params = {
    'SPEED_SCALE': 2.0,   
    'COUPLING': 0.0,      
    'C_LIM': 15.0         
}

# Configurações
TRAIL_LEN = 200
DT = 0.05
N_BODIES = 3

# --- 1. PROJEÇÃO ---
def hopf_projection_scaled(q, scale):
    norm = np.linalg.norm(q)
    if norm == 0: return np.zeros(3)
    q = q / norm
    x = 2*(q[0]*q[2] + q[1]*q[3])
    y = 2*(q[1]*q[2] - q[0]*q[3])
    z = q[0]**2 + q[1]**2 - q[2]**2 - q[3]**2
    return np.array([x, y, z]) * scale

# --- 2. FÍSICA ---
def physics_step(state, t, n_bodies, params):
    coupling_k = params['COUPLING']
    speed_scale = params['SPEED_SCALE']
    c_lim = params['C_LIM']
    
    d_state = np.zeros_like(state)
    
    # Pré-cálculo
    current_3d_pos = []
    for i in range(n_bodies):
        idx = i * 9 
        pos_4d = state[idx:idx+4]
        current_3d_pos.append(hopf_projection_scaled(pos_4d, speed_scale))
        
    for i in range(n_bodies):
        idx = i * 9
        pos_4d = state[idx:idx+4]
        vel_4d = state[idx+4:idx+8]
        
        # Inércia
        inertial_accel = -np.dot(vel_4d, vel_4d) * pos_4d
        
        # Gravidade
        coupling_accel_total = np.zeros(4)
        for j in range(n_bodies):
            if i == j: continue
            idx_j = j * 9
            pos_4d_j = state[idx_j:idx_j+4]
            dist_3d = np.linalg.norm(current_3d_pos[j] - current_3d_pos[i])
            
            force_mag = coupling_k / (dist_3d**2 + 0.1)
            coupling_accel_total += force_mag * (pos_4d_j - pos_4d)
            
        coupling_tangent = coupling_accel_total - np.dot(coupling_accel_total, pos_4d) * pos_4d
        total_accel = inertial_accel + coupling_tangent
        
        # Relatividade (Lag)
        speed_4d = np.linalg.norm(vel_4d)
        effective_load = speed_4d * (speed_scale / 2.0)
        
        safe_c = max(effective_load * 1.01, c_lim)
        dt_local = np.sqrt(max(0, 1 - (effective_load/safe_c)**2))
        
        d_state[idx:idx+4] = vel_4d
        d_state[idx+4:idx+8] = total_accel
        d_state[idx+8] = dt_local 
        
    return d_state

# --- 3. INICIALIZAÇÃO ---
np.random.seed(42)
initial_state = []
base_pos = [np.random.randn(4) for _ in range(N_BODIES)]
base_pos = [p/np.linalg.norm(p) for p in base_pos]
base_speeds = [1.0, 2.5, 6.0] 

for i in range(N_BODIES):
    p = base_pos[i]
    v = np.random.randn(4)
    v -= np.dot(v, p)*p
    v = (v / np.linalg.norm(v)) * base_speeds[i]
    initial_state.extend(p); initial_state.extend(v); initial_state.append(0.0)

current_state = np.array(initial_state)
total_sys_time = 0.0

trails_3d = [deque(maxlen=TRAIL_LEN) for _ in range(N_BODIES)]
history_clock = [deque(maxlen=TRAIL_LEN) for _ in range(N_BODIES)]
history_time = deque(maxlen=TRAIL_LEN)

# --- 4. INTERFACE ---
fig = plt.figure(figsize=(15, 8), facecolor='#f0f0f0')
plt.subplots_adjust(bottom=0.25) 

ax3d = fig.add_axes([0.02, 0.3, 0.45, 0.65], projection='3d') 
ax2d = fig.add_axes([0.55, 0.3, 0.40, 0.65]) 

ax3d.set_facecolor('#f0f0f0')
ax3d.set_xlim([-4, 4]); ax3d.set_ylim([-4, 4]); ax3d.set_zlim([-4, 4])
ax3d.set_title("ESPAÇO 3D (Hopf)", fontweight='bold')

ax2d.set_facecolor('white'); ax2d.grid(True, linestyle=':', alpha=0.6)
ax2d.set_title("TEMPO / LAG (Validação)", fontweight='bold')
ax2d.set_xlabel("Tempo Sistema"); ax2d.set_ylabel("Tempo Vivido")

colors = ['#0099cc', '#ff8800', '#00cc00']
labels = ['Lento', 'Médio', 'Rápido']

lines3d = [ax3d.plot([], [], [], lw=2, color=colors[i])[0] for i in range(N_BODIES)]
heads3d = [ax3d.plot([], [], [], 'o', color=colors[i], ms=6)[0] for i in range(N_BODIES)]
lines2d = [ax2d.plot([], [], lw=2, color=colors[i], label=labels[i])[0] for i in range(N_BODIES)]
ax2d.legend(loc='upper left')

txt_info = fig.text(0.5, 0.95, "", ha='center', fontsize=12, fontweight='bold', family='monospace')

# --- 5. CONTROLES (CORREÇÃO AQUI) ---
controls = [] # Lista global para segurar as referências dos botões

def make_btn(rect, text, func, color):
    ax = plt.axes(rect)
    b = Button(ax, text, color=color, hovercolor='white')
    b.on_clicked(func)
    return b

def update_txt():
    txt_info.set_text(f"VELOCIDADE: {sim_params['SPEED_SCALE']:.1f} | GRAVIDADE: {sim_params['COUPLING']:.1f} | LIMITE C: {sim_params['C_LIM']:.1f}")

def spd_dn(e): sim_params['SPEED_SCALE'] = max(0.5, sim_params['SPEED_SCALE']-0.5); update_txt()
def spd_up(e): sim_params['SPEED_SCALE'] += 0.5; update_txt()
def grv_dn(e): sim_params['COUPLING'] = max(0.0, sim_params['COUPLING']-0.5); update_txt()
def grv_up(e): sim_params['COUPLING'] += 0.5; update_txt()
def clim_dn(e): sim_params['C_LIM'] = max(1.0, sim_params['C_LIM']-1.0); update_txt()
def clim_up(e): sim_params['C_LIM'] += 1.0; update_txt()

# Adicionamos cada botão à lista global 'controls'
controls.append(make_btn([0.15, 0.1, 0.08, 0.05], '- Speed', spd_dn, '#ffdddd'))
controls.append(make_btn([0.24, 0.1, 0.08, 0.05], '+ Speed', spd_up, '#ddffdd'))

controls.append(make_btn([0.45, 0.1, 0.08, 0.05], '- Gravidade', grv_dn, '#ddddff'))
controls.append(make_btn([0.54, 0.1, 0.08, 0.05], '+ Gravidade', grv_up, '#ffffdd'))

controls.append(make_btn([0.75, 0.1, 0.08, 0.05], '- Limite C', clim_dn, '#e0e0e0'))
controls.append(make_btn([0.84, 0.1, 0.08, 0.05], '+ Limite C', clim_up, '#f0f0f0'))

update_txt()

# --- 6. ANIMAÇÃO ---
def animate(frame):
    global current_state, total_sys_time
    
    sol = odeint(physics_step, current_state, [0, DT], args=(N_BODIES, sim_params))
    current_state = sol[-1]
    total_sys_time += DT
    history_time.append(total_sys_time)
    
    max_range = 0
    
    for i in range(N_BODIES):
        idx = i * 9
        
        # 3D
        p4 = current_state[idx:idx+4]
        p3 = hopf_projection_scaled(p4, sim_params['SPEED_SCALE'])
        trails_3d[i].append(p3)
        
        arr = np.array(trails_3d[i])
        lines3d[i].set_data(arr[:,0], arr[:,1])
        lines3d[i].set_3d_properties(arr[:,2])
        heads3d[i].set_data([p3[0]], [p3[1]])
        heads3d[i].set_3d_properties([p3[2]])
        
        max_range = max(max_range, np.max(np.abs(p3)))
        
        # 2D
        clock = current_state[idx+8]
        history_clock[i].append(clock)
        lines2d[i].set_data(history_time, history_clock[i])
        
    limit = max(4.0, max_range * 1.2)
    ax3d.set_xlim(-limit, limit); ax3d.set_ylim(-limit, limit); ax3d.set_zlim(-limit, limit)
    
    if len(history_time) > 1:
        ax2d.set_xlim(history_time[0], history_time[-1] + 1)
        max_clk = max([current_state[i*9+8] for i in range(N_BODIES)])
        ax2d.set_ylim(0, max(5, max_clk * 1.2))
        
    return lines3d + heads3d + lines2d

ani = FuncAnimation(fig, animate, interval=30, cache_frame_data=False)
plt.show()