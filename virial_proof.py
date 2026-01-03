import numpy as np

sim_params = {
    'CURVATURE_K': 12.98,
    'GM_STRENGTH': 5.0,
    'ADAPTIVE_K': 0.01,
    'PLANCK_H': 0.058
}

def get_derivatives(state, params):
    d_state = np.zeros(9)
    q = state[0:4]
    v = state[4:8]
    
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
    
    return d_state, np.dot(q, q)

def step_physics(state, params):
    if len(state) < 9: state = np.append(state, 0.0)
    _, dist_sq = get_derivatives(state, params)
    dt = params['ADAPTIVE_K'] * np.sqrt(dist_sq)
    dt = max(1e-5, min(0.01, dt))
    
    k1, _ = get_derivatives(state, params)
    k2, _ = get_derivatives(state + 0.5*dt*k1, params)
    k3, _ = get_derivatives(state + 0.5*dt*k2, params)
    k4, _ = get_derivatives(state + dt*k3, params)
    
    new_state = state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    
    q = new_state[0:4]
    if np.linalg.norm(q) > 1e-9:
        q = q / np.linalg.norm(q)
        new_state[0:4] = q
        v = new_state[4:8]
        new_state[4:8] = v - np.dot(v, q) * q
        
    return new_state, dt

print("--- Start Metrics ---")
print(f"Params: K={sim_params['CURVATURE_K']}, Spin={sim_params['GM_STRENGTH']}")

state = np.array([1.0, 0.0, 0.0, 0.0,  0.0, 2.0, 1.5, 0.5, 0.0])
state[0:4] /= np.linalg.norm(state[0:4])
total_time = 0
orbit_count = 0
total_energy = 0
total_spin_rot = 0
last_phase = 0
samples = 0

for i in range(50000):
    state, dt = step_physics(state, sim_params)
    total_time += dt
    v = state[4:8]
    v_sq = np.dot(v, v)
    kinetic = 0.5 * v_sq
    potential = 0.5 * sim_params['CURVATURE_K'] * 1.0 # q^2 = 1
    total_e = kinetic + potential
    
    total_energy += total_e
    samples += 1

    current_phase = state[8]
    delta_phase = current_phase - last_phase
    last_phase = current_phase
    
    total_spin_rot += delta_phase

    if i % 100 == 0:
        pass

avg_energy = total_energy / samples
avg_kinetic = (avg_energy - (0.5 * sim_params['CURVATURE_K']))
virial_ratio = avg_kinetic / (0.5 * sim_params['CURVATURE_K'])
action_density = total_spin_rot / total_time
print(f"1. avg_energy: {avg_energy:.6f}")
print(f"3. Virial Ratio(<T>/<V>): {virial_ratio:.6f}")
print(f"   cumulated action (Fase): {total_spin_rot:.4f}")
print(f"   Simulated time: {total_time:.4f}")
print(f"   Action/time ratio: {action_density:.4f}")
ratio_forces = sim_params['CURVATURE_K'] / sim_params['GM_STRENGTH']
print(f"   Ratio K/Spin: {ratio_forces:.4f}")
print(f"   1 / (Avg. Kinetic): {1.0/avg_kinetic:.6f}")