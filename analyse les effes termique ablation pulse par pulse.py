import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image
import matplotlib

# Paramètres de l'expérience
w0 = 25e-6  # Rayon du spot (m)
f_rep = 800e3  # Fréquence de répétition (Hz)
v = 100.0  # Vitesse de balayage (m/s)
F = 0.37  # Fluence (J/cm²)
eta_res = 0.4  # Fraction de chaleur résiduelle
T_offset = 0  # Température initiale (°C)
rho = 7920  # Densité (kg/m³)
c = 500  # Capacité calorifique (J/kg·K)
kappa = 3e-7  # Diffusivité thermique (m²/s)

# Calcul de l'énergie résiduelle
spot_area = np.pi * w0**2
E_p = F * 1e4 * spot_area  # J
E_res = eta_res * E_p

# Température initiale
penetration_depth = 100e-6
volume = spot_area * penetration_depth
mass = rho * volume
delta_T_initial = E_res / (mass * c)
print(f"Température initiale par impulsion : {delta_T_initial:.2f} °C")

# Paramètres temporels
delta_t = 1 / f_rep
N_pulses = 30
t_max = N_pulses * delta_t

# Grille spatiale
x = np.linspace(0, t_max * v, 800)
z = np.linspace(-100e-6, 100e-6, 800)
X, Z = np.meshgrid(x, z)

# Animation
frames = 30
times = np.linspace(0, t_max, frames)
N_impulses = [int(t * f_rep) for t in times]

# Profondeur par impulsion
delta = 0.1e-6

def ablation_depth(x, t, w0, v, delta_t, max_depth=1e-6):
    depth = np.zeros_like(x)
    n_impulses = int(t * f_rep) + 1
    for j in range(n_impulses):
        x0_j = j * v * delta_t
        contribution = delta * np.exp(-2 * ((x - x0_j)**2) / w0**2)
        depth = np.maximum(depth, contribution)
    depth = np.minimum(depth * (t / t_max), max_depth)
    return depth

def T_single_pulse(X_m, Z_m, x0_m, z0_m, t, E_res, w0_m, rho, c, kappa, z=0):
    if t <= 1e-12:
        return delta_T_initial * np.exp(-2 * ((X_m - x0_m)**2 + (Z_m - z0_m)**2) / w0_m**2)
    denom = np.pi * rho * c * np.sqrt(np.pi * kappa * t) * (8 * kappa * t + w0_m**2)
    num = E_res
    arg_exp = ((X_m - x0_m)**2 + (Z_m - z0_m)**2) * (w0_m**2 / (8 * kappa * t + w0_m**2) - 1) / (4 * kappa * t) - z**2 / (4 * kappa * t)
    return num / denom * np.exp(arg_exp)

# Figure avec 3 sous-graphiques
fig = plt.figure(figsize=(14, 18), dpi=120, facecolor='white')

# Carte thermique (x,z)
ax1 = fig.add_subplot(3, 1, 1)
im = ax1.imshow(np.zeros_like(X), extent=[0, t_max * v * 1e6, -100, 100], cmap='jet', vmin=0, vmax=500)
ax1.set_title("Time = 0.00 µs, Temp max = 0.00 °C", fontsize=16, pad=20)
ax1.set_xlabel("Position x (µm)", fontsize=14)
ax1.set_ylabel("Position z (µm)", fontsize=14)
ax1.tick_params(axis='both', labelsize=12)
cbar = fig.colorbar(im, ax=ax1)
cbar.set_label("Température (°C)", fontsize=14)
cbar.ax.tick_params(labelsize=12)

# Profondeur d'ablation
ax2 = fig.add_subplot(3, 1, 2)
depth_line, = ax2.plot(x * 1e6, np.zeros_like(x), 'b-', linewidth=2.5)
impulse_positions = [j * v * delta_t * 1e6 for j in range(N_pulses)]
ax2.scatter(impulse_positions, np.zeros_like(impulse_positions), c='red', marker='v', s=50)
ax2.set_xlim(0, t_max * v * 1e6)
ax2.set_ylim(-1.2, 0)
ax2.set_xlabel("Position x (µm)", fontsize=14)
ax2.set_ylabel("Profondeur (µm)", fontsize=14)
ax2.tick_params(axis='both', labelsize=12)
ax2.grid(True, linestyle='--', alpha=0.7)

# Température max vs temps
ax3 = fig.add_subplot(3, 1, 3)
line_temp, = ax3.plot([], [], 'r-', linewidth=2.5)
ax3.set_xlim(0, t_max * 1e6)
ax3.set_ylim(0, 1000)
ax3.set_xlabel("Temps (µs)", fontsize=14)
ax3.set_ylabel("Température max (°C)", fontsize=14)
ax3.grid(True, linestyle='--', alpha=0.7)
ax3.set_title("Évolution de la température maximale", fontsize=16)

# Liste pour stocker température max
temp_max_list = []

def update(frame):
    for coll in ax1.collections[1:]:
        coll.remove()

    t = times[frame]
    N = N_impulses[frame]
    T_total = np.zeros_like(X)
    Z0 = ablation_depth(x, t, w0, v, delta_t)

    for j in range(N + 1):
        t_j = t - j * delta_t
        x0_j = j * v * delta_t
        if t_j <= 0:
            continue
        temp_contribution = T_single_pulse(X, Z, x0_j, 0, t_j, E_res, w0, rho, c, kappa)
        T_total += temp_contribution

    T_total += T_offset
    mask = Z < -1 - Z0[np.newaxis, :]
    T_total[mask] = 0

    im.set_array(T_total)
    ax1.set_title(f"Time = {t*1e6:.2f} µs, Temp max = {T_total.max():.2f} °C", fontsize=16)

    # Contours (non retournés)
    levels = np.linspace(0, 1000, 10)
    ax1.contour(X*1e6, Z*1e6, T_total, levels=levels, cmap='cool', linewidths=1.5, alpha=0.8)

    depth_line.set_ydata(-Z0 * 1e6)

    # Température max
    temp_max = T_total.max()
    temp_max_list.append(temp_max)
    line_temp.set_data(times[:frame+1] * 1e6, temp_max_list)
    ax3.set_ylim(0, max(1000, max(temp_max_list) * 1.1))

    
