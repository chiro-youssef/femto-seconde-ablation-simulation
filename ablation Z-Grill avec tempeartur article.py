import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from math import ceil, log

# Paramètres
d = 28  # Diamètre du spot laser (µm)
w0 = d / 2
phi_LO = 0.82
F0 = 0.37
Fth_1 = 0.055
delta_1 = 18e-3
S = 0.8
v = 4e6
f = 800e3
eta_res = 0.4
rho = 7920
c = 500
kappa = 4e-6
T_offset = 23
penetration_depth = 18e-6
T_th = 607

# Conversion des unités
w0_m = w0 * 1e-6
v_m = v * 1e-6
spot_area = np.pi * w0_m**2
E_p = F0 * 1e4 * spot_area
E_res = eta_res * E_p
volume = spot_area * penetration_depth
mass = rho * volume
delta_T_initial = E_res / (mass * c)

# Taux de chevauchement et distances
phi_PO = 1 - v / (2 * w0 * f)
if phi_PO < 0 or phi_PO > 1:
    raise ValueError(f"phi_PO = {phi_PO} est hors de la plage [0, 1].")
Delta_x = (1 - phi_PO) * d
dH = (1 - phi_LO) * d
N_eff = 2 * w0 / Delta_x

def varying_params(N_eff):
    power = N_eff**(S - 1)
    return delta_1 * power, Fth_1 * power

# Grille spatiale
Lx = 160
Ly = 160
nx, ny = 100, 100
x = np.linspace(-100, 100, nx)
y = np.linspace(-100, 100, ny)
X, Y = np.meshgrid(x, y)
X_m = X * 1e-6
Y_m = Y * 1e-6

# Paramètres temporels
delta_t = 1 / f
N_pulses = 100
t_max = N_pulses * delta_t
t_after_7_impulses = 7 * delta_t
n_impulses_per_line = ceil(Lx / Delta_x)
n_lines = ceil(Ly / dH)

# Instants pour l'animation
frames = 100
times = np.linspace(0, t_max, frames)
N_impulses_per_frame = [min(int(t * f) + 1, N_pulses) for t in times]

# Points temporels pour la courbe
t_eval_per_pulse = np.linspace(1e-9, delta_t, 50)
t_eval = []
T_center_values = []

# Fonction de température
def T_single_pulse(X_m, Y_m, x0_m, y0_m, t, E_res, w0_m, rho, c, kappa):
    if t <= 1e-12:
        return delta_T_initial
    denom = np.pi * rho * c * np.sqrt(np.pi * kappa * t) * (8 * kappa * t + w0_m**2)
    num = 2 * E_res
    arg_exp = ((X_m - x0_m)**2 + (Y_m - y0_m)**2)*(w0_m**2/(8 * kappa * t + w0_m**2)-1) / (4 * kappa * t)
    return num / denom * np.exp(arg_exp)

# Calcul de la température au centre
for i in range(N_pulses):
    t_pulse = i * delta_t
    y0_i = -80 + (i // n_impulses_per_line) * dH
    x0_i = -80 + (i % n_impulses_per_line) * Delta_x
    t_current = t_pulse + t_eval_per_pulse
    t_eval.extend(t_current * 1e6)
    for t in t_eval_per_pulse:
        T = T_offset
        x_eval = x0_i * 1e-6
        y_eval = y0_i * 1e-6
        for j in range(i + 1):
            t_j = t + (i - j) * delta_t
            x0_j = -80 + (j % n_impulses_per_line) * Delta_x
            y0_j = -80 + (j // n_impulses_per_line) * dH
            if t_j <= 0:
                continue
            T += T_single_pulse(x_eval, y_eval, x0_j * 1e-6, y0_j * 1e-6, t_j, E_res, w0_m, rho, c, kappa)
        T_center_values.append(T)

# Animation de la courbe de température
fig_temp_anim = plt.figure(figsize=(8, 6))
ax_temp_anim = fig_temp_anim.add_subplot(111)
ax_temp_anim.set_xlabel('Temps (µs)')
ax_temp_anim.set_ylabel('Température (°C)')
ax_temp_anim.set_title('Variation de la température au centre')
ax_temp_anim.grid(True)
ax_temp_anim.set_yticks(np.arange(0, 1000, 200))
ax_temp_anim.set_ylim(0, 1000)
ax_temp_anim.set_xlim(0, max(t_eval))
line, = ax_temp_anim.plot([], [], label='Température au centre', color='black')
ax_temp_anim.axhline(y=607, color='red', linestyle='--', label='Seuil de saturation (607 °C)')
ax_temp_anim.axhline(y=T_th, color='orange', linestyle='--', label=f'Température critique ({T_th} °C)')
ax_temp_anim.legend()

def update_temp_curve(frame):
    idx = min(int(frame * len(t_eval) / frames), len(t_eval) - 1)
    line.set_data(t_eval[:idx], T_center_values[:idx])
    return [line]

ani_temp = FuncAnimation(fig_temp_anim, update_temp_curve, frames=frames, interval=50, blit=True)
ani_temp.save('temperature_curve_animation.gif', writer='pillow', fps=10)
try:
    ani_temp.save('temperature_curve_animation.mp4', writer='ffmpeg', fps=10)
except Exception as e:
    print(f"Erreur lors de la sauvegarde MP4 (courbe) : {e}. Installez ffmpeg.")
plt.close(fig_temp_anim)

# Courbe statique
fig_temp = plt.figure(figsize=(8, 6))
ax_temp = fig_temp.add_subplot(111)
ax_temp.plot(t_eval, T_center_values, label='Température au centre', color='black')
ax_temp.axhline(y=607, color='red', linestyle='--', label='Seuil de saturation (607 °C)')
ax_temp.axhline(y=T_th, color='orange', linestyle='--', label=f'Température critique ({T_th} °C)')
ax_temp.set_xlabel('Temps (µs)')
ax_temp.set_ylabel('Température (°C)')
ax_temp.set_title('Variation de la température au centre')
ax_temp.grid(True)
ax_temp.legend()
ax_temp.set_yticks(np.arange(0, 1000, 200))
ax_temp.set_ylim(0, 1000)
plt.tight_layout()
fig_temp.savefig('temperature_curve_static.png', dpi=300, bbox_inches='tight')
plt.close(fig_temp)

# Animation de la carte de chaleur
Z = np.zeros((ny, nx))
N = np.zeros((ny, nx))
impulse_positions = []
hot_spots = np.zeros((ny, nx), dtype=bool)
T_sat_values = np.zeros((ny, nx))
sat_hot_spots = np.zeros((ny, nx), dtype=bool)
T_sat_calculated = False

impulse_idx = 0
for j in range(n_lines):
    yj = -80 + j * dH
    yj_m = yj * 1e-6
    for i in range(n_impulses_per_line):
        xi = -80 + i * Delta_x
        xi_m = xi * 1e-6
        t_i = impulse_idx * delta_t
        if impulse_idx >= N_pulses:
            break
        impulse_positions.append((xi, yj, t_i))
        r2 = (X - xi)**2 + (Y - yj)**2
        N += (r2 <= (w0**2)).astype(int)
        delta_N, Fth_N = varying_params(N_eff)
        r_th_squared = (w0**2 / 2) * np.log(F0 / Fth_N)
        mask = (r2 <= r_th_squared) & (r_th_squared > 0)
        ln_F0_Fth_N = np.log(F0 / Fth_N)
        contribution = delta_N * (ln_F0_Fth_N - 2 * r2 / (w0**2))
        Z += np.where(mask, contribution, 0)
        impulse_idx += 1
    if impulse_idx >= N_pulses:
        break

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
im = ax.imshow(np.zeros_like(X), extent=[-100, 100, -100, 100], cmap='jet', vmin=0, vmax=1000)
ax.set_title("Time = 0.00 µs")
ax.set_xlabel("Position x (µm)")
ax.set_ylabel("Position y (µm)")
fig.colorbar(im, ax=ax, label="Température (°C)")

def update(frame):
    global hot_spots, T_sat_values, sat_hot_spots, T_sat_calculated
    t = times[frame]
    N_imp = N_impulses_per_frame[frame]
    for coll in ax.collections:
        coll.remove()
    T_total = np.zeros_like(X)
    for idx, (xi, yj, t_i) in enumerate(impulse_positions):
        if idx >= N_imp:
            break
        t_diff = t - t_i
        if t_diff <= 0:
            continue
        xi_m = xi * 1e-6
        yj_m = yj * 1e-6
        temp_contribution = T_single_pulse(X_m, Y_m, xi_m, yj_m, t_diff, E_res, w0_m, rho, c, kappa)
        T_total += temp_contribution
    T_total += T_offset
    hot_spots |= (T_total > T_th)
    if not T_sat_calculated and t >= t_after_7_impulses:
        T_sat_values[...] = T_total
        sat_hot_spots[...] = (T_sat_values > T_th)
        T_sat_calculated = True
    im.set_array(T_total)
    ax.set_title(f"Time = {t*1e6:.2f} µs")
    if N_imp > 0 and N_imp <= len(impulse_positions):
        xi, yj, t_i = impulse_positions[N_imp - 1]
        t_diff = t - t_i
        if t_diff >= 0:
            circle = plt.Circle((xi, yj), w0, color='white', fill=False, linestyle='--', alpha=0.007)
            ax.add_patch(circle)
    Z_levels = np.linspace(Z.min(), Z.max(), 5)[1:-1]
    ax.contour(X, Y, -Z, levels=Z_levels, colors='black', linestyles='-', alpha=0.5)
    return [im]

ani = FuncAnimation(fig, update, frames=frames, interval=50, blit=True)
ani.save('heatmap_animation.gif', writer='pillow', fps=10)
try:
    ani.save('heatmap_animation.mp4', writer='ffmpeg', fps=10)
except Exception as e:
    print(f"Erreur lors de la sauvegarde MP4 (carte) : {e}. Installez ffmpeg.")
plt.close(fig)

# Carte statique de la température (T_sat après 7 impulsions)
fig_heatmap = plt.figure(figsize=(8, 6))
ax_heatmap = fig_heatmap.add_subplot(111)
im_heatmap = ax_heatmap.imshow(T_sat_values, extent=[-100, 100, -100, 100], cmap='jet', vmin=0, vmax=1000)
ax_heatmap.set_xlabel('X (µm)')
ax_heatmap.set_ylabel('Y (µm)')
ax_heatmap.set_title('Carte de chaleur : Température après 7 impulsions')
fig_heatmap.colorbar(im_heatmap, ax=ax_heatmap, label='Température (°C)')
plt.tight_layout()
fig_heatmap.savefig('heatmap_static.png', dpi=300, bbox_inches='tight')
plt.close(fig_heatmap)

# Carte statique de l'ablation
fig_static = plt.figure(figsize=(8, 6))
ax_static = fig_static.add_subplot(111)
im_static = ax_static.imshow(-Z, extent=[-80, 80, -80, 80], cmap='viridis')
ax_static.set_xlabel('X (µm)')
ax_static.set_ylabel('Y (µm)')
ax_static.set_title('Profondeur d\'ablation finale')
fig_static.colorbar(im_static, ax=ax_static, label='Profondeur (µm)')
plt.tight_layout()
fig_static.savefig('ablation_final.png', dpi=300, bbox_inches='tight')
plt.close(fig_static)

# Carte des effets thermiques
fig_heat_effect = plt.figure(figsize=(8, 6))
ax_heat_effect = fig_heat_effect.add_subplot(111)
if np.any(sat_hot_spots):
    display_array = np.ones_like(Z)
    im_heat_effect = ax_heat_effect.imshow(display_array, extent=[-80, 80, -80, 80], cmap='Reds', vmin=0, vmax=1)
    ax_heat_effect.set_title('Effets de chaleur : T_sat > 607°C')
else:
    im_heat_effect = ax_heat_effect.imshow(-Z, extent=[-80, 80, -80, 80], cmap='viridis')
    ax_heat_effect.set_title('Effets de chaleur : T_sat <= 607°C')
ax_heat_effect.set_xlabel('X (µm)')
ax_heat_effect.set_ylabel('Y (µm)')
fig_heat_effect.colorbar(im_heat_effect, ax=ax_heat_effect, label='Indicateur' if np.any(sat_hot_spots) else 'Profondeur (µm)')
plt.tight_layout()
fig_heat_effect.savefig('heat_effect.png', dpi=300, bbox_inches='tight')
plt.close(fig_heat_effect)