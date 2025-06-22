import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from math import ceil, log

# Paramètres du modèle
d = 28  # Diamètre du spot laser (µm)
w0 = d / 2  # Rayon du spot (µm)
F0 = 10  # Fluence de pic (J/cm^2)
Fth_1 = 0.055  # Fluence seuil initiale pour N=1 (J/cm^2)
delta_1 = 18e-3  # Profondeur de pénétration initiale pour N=1 (µm)
S = 0.8  # Paramètre d'incubation pour l'acier inoxydable
v = 1e6  # Vitesse de balayage (µm/s)
f = 200e3  # Fréquence de répétition (Hz)
phi_PO = 1 - v / (2 * w0 * f)  # Taux de chevauchement

# Paramètres thermiques
eta_res = 0.4  # Fraction de chaleur résiduelle
rho = 7920  # Densité de l'acier (kg/m³)
c = 500  # Capacité calorifique (J/kg·K)
kappa = 4e-6  # Diffusivité thermique (m²/s)
T_offset = 23  # Température initiale (°C)
penetration_depth = 18e-6  # Profondeur de pénétration thermique (m)
T_th = 607  # Température critique (°C)

# Conversion des unités
w0_m = w0 * 1e-6  # Rayon du spot en mètres
v_m = v * 1e-6  # Vitesse en m/s
spot_area = np.pi * w0_m**2  # Surface du spot (m²)
E_p = F0 * 1e4 * spot_area  # Énergie par impulsion (J)
E_res = eta_res * E_p  # Chaleur résiduelle par impulsion (J)
volume = spot_area * penetration_depth  # Volume chauffé (m³)
mass = rho * volume  # Masse du volume chauffé (kg)
delta_T_initial = E_res / (mass * c)  # Hausse de température initiale (°C)
print(f"Température initiale par impulsion : {delta_T_initial:.2f} °C")

# Vérification du taux de chevauchement
if phi_PO < 0 or phi_PO > 1:
    raise ValueError(f"phi_PO = {phi_PO} est hors de la plage [0, 1].")
Delta = (1 - phi_PO) * d * 1e-6  # Espacement en mètres
N_eff = 2 * w0 / (Delta * 1e6)  # Nombre effectif d'impulsions

# Dimensions de la zone simulée (µm)
Lx = 160  # Longueur en x (de -80 à 80 µm)
Ly = 160  # Longueur en y (de -80 à 80 µm)

# Grille de points
nx, ny = 100, 100  # Résolution
x = np.linspace(-100, 100, nx)
y = np.linspace(-100, 100, ny)
X, Y = np.meshgrid(x, y)
X_m = X * 1e-6  # Grille x en mètres
Y_m = Y * 1e-6  # Grille y en mètres

# Paramètres de la spirale d'Archimède
a = (50e-6) / (20 * np.pi)  # Espacement pour r_max ≈ 40 µm après 5 tours
theta_max = 25 * np.pi  # 5 tours
theta_dense = np.linspace(0, theta_max, 5000)  # Réduit pour moins d'impulsions
r_dense = a * theta_dense
x_dense = r_dense * np.cos(theta_dense)  # Centré à (0, 0)
y_dense = r_dense * np.sin(theta_dense)

# Calcul des paramètres temporels
delta_t = 1 / f  # Intervalle entre impulsions (s)

# Générer les positions des impulsions
spiral_x, spiral_y, thetas, times = [x_dense[0]], [y_dense[0]], [0.0], [0.0]
last_x, last_y = x_dense[0], y_dense[0]
impulse_idx = 0

for i in range(1, len(theta_dense)):
    dist = np.sqrt((x_dense[i] - last_x)**2 + (y_dense[i] - last_y)**2)
    if dist >= Delta:
        spiral_x.append(x_dense[i])
        spiral_y.append(y_dense[i])  # Corrigé
        thetas.append(theta_dense[i])
        times.append(impulse_idx * delta_t)
        last_x, last_y = x_dense[i], y_dense[i]
        impulse_idx += 1
        if impulse_idx >= 1000:  # Limiter à 1000 impulsions comme dans le second code
            break

spiral_x, spiral_y, times = np.array(spiral_x), np.array(spiral_y), np.array(times)
N_pulses = len(spiral_x)  # Nombre total d'impulsions
t_max = N_pulses * delta_t  # Temps total

# Visualisation de la trajectoire de la spirale
fig_spiral = plt.figure(figsize=(6, 6))
ax_spiral = fig_spiral.add_subplot(111)
ax_spiral.plot(spiral_x * 1e6, spiral_y * 1e6, 'b-', label='Trajectoire spirale')
ax_spiral.scatter(spiral_x * 1e6, spiral_y * 1e6, c='red', s=10, label='Impulsions')
ax_spiral.set_xlabel('X (µm)')
ax_spiral.set_ylabel('Y (µm)')
ax_spiral.set_title('Trajectoire de la spirale d\'Archimède')
ax_spiral.legend()
ax_spiral.grid(True)
plt.tight_layout()
fig_spiral.savefig('spiral_trajectory.png', dpi=300, bbox_inches='tight')
plt.show()

# Fonction pour les paramètres variables avec effet d'incubation
def varying_params(N_eff):
    power = N_eff**(S - 1)
    return delta_1 * power, Fth_1 * power

# Fonction pour la température d'une impulsion unique
def T_single_pulse(X_m, Y_m, x0_m, y0_m, t, E_res, w0_m, rho, c, kappa):
    if t <= 1e-12:  # Juste après l'impulsion
        return delta_T_initial
    denom = np.pi * rho * c * np.sqrt(np.pi * kappa * t) * (8 * kappa * t + w0_m**2)
    num = 2 * E_res
    arg_exp = ((X_m - x0_m)**2 + (Y_m - y0_m)**2)*(w0_m**2/(8 * kappa * t + w0_m**2)-1) / (4 * kappa * t)
    return num / denom * np.exp(arg_exp)

# Initialisation des matrices
Z = np.zeros((ny, nx))  # Profondeur d'ablation (µm)
N = np.zeros((ny, nx))  # Compteur d'impulsions
impulse_positions = []  # Stocker (xi, yj, t_i)
hot_spots = np.zeros((ny, nx), dtype=bool)  # Suivi des zones où T > T_th

# Calcul de l'ablation en spirale
for idx, (xi, yj, t_i) in enumerate(zip(spiral_x * 1e6, spiral_y * 1e6, times)):  # Convertir en µm
    if -80 <= xi <= 80 and -80 <= yj <= 80:  # Vérifier les limites
        impulse_positions.append((xi, yj, t_i))
        r2 = (X - xi)**2 + (Y - yj)**2
        N += (r2 <= (w0**2)).astype(int)
        delta_N, Fth_N = varying_params(N_eff)
        r_th_squared = (w0**2 / 2) * np.log(F0 / Fth_N)
        mask = (r2 <= r_th_squared) & (r_th_squared > 0)
        ln_F0_Fth_N = np.log(F0 / Fth_N)
        contribution = delta_N * (ln_F0_Fth_N - 2 * r2 / (w0**2))
        Z += np.where(mask, contribution, 0)

# Paramètres pour la courbe de température au centre
t_eval_per_pulse = np.linspace(1e-9, delta_t, 50)
t_eval = []
T_center_values = []

# Calcul de la température au centre (0, 0)
for i in range(N_pulses):
    t_pulse = i * delta_t
    xi = spiral_x[i] * 1e6  # µm
    yj = spiral_y[i] * 1e6  # µm
    t_current = t_pulse + t_eval_per_pulse
    t_eval.extend(t_current * 1e6)  # Convertir en µs
    for t in t_eval_per_pulse:
        T = T_offset
        x_eval = 0  # Centre à (0, 0)
        y_eval = 0
        for j in range(i + 1):
            t_j = t + (i - j) * delta_t
            if t_j <= 0 or t_j > 10 * delta_t:  # Limiter les contributions anciennes
                continue
            x0_j = spiral_x[j] * 1e6  # µm
            y0_j = spiral_y[j] * 1e6  # µm
            T += T_single_pulse(x_eval * 1e-6, y_eval * 1e-6, x0_j * 1e-6, y0_j * 1e-6, t_j, E_res, w0_m, rho, c, kappa)
        T_center_values.append(T)

# Paramètres de l'animation
frames = 100
times_anim = np.linspace(0, t_max, frames)
N_impulses_per_frame = [min(int(t * f) + 1, N_pulses) for t in times_anim]

# Initialisation de la figure pour l'animation
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
im = ax.imshow(np.zeros_like(X), extent=[-100, 100, -100, 100], cmap='jet', vmin=0, vmax=1000)
ax.set_title("Time = 0.00 µs")
ax.set_xlabel("Position x (µm)")
ax.set_ylabel("Position y (µm)")
fig.colorbar(im, ax=ax, label="Température (°C)")

# Fonction de mise à jour pour l'animation
def update(frame):
    global hot_spots
    t = times_anim[frame]
    N_imp = N_impulses_per_frame[frame]
    
    for coll in ax.collections:
        coll.remove()
    
    T_total = np.zeros_like(X)
    for idx, (xi, yj, t_i) in enumerate(impulse_positions):
        if idx >= N_imp:
            break
        t_diff = t - t_i
        if t_diff <= 0 or t_diff > 10 * delta_t:  # Limiter les contributions
            continue
        xi_m = xi * 1e-6
        yj_m = yj * 1e-6
        temp_contribution = T_single_pulse(X_m, Y_m, xi_m, yj_m, t_diff, E_res, w0_m, rho, c, kappa)
        T_total += temp_contribution
    
    T_total += T_offset
    
    if T_total.shape == hot_spots.shape:
        hot_spots |= (T_total > T_th)
    
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
    
    print(f"Frame {frame + 1}/{frames}, t = {t*1e6:.2f} µs, Temp max = {T_total.max():.2f} °C, Profondeur max = {-Z.min():.2f} µm")
    return [im]

# Créer et sauvegarder l'animation
ani = FuncAnimation(fig, update, frames=frames, interval=100, blit=False)
ani.save('ablation_spiral_thermal_fixed.gif', writer='pillow', fps=10)
plt.show()

# Courbe statique de la température au centre
fig_temp = plt.figure(figsize=(8, 6))
ax_temp = fig_temp.add_subplot(111)
ax_temp.plot(t_eval, T_center_values, label='Température au centre', color='black')
ax_temp.axhline(y=T_th, color='orange', linestyle='--', label=f'Température critique ({T_th} °C)')
ax_temp.set_xlabel('Temps (µs)')
ax_temp.set_ylabel('Température (°C)')
ax_temp.set_title('Variation de la température au centre')
ax_temp.legend()
ax_temp.grid(True)
ax_temp.set_ylim(0, 1000)
plt.tight_layout()
fig_temp.savefig('temperature_center_static.png', dpi=300, bbox_inches='tight')
plt.show()

# Carte 2D statique de l'ablation
fig_static = plt.figure(figsize=(8, 6))
ax_static = fig_static.add_subplot(111)
im_static = ax_static.imshow(-Z, extent=[-80, 80, -80, 80], cmap='viridis')
ax_static.set_xlabel('X (µm)')
ax_static.set_ylabel('Y (µm)')
ax_static.set_title('Profondeur d\'ablation finale (Spirale)')
fig_static.colorbar(im_static, ax=ax_static, label='Profondeur (µm)')
plt.tight_layout()
fig_static.savefig('ablation_final.png', dpi=300, bbox_inches='tight')
plt.show()

# Carte 3D statique de l'ablation
fig_3d = plt.figure(figsize=(10, 8))
ax_3d = fig_3d.add_subplot(111, projection='3d')
surf = ax_3d.plot_surface(X, Y, -Z, cmap='viridis', edgecolor='none')
ax_3d.set_xlabel('X (µm)')
ax_3d.set_ylabel('Y (µm)')
ax_3d.set_zlabel('Profondeur (µm)')
ax_3d.set_title('Profondeur d\'ablation finale (3D, Spirale)')
fig_3d.colorbar(surf, ax=ax_3d, label='Profondeur (µm)', shrink=0.5, aspect=10)
plt.tight_layout()
fig_3d.savefig('ablation_final_3d.png', dpi=300, bbox_inches='tight')
plt.show()

# Carte des effets thermiques
fig_heat_effect = plt.figure(figsize=(8, 6))
ax_heat_effect = fig_heat_effect.add_subplot(111)
if np.any(hot_spots):
    display_array = np.ones_like(Z)
    im_heat_effect = ax_heat_effect.imshow(display_array, extent=[-80, 80, -80, 80], cmap='Reds', vmin=0, vmax=1)
    ax_heat_effect.set_title('Effets de chaleur : T > 607°C')
else:
    im_heat_effect = ax_heat_effect.imshow(-Z, extent=[-80, 80, -80, 80], cmap='viridis')
    ax_heat_effect.set_title('Effets de chaleur : T <= 607°C')
ax_heat_effect.set_xlabel('X (µm)')
ax_heat_effect.set_ylabel('Y (µm)')
fig_heat_effect.colorbar(im_heat_effect, ax=ax_heat_effect, label='Indicateur' if np.any(hot_spots) else 'Profondeur (µm)')
plt.tight_layout()
fig_heat_effect.savefig('heat_effect.png', dpi=300, bbox_inches='tight')
plt.show()