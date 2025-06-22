import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from math import ceil, log

# Paramètres du laser et de l'ablation
d = 28  # Diamètre du spot laser (µm)
w0 = d / 2  # Rayon du spot (µm)
phi_LO = 0.9  # Taux de chevauchement des lignes
F0 = 10  # Fluence de pic (J/cm^2)
Fth_1 = 0.055  # Fluence seuil initiale pour N=1 (J/cm^2)
delta_1 = 18e-3  # Profondeur de pénétration initiale pour N=1 (µm)
S = 0.8  # Paramètre d'incubation pour l'acier inoxydable
v = 1e6  # Vitesse de balayage (µm/s)
f = 200e3  # Fréquence de répétition (Hz)

# Paramètres thermiques
eta_res = 0.4  # Fraction de chaleur résiduelle
rho = 7920  # Densité de l'acier (kg/m³)
c = 500  # Capacité calorifique (J/kg·K)
kappa = 4e-6  # Diffusivité thermique (m²/s)
T_offset = 23  # Température initiale (°C)
penetration_depth = 18e-6  # Profondeur de pénétration thermique (m)
T_th = 607  # Température critique en °C

# Conversion des unités pour calculs thermiques
w0_m = w0 * 1e-6  # Rayon du spot en mètres
v_m = v * 1e-6  # Vitesse en m/s
spot_area = np.pi * w0_m**2  # Surface du spot (m²)
E_p = F0 * 1e4 * spot_area  # Énergie par impulsion (J), conversion J/cm² -> J/m²
E_res = eta_res * E_p  # Chaleur résiduelle par impulsion (J)
volume = spot_area * penetration_depth  # Volume chauffé (m³)
mass = rho * volume  # Masse du volume chauffé (kg)
delta_T_initial = E_res / (mass * c)  # Hausse de température initiale (°C)
# Ajustement pour réduire la température en dessous du seuil de saturation
temp_scale_factor = 0.5  # Facteur d'échelle pour réduire fortement l'amplitude de la température
delta_T_initial = delta_T_initial * temp_scale_factor
print(f"Température initiale par impulsion (ajustée) : {delta_T_initial:.2f} °C")

# Calcul du taux de chevauchement des impulsions
phi_PO = 1 - v / (2 * w0 * f)
if phi_PO < 0 or phi_PO > 1:
    raise ValueError(f"phi_PO = {phi_PO} est hors de la plage [0, 1]. Ajustez v ou f.")

# Calcul des distances
Delta_x = (1 - phi_PO) * d  # Distance entre impulsions consécutives (µm)
dH = (1 - phi_LO) * d  # Espacement entre les lignes (µm)

# Dimensions de la zone simulée (µm)
Lx = 160  # Longueur en x (de -80 à 80 µm)
Ly = 160  # Longueur en y (de -80 à 80 µm)

# Grille de points (µm pour ablation, mètres pour thermique)
nx, ny = 100, 100  # Résolution de la grille
x = np.linspace(-100, 100, nx)
y = np.linspace(-100, 100, ny)
X, Y = np.meshgrid(x, y)
X_m = X * 1e-6  # Grille x en mètres
Y_m = Y * 1e-6  # Grille y en mètres

# Paramètres temporels
delta_t = 1 / f  # Intervalle entre impulsions (s)
N_pulses = 100  # Nombre total d'impulsions
t_max = N_pulses * delta_t  # Temps total simulé (s)
t_after_7_impulses = 7 * delta_t  # Temps après 7 impulsions

# Nombre d'impulsions et de lignes
n_impulses_per_line = ceil(Lx / Delta_x)  # Impulsions par ligne
n_lines = ceil(Ly / dH)

# Instants pour l'animation
frames = 100  # Nombre de frames
times = np.linspace(10e-9, t_max, frames)  # Temps uniformément espacés
N_impulses_per_frame = [min(int(t * f) + 1, N_pulses) for t in times]

# Nouvelle fonction d'ablation
def ablation_depth(y, Fc_loc, delta_loc, Fth_loc, delta_x_loc):
    """
    Profondeur d’ablation par impulsion selon la formule :
      L = ln(Fc_loc / Fth_loc)
      z(y) = (Δx·δ·√2)/(3·ω₀) · √([L − 2y²/ω₀²]) · [ (2ω₀²/Δx²)·(L − 2y²/ω₀²) − 1 ]
    """
    L = np.log(Fc_loc / Fth_loc)
    arg = L - 2 * (y**2) / (w0_m**2)
    positive = arg > 0
    fac = (delta_x_loc * delta_loc * np.sqrt(2)) / (3 * w0_m)
    sqrt_term = np.sqrt(np.where(positive, arg, 0.0))
    bracket = (2 * w0_m**2 / delta_x_loc**2) * arg - 1
    z = fac * sqrt_term * bracket
    return np.where(positive, z, 0.0)

# Fonction pour les paramètres variables
def varying_params(N_eff):
    power = N_eff**(S - 1)
    return delta_1 * 1e-6 * power, Fth_1 * 1e4 * power  # Convertir µm en m et J/cm² en J/m²

# Fonction pour la température d'une impulsion unique
def T_single_pulse(X_m, Y_m, x0_m, y0_m, t, E_res, w0_m, rho, c, kappa, z=0):
    if t <= 1e-12:  # Seuil comme dans le premier code
        return delta_T_initial  # Condition initiale constante, déjà ajustée
    denom = np.pi * rho * c * np.sqrt(np.pi * kappa * t) * (8 * kappa * t + w0_m**2) * 2  # Facteur 2 pour réduire l'amplitude
    num = 2 * E_res * temp_scale_factor  # Appliquer le facteur d'échelle pour réduire la température
    arg_exp = ((X_m - x0_m)**2 + (Y_m - y0_m)**2)*(w0_m**2/(8 * kappa * t + w0_m**2)-1) / (4 * kappa * t) - z**2 / (4 * kappa * t)
    return num / denom * np.exp(arg_exp)

# Initialisation des matrices
Z = np.zeros((ny, nx))  # Profondeur d'ablation (µm)
N = np.zeros((ny, nx))  # Compteur d'impulsions
impulse_positions = []  # Stocker (xi, yj, t_i)
hot_spots = np.zeros((ny, nx), dtype=bool)  # Matrice pour marquer les zones où T > T_th
T_sat_values = np.zeros((ny, nx))  # Matrice pour stocker T_sat pour chaque point
sat_hot_spots = np.zeros((ny, nx), dtype=bool)  # Matrice pour marquer où T_sat > T_th
T_sat_calculated = False  # Indicateur pour savoir si T_sat a été calculé

# Calcul de l'ablation et stockage des positions
impulse_idx = 0
Nmax = 25  # Limite sur N_eff
N_eff = 2 * w0 / Delta_x  # N_eff basé sur l'espacement des impulsions
if Nmax is not None:
    N_eff = min(N_eff, Nmax)
delta_N, Fth_N = varying_params(N_eff)
Fc = F0 * 1e4  # Conversion de F0 en J/m²
for j in range(n_lines):
    yj = -80 + j * dH  # Début à y = -80 µm
    yj_m = yj * 1e-6
    for i in range(n_impulses_per_line):
        xi = -80 + i * Delta_x  # Début à x = -80 µm
        xi_m = xi * 1e-6
        t_i = impulse_idx * delta_t  # Temps séquentiel
        if impulse_idx >= N_pulses:
            break
        impulse_positions.append((xi, yj, t_i))
        y_relative = Y_m - yj_m  # Distance transversale en mètres
        Z += ablation_depth(y_relative, Fc, delta_N, Fth_N, Delta_x * 1e-6) * 1e6  # Convertir m en µm
        impulse_idx += 1
    if impulse_idx >= N_pulses:
        break

# Calcul de la température au centre pour la courbe 1D en fonction du temps
t_eval_per_pulse = np.linspace(1e-9, delta_t, 50)
t_eval = []
T_center_values = []
for i in range(N_pulses):
    t_pulse = i * delta_t
    y0_i = -80 + (i // n_impulses_per_line) * dH  # Coordonnée y de la ligne
    x0_i = -80 + (i % n_impulses_per_line) * Delta_x  # Coordonnée x de l'impulsion
    t_current = t_pulse + t_eval_per_pulse
    t_eval.extend(t_current * 1e6)  # Convertir en µs
    for t in t_eval_per_pulse:
        T = T_offset
        x_eval = x0_i * 1e-6  # Suivre le centre de l'impulsion
        y_eval = y0_i * 1e-6
        for j in range(i + 1):
            t_j = t + (i - j) * delta_t
            x0_j = -80 + (j % n_impulses_per_line) * Delta_x
            y0_j = -80 + (j // n_impulses_per_line) * dH
            if t_j <= 0:
                continue
            T += T_single_pulse(x_eval, y_eval, x0_j * 1e-6, y0_j * 1e-6, t_j, E_res, w0_m, rho, c, kappa)
        T_center_values.append(T)

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
    
    print(f"Frame {frame + 1}/{frames}, t = {t*1e6:.2f} µs, Temp max = {T_total.max():.2f} °C, Profondeur max = {-Z.min():.2f} µm")
    return [im]

# Créer l'animation
ani = FuncAnimation(fig, update, frames=frames, interval=100, blit=False)
ani.save('ablation_thermal_impulse_corrected_v8.gif', writer='pillow', fps=10)
plt.show()

# Afficher la carte 2D statique de l'ablation
fig_static = plt.figure(figsize=(8, 6))
ax_static = fig_static.add_subplot(111)
im_static = ax_static.imshow(-Z, extent=[-80, 80, -80, 80], cmap='viridis')
ax_static.set_xlabel('X (µm)')
ax_static.set_ylabel('Y (µm)')
ax_static.set_title('Profondeur d\'ablation finale')
fig_static.colorbar(im_static, ax=ax_static, label='Profondeur (µm)')
plt.tight_layout()
plt.show()

# Afficher la carte 3D statique de l'ablation
fig_3d = plt.figure(figsize=(10, 8))
ax_3d = fig_3d.add_subplot(111, projection='3d')
surf = ax_3d.plot_surface(X, Y, -Z, cmap='viridis', edgecolor='none')
ax_3d.set_xlabel('X (µm)')
ax_3d.set_ylabel('Y (µm)')
ax_3d.set_zlabel('Profondeur (µm)')
ax_3d.set_title('Profondeur d\'ablation finale (3D)')
fig_3d.colorbar(surf, ax=ax_3d, label='Profondeur (µm)', shrink=0.5, aspect=10)
plt.tight_layout()
plt.show()

# Afficher la courbe 1D : z(x) à y = 0 µm
plt.figure(figsize=(12, 8))
x_slice = x  # x en µm
y_index = len(y) // 2  # Correspond à y ≈ 0 µm
z_slice = -Z[y_index, :]  # Profondeur le long de x à y fixe
plt.plot(x_slice, z_slice, color='navy', linewidth=2, marker='o', markersize=4, 
         markevery=10, label='Profil à y = 0 µm')
plt.xlabel('Position x (µm)', fontsize=12, fontweight='bold')
plt.ylabel('Profondeur Z (µm)', fontsize=12, fontweight='bold')
plt.title(f'Profondeur d\'ablation pour Δy = {dH:.1f} µm, R_y = {phi_LO*100:.1f}%', 
          fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Afficher la courbe 1D : z(y) à x = 50 µm
plt.figure(figsize=(12, 8))
y_slice = y  # y en µm
x_index = np.argmin(np.abs(x - 50))  # Correspond à x ≈ 50 µm
z_slice = -Z[:, x_index]  # Profondeur le long de y à x fixe
plt.plot(y_slice, z_slice, color='crimson', linewidth=2, marker='s', markersize=2, 
         markevery=10, label='Profil à x = 50 µm')
plt.xlabel('Position y (µm)', fontsize=12, fontweight='bold')
plt.ylabel('Profondeur Z (µm)', fontsize=12, fontweight='bold')
plt.title(f'Profondeur d\'ablation pour Δy = {dH:.1f} µm, R_y = {phi_LO*100:.1f}%', 
          fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Afficher la carte 2D des effets de chaleur
fig_heat_effect = plt.figure(figsize=(8, 6))
ax_heat_effect = fig_heat_effect.add_subplot(111)
if np.any(sat_hot_spots):
    display_array = np.ones_like(Z)  # Matrice de 1 pour une couleur uniforme
    im_heat_effect = ax_heat_effect.imshow(display_array, extent=[-80, 80, -80, 80], cmap='Reds', vmin=0, vmax=1)
    ax_heat_effect.set_title('Effets de chaleur : T_sat > 607°C (toute la surface en rouge)')
else:
    im_heat_effect = ax_heat_effect.imshow(-Z, extent=[-80, 80, -80, 80], cmap='viridis')
    ax_heat_effect.set_title('Effets de chaleur : T_sat ≤ 607°C (profondeur d\'ablation)')
ax_heat_effect.set_xlabel('X (µm)')
ax_heat_effect.set_ylabel('Y (µm)')
fig_heat_effect.colorbar(im_heat_effect, ax=ax_heat_effect, label='Indicateur' if np.any(sat_hot_spots) else 'Profondeur (µm)')
plt.tight_layout()
plt.show()

# Afficher la courbe 1D de température en fonction du temps
fig_temp = plt.figure(figsize=(8, 6))
ax_temp = fig_temp.add_subplot(111)
ax_temp.plot(t_eval, T_center_values, label='Température au centre de l\'impulsion', color='black')
ax_temp.axhline(y=607, color='red', linestyle='--', label='Seuil de saturation (607 °C)')
ax_temp.axhline(y=T_th, color='orange', linestyle='--', label=f'Température critique ({T_th} °C)')
ax_temp.set_xlabel('Temps (µs)')
ax_temp.set_ylabel('Température (°C)')
ax_temp.set_title('Variation de la température au centre de l\'impulsion')
ax_temp.grid(True)
ax_temp.legend()
ax_temp.set_yticks(np.arange(0, 1000, 200))
ax_temp.set_ylim(0, 1000)
plt.tight_layout()
plt.show()