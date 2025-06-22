import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from math import ceil, log

# Paramètres du premier code (ablation, ajustés partiellement)
d = 50  # Diamètre du spot laser (µm)
w0 = d / 2  # Rayon du spot (µm)
phi_LO = 0.82  # Taux de chevauchement des lignes
F0 = 0.37  # Fluence de pic (J/cm^2)
Fth_1 = 0.055  # Fluence seuil initiale pour N=1 (J/cm^2)
delta_1 = 18e-3  # Profondeur de pénétration initiale pour N=1 (µm)
S = 0.8  # Paramètre d'incubation pour l'acier inoxydable
v = 6e6  # Vitesse de balayage (µm/s) - Réduite à 3 m/s pour dépasser T_th
f = 800e3  # Fréquence de répétition (Hz)

# Paramètres thermiques
eta_res = 0.4  # Fraction de chaleur résiduelle
rho = 7920  # Densité de l'acier (kg/m³)
c = 500  # Capacité calorifique (J/kg·K)
kappa = 4e-6  # Diffusivité thermique (m²/s)
T_offset = 23  # Température initiale (°C)
penetration_depth = 18e-6  # Profondeur de pénétration thermique (m)

# Température critique (Tth)
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

# Calcul du taux de chevauchement des impulsions
phi_PO = 1 - v / (2 * w0 * f)
if phi_PO < 0 or phi_PO > 1:
    raise ValueError(f"phi_PO = {phi_PO} est hors de la plage [0, 1]. Ajustez v ou f.")

# Calcul des distances
Delta_x = (1 - phi_PO) * d  # Distance entre impulsions consécutives (µm)
dH = (1 - phi_LO) * d  # Espacement entre les lignes (µm)

# Calcul de N_eff
N_eff = 2 * w0 / Delta_x  # Nombre effectif d'impulsions

# Fonction pour les paramètres variables avec effet d'incubation
def varying_params(N_eff):
    power = N_eff**(S - 1)
    return delta_1 * power, Fth_1 * power

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
N_pulses = 50  # Nombre total d'impulsions
t_max = N_pulses * delta_t  # Temps total simulé (s)

# Temps après 7 impulsions pour calculer T_sat
t_after_7_impulses = 7 * delta_t  # 7 * (1/800e3) = 8.75e-6 s = 8.75 µs

# Nombre d'impulsions et de lignes
n_impulses_per_line = ceil(Lx / Delta_x)  # Impulsions par ligne
n_lines = ceil(Ly / dH)

# Instants pour l'animation
frames = 100  # Nombre de frames
times = np.linspace(0, t_max, frames)  # Temps uniformément espacés
N_impulses_per_frame = [min(int(t * f) + 1, N_pulses) for t in times]

# Points temporels pour la courbe de température
t_eval_per_pulse = np.linspace(1e-9, delta_t, 50)
t_eval = []
T_center_values = []

# Fonction pour la température d'une impulsion unique
def T_single_pulse(X_m, Y_m, x0_m, y0_m, t, E_res, w0_m, rho, c, kappa, z=0):
    if t <= 1e-12:  # Juste après l'impulsion
        return delta_T_initial  # Condition initiale constante
    denom = np.pi * rho * c * np.sqrt(np.pi * kappa * t) * (8 * kappa * t + w0_m**2)
    num = 2 * E_res
    arg_exp = ((X_m - x0_m)**2 + (Y_m - y0_m)**2)*(w0_m**2/(8 * kappa * t + w0_m**2)-1) / (4 * kappa * t) - z**2 / (4 * kappa * t)
    return num / denom * np.exp(arg_exp)

# Calcul de la température au centre de l'impulsion courante
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
        # Ablation
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

# Initialisation de la figure
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
    
    # Supprimer les anciens contours
    for coll in ax.collections:
        coll.remove()
    
    # Calculer la température
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
    
    # Mettre à jour les hot spots de manière cumulative (T > T_th)
    hot_spots |= (T_total > T_th)
    
    # Mettre à jour T_sat_values après 7 impulsions (t = 8.75 µs)
    if not T_sat_calculated and t >= t_after_7_impulses:
        T_sat_values[...] = T_total  # On prend la température après 7 impulsions comme T_sat
        sat_hot_spots[...] = (T_sat_values > T_th)
        T_sat_calculated = True
    
    # Mettre à jour la carte de chaleur
    im.set_array(T_total)
    ax.set_title(f"Time = {t*1e6:.2f} µs")
    
    # Ajouter un contour pour l'impulsion courante
    if N_imp > 0 and N_imp <= len(impulse_positions):
        xi, yj, t_i = impulse_positions[N_imp - 1]
        t_diff = t - t_i
        if t_diff >= 0:
            circle = plt.Circle((xi, yj), w0, color='white', fill=False, linestyle='--', alpha=0.007)
            ax.add_patch(circle)
    
    # Ajouter des contours pour la profondeur
    Z_levels = np.linspace(Z.min(), Z.max(), 5)[1:-1]
    ax.contour(X, Y, -Z, levels=Z_levels, colors='black', linestyles='-', alpha=0.5)
    
    return [im]

# Créer l'animation
ani = FuncAnimation(fig, update, frames=frames, interval=50, blit=True)

# Sauvegarder en GIF
ani.save('ablation_thermal_impulse_aligned_v5.gif', writer='pillow', fps=10)

# Afficher l'animation
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

# Afficher une carte 2D pour les effets de chaleur
fig_heat_effect = plt.figure(figsize=(8, 6))
ax_heat_effect = fig_heat_effect.add_subplot(111)

# Vérifier si T_sat dépasse T_th quelque part
if np.any(sat_hot_spots):  # Si T_sat > T_th en au moins un point
    # Afficher toute la surface en rouge
    display_array = np.ones_like(Z)  # Matrice de 1 pour une couleur uniforme
    im_heat_effect = ax_heat_effect.imshow(display_array, extent=[-80, 80, -80, 80], cmap='Reds', vmin=0, vmax=1)
    ax_heat_effect.set_title('Effets de chaleur : T_sat > 607°C (toute la surface en rouge)')
else:
    # Sinon, afficher la profondeur d'ablation
    im_heat_effect = ax_heat_effect.imshow(-Z, extent=[-80, 80, -80, 80], cmap='viridis')
    ax_heat_effect.set_title('Effets de chaleur : T_sat ≤ 607°C (profondeur d\'ablation)')

ax_heat_effect.set_xlabel('X (µm)')
ax_heat_effect.set_ylabel('Y (µm)')
fig_heat_effect.colorbar(im_heat_effect, ax=ax_heat_effect, label='Indicateur' if np.any(sat_hot_spots) else 'Profondeur (µm)')
plt.tight_layout()
plt.show()

# Afficher la courbe de température au centre de l'impulsion
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