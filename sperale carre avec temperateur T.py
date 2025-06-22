import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from math import ceil, log

# Paramètres du modèle (adaptés du premier code)
d = 28  # Diamètre du spot laser (µm)
w0 = d / 2  # Rayon du spot (µm)
F0 = 10  # Fluence de pic (J/cm^2)
Fth_1 = 0.055  # Fluence seuil initiale pour N=1 (J/cm^2)
delta_1 = 18e-3  # Profondeur de pénétration initiale pour N=1 (µm)
S = 0.8  # Paramètre d'incubation pour l'acier inoxydable
v = 1000000  # Vitesse de balayage (µm/s)
f = 200e3  # Fréquence de répétition (Hz)

# Paramètres thermiques
eta_res = 0.4  # Fraction de chaleur résiduelle
rho = 7920  # Densité de l'acier (kg/m³)
c = 500  # Capacité calorifique (J/kg·K)
kappa = 4e-6  # Diffusivité thermique (m²/s)
T_offset = 23  # Température initiale (°C)
penetration_depth = 18e-9  # Profondeur de pénétration thermique (m)

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

# Dimensions de la zone simulée (µm)
Lx = 160  # Longueur en x (de -80 à 80 µm)
Ly = 160  # Longueur en y (de -80 à 80 µm)

# Grille de points
nx, ny = 100, 100  # Résolution
x = np.linspace(-80, 80, nx)
y = np.linspace(-80, 80, ny)
X, Y = np.meshgrid(x, y)
X_m = X * 1e-6  # Grille x en mètres
Y_m = Y * 1e-6  # Grille y en mètres

# Génération des points de la spirale carrée
Delta = v_m / f  # Espacement entre impulsions (m)
spiral_x, spiral_y, times = [0.0], [0.0], [0.0]  # Commencer au centre
current_x, current_y = 0.0, 0.0
direction = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # Droite, Haut, Gauche, Bas
n = 10  # Nombre de tours
impulse_idx = 0
delta_t = 1 / f  # Intervalle entre impulsions (s)

for k in range(n):
    for i in range(4):  # 4 segments par tour
        dir_idx = i % 4
        dx, dy = direction[dir_idx]
        if i in [0, 1]:  # Droite ou Haut
            steps = 1 + 2 * k
        else:  # Gauche ou Bas
            steps = 2 + 2 * k
        length = steps * Delta  # Longueur du segment (m)
        num_points = max(2, int(length / Delta))  # Nombre de points
        for t in np.linspace(0, length, num_points, endpoint=False):
            current_x += dx * Delta * 1e6  # Convertir en µm
            current_y += dy * Delta * 1e6
            spiral_x.append(current_x)
            spiral_y.append(current_y)
            times.append(impulse_idx * delta_t)
            impulse_idx += 1

spiral_x, spiral_y, times = np.array(spiral_x), np.array(spiral_y), np.array(times)
N_pulses = len(spiral_x)  # Nombre total d'impulsions
t_max = N_pulses * delta_t  # Temps total

# Fonction pour la température d'une impulsion unique
def T_single_pulse(X_m, Y_m, x0_m, y0_m, t, E_res, w0_m, rho, c, kappa, z=0):
    if t <= 1e-12:  # Juste après l'impulsion
        return delta_T_initial * np.exp(-2 * ((X_m - x0_m)**2 + (Y_m - y0_m)**2) / w0_m**2)
    denom = np.pi * rho * c * np.sqrt(np.pi * kappa * t) * (8 * kappa * t + w0_m**2)
    num = E_res
    arg_exp = ((X_m - x0_m)**2 + (Y_m - y0_m)**2)*(w0_m**2/(8 * kappa * t + w0_m**2)-1) / (4 * kappa * t) - z**2 / (4 * kappa * t)
    return num / denom * np.exp(arg_exp)

# Initialisation des matrices
Z = np.zeros((ny, nx))  # Profondeur d'ablation (µm)
N = np.zeros((ny, nx))  # Compteur d'impulsions
impulse_positions = []  # Stocker (xi, yj, t_i)

# Calcul de l'ablation en spirale carrée
for idx, (xi, yj, t_i) in enumerate(zip(spiral_x, spiral_y, times)):
    if -80 <= xi <= 80 and -80 <= yj <= 80:  # Vérifier les limites
        impulse_positions.append((xi, yj, t_i))
        r2 = (X - xi)**2 + (Y - yj)**2
        N += (r2 <= (w0**2)).astype(int)
        Fth_N = Fth_1 * np.power(np.maximum(N, 1), S - 1)
        delta_N = delta_1 * np.power(np.maximum(N, 1), S - 1)
        r_th_squared = (w0**2 / 2) * np.log(F0 / Fth_N)
        mask = (r2 <= r_th_squared) & (r_th_squared > 0)
        ln_F0_Fth_N = np.log(F0 / Fth_N)
        contribution = delta_N * (ln_F0_Fth_N - 2 * r2 / (w0**2))
        Z += np.where(mask, contribution, 0)

# Paramètres de l'animation
frames = 100
times_anim = np.linspace(0, t_max, frames)
N_impulses_per_frame = [min(int(t * f) + 1, N_pulses) for t in times_anim]

# Initialisation de la figure
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
im = ax.imshow(np.zeros_like(X), extent=[-80, 80, -80, 80], cmap='jet', vmin=0, vmax=1400)
ax.set_title("Time = 0.00 µs")
ax.set_xlabel("Position x (µm)")
ax.set_ylabel("Position y (µm)")
fig.colorbar(im, ax=ax, label="Température (°C)")

# Fonction de mise à jour pour l'animation
def update(frame):
    t = times_anim[frame]
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
    
    print(f"Frame {frame + 1}/{frames}, t = {t*1e6:.2f} µs, Temp max = {T_total.max():.2f} °C, Profondeur max = {-Z.min():.2f} µm")
    return [im]

# Créer et sauvegarder l'animation
ani = FuncAnimation(fig, update, frames=frames, interval=100, blit=False)
ani.save('ablation_square_spiral_thermal.gif', writer='pillow', fps=10)
plt.show()

# Carte 2D statique de l'ablation
fig_static = plt.figure(figsize=(8, 6))
ax_static = fig_static.add_subplot(111)
im_static = ax_static.imshow(-Z, extent=[-80, 80, -80, 80], cmap='viridis')
ax_static.set_xlabel('X (µm)')
ax_static.set_ylabel('Y (µm)')
ax_static.set_title('Profondeur d\'ablation finale (Spirale carrée)')
fig_static.colorbar(im_static, ax=ax_static, label='Profondeur (µm)')
plt.tight_layout()
plt.show()

# Carte 3D statique de l'ablation
fig_3d = plt.figure(figsize=(10, 8))
ax_3d = fig_3d.add_subplot(111, projection='3d')
surf = ax_3d.plot_surface(X, Y, -Z, cmap='viridis', edgecolor='none')
ax_3d.set_xlabel('X (µm)')
ax_3d.set_ylabel('Y (µm)')
ax_3d.set_zlabel('Profondeur (µm)')
ax_3d.set_title('Profondeur d\'ablation finale (3D, Spirale carrée)')
fig_3d.colorbar(surf, ax=ax_3d, label='Profondeur (µm)', shrink=0.5, aspect=10)
plt.tight_layout()
plt.show() 

# Profil d'ablation suivant X (X-Z)
fig_x = plt.figure(figsize=(8, 6))
ax_x = fig_x.add_subplot(111)
y_mid = 0  # Centré en y = 0
y_idx = np.argmin(np.abs(y - y_mid))
ax_x.plot(x, -Z[y_idx, :], label=f'Profil X-Z (y={y[y_idx]:.2f} µm)')
ax_x.set_xlabel('X (µm)')
ax_x.set_ylabel('Profondeur (µm)')
ax_x.set_title('Profil d\'ablation suivant X')
ax_x.grid(True)
ax_x.legend()
plt.tight_layout()
plt.show()

#Profil d'ablation suivant Y (Y-Z)
fig_y = plt.figure(figsize=(8, 6))
ax_y = fig_y.add_subplot(111)
x_mid = 0  # Centré en x = 0
x_idx = np.argmin(np.abs(x - x_mid))
ax_y.plot(y, -Z[:, x_idx], label=f'Profil Y-Z (x={x[x_idx]:.2f} µm)')
ax_y.set_xlabel('Y (µm)')
ax_y.set_ylabel('Profondeur (µm)')
ax_y.set_title('Profil d\'ablation suivant Y')
ax_y.grid(True)
ax_y.legend()
plt.tight_layout()
plt.show()

