import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from math import log

# Paramètres de l'expérience
w0 = 14e-6  # Rayon du spot (m)
f = 800e3  # Fréquence de répétition (Hz)
F0 = 10.0  # Fluence de pic (J/cm²), augmentée
Fth = 0.1  # Fluence seuil (J/cm²)
delta = 18e-9  # Profondeur de pénétration (m)
eta_res = 0.4  # Fraction de chaleur résiduelle
T_offset = 0  # Température initiale (°C)
rho = 7920  # Densité de l'acier (kg/m³)
c = 500  # Capacité calorifique (J/kg·K), corrigée
kappa = 1e-7  # Diffusivité thermique (m²/s), réduite

# Calcul de l'énergie résiduelle
spot_area = np.pi * w0**2
E_p = F0 * 1e4 * spot_area
E_res = eta_res * E_p

# Volume chauffé et température initiale
penetration_depth = 18e-9  # Réduit pour augmenter la température
volume = spot_area * penetration_depth
mass = rho * volume
delta_T_initial = E_res / (mass * c)
print(f"Température initiale par impulsion : {delta_T_initial:.2f} °C")

# Paramètres temporels
delta_t = 1 / f
N_pulses = 20
t_max = N_pulses * delta_t

# Dimensions de la zone simulée (m)
Lx = 100e-6
Ly = 100e-6

# Grille spatiale 2D (plan x, y)
nx, ny = 400, 400
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)

# Paramètres de la spirale d'Archimède
a = (10e-6) / (5 * np.pi)  # Espacement pour r_max ≈ 50 µm après 5 tours
theta_max = 10 * np.pi  # 5 tours
v0 = 1.0  # Vitesse initiale (m/s)
k = 0  # Pas d'augmentation de vitesse

# Générer des points pour la spirale
theta_dense = np.linspace(0, theta_max, 1000)
r_dense = a * theta_dense
x_dense = r_dense * np.cos(theta_dense) + Lx/2
y_dense = r_dense * np.sin(theta_dense) + Ly/2

# Sélectionner les points en fonction de l'espacement
spiral_x, spiral_y = [x_dense[0]], [y_dense[0]]  # Première impulsion au centre
thetas = [0.0]
last_x, last_y = x_dense[0], y_dense[0]

for i in range(1, len(theta_dense)):
    theta = theta_dense[i]
    v_theta = v0 + k * theta
    Delta = v_theta / f
    dist = np.sqrt((x_dense[i] - last_x)**2 + (y_dense[i] - last_y)**2)
    if dist >= Delta:
        spiral_x.append(x_dense[i])
        spiral_y.append(y_dense[i])
        thetas.append(theta)
        last_x, last_y = x_dense[i], y_dense[i]

spiral_x, spiral_y = np.array(spiral_x), np.array(spiral_y)

# Instants pour l'animation
frames = 50
times = np.linspace(0, t_max, frames)
N_impulses = [int(t * f) for t in times]

# Fonction pour la température d'une impulsion unique
def T_single_pulse(x, y, t, x0, y0, E_res, w0, rho, c, kappa):
    if t <= 1e-12:
        return delta_T_initial * np.exp(-2 * ((x - x0)**2 + (y - y0)**2) / w0**2)
    denom = rho * c * (4 * np.pi * kappa * t)
    num = E_res
    arg_exp = -((x - x0)**2 + (y - y0)**2) / (4 * kappa * t)
    return num / denom * np.exp(arg_exp)

# Fonction de profondeur d'ablation
def ablation_depth(x, y, t, spiral_x, spiral_y, w0, f, F0=2.0, Fth=0.1, delta=0.5e-6):
    Z = np.zeros_like(x)
    ln_F0_Fth = log(F0 / Fth)
    r_th_squared = (w0**2 / 2) * ln_F0_Fth
    N = min(int(t * f), len(spiral_x) - 1)
    for i in range(N + 1):
        xi, yj = spiral_x[i], spiral_y[i]
        if 0 <= xi <= Lx and 0 <= yj <= Ly:
            r2 = (x - xi)**2 + (y - yj)**2
            mask = r2 <= r_th_squared
            contribution = delta * (ln_F0_Fth - 2 * r2 / (w0**2))
            Z += np.where(mask, contribution, 0)
    return np.where(Z >= 0, Z, 0)

# Initialisation de la figure
fig = plt.figure(figsize=(15, 10))

# Carte de chaleur 2D (haut)
ax1 = fig.add_subplot(2, 1, 1)
im = ax1.imshow(np.zeros((ny, nx)), extent=[0, Lx * 1e6, 0, Ly * 1e6], cmap='jet', vmin=0, vmax=300)
ax1.set_title("Time = 0.00 µs")
ax1.set_xlabel("Position x (µm)")
ax1.set_ylabel("Position y (µm)")
fig.colorbar(im, ax=ax1, label="Température (°C)")

# Sous-graphiques pour tranches (bas)
ax2 = fig.add_subplot(2, 1, 2)
ax2.set_axis_off()

# Créer plusieurs tranches radiales ou angulaires
n_slices = 10  # Nombre de tranches
angles = np.linspace(0, 2 * np.pi, n_slices, endpoint=False)  # Angles pour les tranches
slice_axes = []

for i in range(n_slices):
    ax_temp = fig.add_axes([0.1 + i*0.08, 0.1, 0.03, 0.35])  # Position des tranches de température
    ax_depth = fig.add_axes([0.1 + i*0.08, 0.05, 0.03, 0.05])  # Position des tranches de profondeur
    slice_axes.append((ax_temp, ax_depth))
    ax_temp.set_xticks([])
    ax_temp.set_yticks(np.linspace(0, 50, 5))  # Rayon max ≈ 50 µm
    ax_depth.set_xticks([])
    ax_depth.set_yticks(np.linspace(-2, 0, 5))
    if i == 0:
        ax_temp.set_ylabel("Rayon (µm)")
        ax_depth.set_ylabel("Profondeur (µm)")
    ax_temp.set_title(f"θ={angles[i]*180/np.pi:.0f}°", fontsize=8)

# Fonction de mise à jour pour l'animation
def update(frame):
    for ax_temp, ax_depth in slice_axes:
        ax_temp.clear()
        ax_depth.clear()
        ax_temp.set_xticks([])
        ax_depth.set_xticks([])
        ax_temp.set_yticks(np.linspace(0, 50, 5))
        ax_depth.set_yticks(np.linspace(-2, 0, 5))

    t = times[frame]
    N = N_impulses[frame]
    T_total = np.zeros((ny, nx))
    
    # Calculer la profondeur d'ablation
    Z0 = ablation_depth(X, Y, t, spiral_x, spiral_y, w0, f, F0, Fth, delta)
    
    # Sommer les contributions de température
    for j in range(min(N + 1, len(spiral_x))):
        t_j = t - j * delta_t
        x0_j, y0_j = spiral_x[j], spiral_y[j]
        if t_j <= 0:
            continue
        temp_contribution = T_single_pulse(X, Y, t_j, x0_j, y0_j, E_res, w0, rho, c, kappa)
        T_total += temp_contribution

    T_total += T_offset
    mask = Z0 > 0.1e-6
    T_total[mask] = 0

    im.set_array(T_total)
    ax1.set_title(f"Time = {t*1e6:.2f} µs")
    ax1.contour(X*1e6, Y*1e6, T_total, levels=5, colors='white', alpha=0.5)

    # Afficher les tranches radiales
    for i, (ax_temp, ax_depth) in enumerate(slice_axes):
        angle = angles[i]
        # Calculer les distances radiales par rapport au centre
        r = np.sqrt((X - Lx/2)**2 + (Y - Ly/2)**2)
        theta = np.arctan2(Y - Ly/2, X - Lx/2)
        # Sélectionner les points proches de l'angle donné (tolérance de 10°)
        mask_angle = np.abs((theta - angle + np.pi) % (2 * np.pi) - np.pi) < np.pi / 18
        r_slice = r[mask_angle]
        temp_slice = T_total[mask_angle]
        depth_slice = -Z0[mask_angle] * 1e6  # Profondeur en µm

        # Trier par rayon pour une visualisation cohérente
        sort_idx = np.argsort(r_slice.flatten())
        r_slice = r_slice.flatten()[sort_idx] * 1e6  # Convertir en µm
        temp_slice = temp_slice.flatten()[sort_idx]
        depth_slice = depth_slice.flatten()[sort_idx]

        # Afficher la température
        ax_temp.imshow(temp_slice[:, np.newaxis], extent=[0, 1, 0, 50], cmap='jet', vmin=0, vmax=3000, aspect='auto')
        ax_temp.set_title(f"θ={angles[i]*180/np.pi:.0f}°", fontsize=8)

        # Afficher la profondeur
        ax_depth.imshow(depth_slice[:, np.newaxis], extent=[0, 1, -2, 0], cmap='viridis', vmin=0, vmax=2, aspect='auto')

    print(f"Frame {frame + 1}/{frames}, t = {t*1e6:.2f} µs, Temp max = {T_total.max():.2f} °C, Profondeur max = {Z0.max()*1e6:.2f} µm")
    return [im] + [ax for ax_pair in slice_axes for ax in ax_pair]

# Créer l'animation
ani = FuncAnimation(fig, update, frames=frames, interval=100, blit=False)

# Sauvegarder en GIF
ani.save('spiral_heat_and_depth_distribution_with_slices.gif', writer='pillow', fps=10)

# Afficher (facultatif)
plt.show()