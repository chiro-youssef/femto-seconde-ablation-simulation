import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image
import matplotlib

# Paramètres de l'expérience
w0 = 25e-6  # Rayon du spot (m), 2w0 = 50 µm
f_rep = 800e3  # Fréquence de répétition (Hz)
v = 4.0  # Vitesse de balayage (m/s)
F = 0.37  # Fluence (J/cm²)
eta_res = 0.4  # Fraction de chaleur résiduelle
T_offset = 0  # Température initiale (°C)
rho = 7920  # Densité de l'acier (kg/m³)
c = 500  # Capacité calorifique (J/kg·K)
kappa = 3e-7  # Diffusivité thermique très réduite pour limiter la propagation (m²/s)

# Calcul de l'énergie résiduelle
spot_area = np.pi * w0**2  # Surface du spot (m²)
E_p = F * 1e4 * spot_area  # Énergie par impulsion (J), conversion J/cm² -> J/m²
E_res = eta_res * E_p  # Chaleur résiduelle par impulsion (J)

# Volume chauffé pour estimer la température initiale
penetration_depth = 100e-6  # Profondeur de pénétration (m), ajustée pour atteindre ~3000 °C
volume = spot_area * penetration_depth  # Volume chauffé (m³)
mass = rho * volume  # Masse du volume chauffé (kg)
delta_T_initial = E_res / (mass * c)  # Hausse de température initiale (°C)
print(f"Température initiale par impulsion : {delta_T_initial:.2f} °C")

# Paramètres temporels
delta_t = 1 / f_rep  # Intervalle entre impulsions (s)
N_pulses = 30  # Nombre d'impulsions (pour atteindre ~20 µs)
t_max = N_pulses * delta_t  # Temps total simulé (s)

# Grille spatiale 2D (plan x, z) - Résolution augmentée
x = np.linspace(0, t_max * v, 800)  # Résolution augmentée pour détails
z = np.linspace(-100e-6, 100e-6, 800)  # m (axe z, profondeur dans le matériau)
X, Z = np.meshgrid(x, z)

# Instants pour l'animation
frames = 30  # Nombre de frames pour l'animation
times = np.linspace(0, t_max, frames)  # Temps uniformément espacés
N_impulses = [int(t * f_rep) for t in times]

# Modèle d'ablation : profondeur entre 0 et 1 µm
def ablation_depth(x, t, w0, v, delta_t, max_depth=1e-6):
    """Modèle corrigé de la profondeur d'ablation avec saturation"""
    depth = np.zeros_like(x)
    n_impulses = int(t * f_rep) + 1  # Nombre total d'impulsions jusqu'à t
    for j in range(n_impulses):
        x0_j = j * v * delta_t  # Position x de l'impulsion j
        contribution = delta * np.exp(-2 * ((x - x0_j)**2) / w0**2)
        depth = np.maximum(depth, contribution)  # Prendre la profondeur maximale
    depth = np.minimum(depth * (t / t_max), max_depth)  # Limiter à max_depth
    return depth

# Profondeur par impulsion
delta = 0.1e-6  # Profondeur par impulsion (0.1 µm), ajustée pour limiter entre 0 et 1 µm

# Fonction pour la température d'une impulsion unique
def T_single_pulse(X_m, Z_m, x0_m, z0_m, t, E_res, w0_m, rho, c, kappa, z=0):
    if t <= 1e-12:  # Juste après l'impulsion
        return delta_T_initial * np.exp(-2 * ((X_m - x0_m)**2 + (Z_m - z0_m)**2) / w0_m**2)
    denom = np.pi * rho * c * np.sqrt(np.pi * kappa * t) * (8 * kappa * t + w0_m**2)
    num = E_res
    arg_exp = ((X_m - x0_m)**2 + (Z_m - z0_m)**2)*(w0_m**2/(8 * kappa * t + w0_m**2)-1) / (4 * kappa * t) - z**2 / (4 * kappa * t)
    return num / denom * np.exp(arg_exp)


# Initialisation de la figure avec deux sous-graphiques
fig = plt.figure(figsize=(14, 14), dpi=120, facecolor='white')  # Taille et résolution augmentées

# Sous-graphique pour la carte de chaleur (x, z)
ax1 = fig.add_subplot(2, 1, 1)
im = ax1.imshow(np.zeros_like(X), extent=[0, t_max * v * 1e6, -100, 100], cmap='jet', vmin=0, vmax=500)  # jet pour contraste, vmax réduit
ax1.set_title("Time = 0.00 µs, Temp max = 0.00 °C", fontsize=16, pad=20)
ax1.set_xlabel("Position x (µm)", fontsize=14, labelpad=15)
ax1.set_ylabel("Position z (µm)", fontsize=14, labelpad=15)
ax1.tick_params(axis='both', labelsize=12)

# Barre de couleur avec des couleurs distinctes pour chaque intensité
cbar = fig.colorbar(im, ax=ax1)
cbar.set_label("Température (°C)", fontsize=14, labelpad=15)
cbar.ax.tick_params(labelsize=12)

# Sous-graphique pour la profondeur d'ablation (x)
ax2 = fig.add_subplot(2, 1, 2)
depth_line, = ax2.plot(x * 1e6, np.zeros_like(x), 'b-', label='Profondeur d\'ablation', linewidth=2.5)
# Marqueurs pour les positions des impulsions
impulse_positions = [j * v * delta_t * 1e6 for j in range(N_pulses)]
ax2.scatter(impulse_positions, np.zeros_like(impulse_positions), c='red', marker='v', s=50, label='Positions des impulsions')
ax2.set_xlim(0, t_max * v * 1e6)
ax2.set_ylim(-0.2, 0)  # Limites ajustées pour 0 à 1 µm
ax2.set_xlabel("Position x (µm)", fontsize=14, labelpad=15)
ax2.set_ylabel("Profondeur (µm)", fontsize=14, labelpad=15)
ax2.tick_params(axis='both', labelsize=12)
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.legend(fontsize=12)

# Ajuster la mise en page
plt.tight_layout(pad=4.0)

# Fonction de mise à jour pour l'animation
def update(frame):
    # Supprimer les anciens contours
    for coll in ax1.collections:
        coll.remove()
    
    t = times[frame]
    N = N_impulses[frame]
    T_total = np.zeros_like(X)
    
    # Calculer la profondeur d'ablation
    Z0 = ablation_depth(x, t, w0, v, delta_t)
    
    # Sommer les contributions de toutes les impulsions
    for j in range(N + 1):
        t_j = t - j * delta_t
        x0_j = j * v * delta_t
        z0_j = 0  # Ajusté pour correspondre à z=0 dans T_single_pulse
        if t_j <= 0:
            continue
        temp_contribution = T_single_pulse(X, Z, x0_j, z0_j, t_j, E_res, w0, rho, c, kappa, z=0)
        T_total += temp_contribution

    # Ajuster la température pour inclure l'offset
    T_total += T_offset

    # Appliquer un masque pour simuler la tranchée
    mask = Z < -1-Z0[np.newaxis, :]
    T_total[mask] = 0

    # Mettre à jour la carte de chaleur
    im.set_array(T_total)
    ax1.set_title(f"Time = {t*1e6:.2f} µs, Temp max = {T_total.max():.2f} °C", fontsize=16, pad=20)
    # Contours avec couleurs distinctes pour chaque intensité
    levels = np.linspace(0, 1000, 10)  # 10 niveaux de température entre 0 et 1000 °C
    contour = ax1.contour(X*1e6, Z*1e6, T_total, levels=levels, cmap='cool', linewidths=1.5, alpha=0.8)

    # Mettre à jour la profondeur d'ablation
    depth_line.set_ydata(-Z0 * 1e6)

    print(f"Frame {frame + 1}/{frames}, t = {t*1e6:.2f} µs, Temp max = {T_total.max():.2f} °C, Profondeur max = {Z0.max()*1e6:.2f} µm")
    return [im, depth_line]

# Créer l'animation
ani = FuncAnimation(fig, update, frames=frames, interval=100, blit=True)

# Sauvegarder en GIF
ani.save('heat_and_depth_distribution_optimized.gif', writer='pillow', fps=10)

# Afficher l'animation (facultatif)
plt.show()