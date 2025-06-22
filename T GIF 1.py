import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image

# Paramètres de l'expérience
w0 = 25e-6  # Rayon du spot (m), 2w0 = 50 µm
f_rep = 800e3  # Fréquence de répétition (Hz)
v = 20.0  # Vitesse de balayage (m/s)
F = 0.37  # Fluence (J/cm²)
eta_res = 0.4  # Fraction de chaleur résiduelle
T_offset = 0  # Température initiale (°C)
rho = 7920  # Densité de l'acier (kg/m³)
c = 10  # Capacité calorifique (J/kg·K)
kappa = 1e-6  # Diffusivité thermique (m²/s)

# Calcul de l'énergie résiduelle
spot_area = np.pi * w0**2  # Surface du spot (m²)
E_p = F * 1e4 * spot_area  # Énergie par impulsion (J), conversion J/cm² -> J/m²
E_res = eta_res * E_p  # Chaleur résiduelle par impulsion (J)

# Volume chauffé pour estimer la température initiale
penetration_depth = 0.1e-6  # Profondeur de pénétration (m), ajustée pour atteindre ~3000 °C
volume = spot_area * penetration_depth  # Volume chauffé (m³)
mass = rho * volume  # Masse du volume chauffé (kg)
delta_T_initial = E_res / (mass * c)  # Hausse de température initiale (°C)
print(f"Température initiale par impulsion : {delta_T_initial:.2f} °C")

# Paramètres temporels
delta_t = 1 / f_rep  # Intervalle entre impulsions (s)
N_pulses = 50  # Nombre d'impulsions (pour atteindre ~20 µs)
t_max = N_pulses * delta_t  # Temps total simulé (s)

# Grille spatiale 2D (plan x, z) - Augmentation de la résolution
x = np.linspace(0, t_max * v, 400)  # m (axe x, direction de balayage, de 0 à la distance parcourue)
z = np.linspace(-100e-6, 100e-6, 800)  # m (axe z, profondeur dans le matériau)
X, Z = np.meshgrid(x, z)

# Instants pour l'animation (plus d'échantillons pour une animation fluide)
frames = 50  # Nombre de frames pour l'animation
times = np.linspace(0, t_max, frames)  # Temps uniformément espacés
N_impulses = [int(t * f_rep) for t in times]

# Modèle d'ablation simplifié : profondeur de la tranchée en fonction de x et du temps
def ablation_depth(x, t, w0, v, delta_t, max_depth=50e-6):
    """Modèle simplifié de la profondeur d'ablation (gaussienne)"""
    depth = np.zeros_like(x)
    for j in range(int(t * f_rep) + 1):
        t_j = t - j * delta_t
        x0_j = j * v * delta_t  # Position x de l'impulsion j
        if t_j <= 0:
            continue
        contribution = min(max_depth * t_j / (15 * delta_t), max_depth) * np.exp(-2 * ((x - x0_j)**2) / w0**2)
        depth += contribution
    return depth  # Retourne une profondeur 1D en fonction de x

# Fonction pour la température d'une impulsion unique, ajustée pour grilles 2D
def T_single_pulse(x, z, t, x0, y0, E_res, w0, rho, c, kappa):
    if t <= 1e-12:  # Juste après l'impulsion (impulsion ultra-courte)
        return delta_T_initial * np.exp(-2 * ((x - x0)**2) / w0**2)
    denom = rho * c * (4 * np.pi * kappa * t)**(3/2)
    num = E_res
    arg_exp = -((x - x0)**2 + z**2) / (4 * kappa * t)
    return num / denom * np.exp(arg_exp)

# Initialisation de la figure avec deux sous-graphiques
fig = plt.figure(figsize=(10, 10))
# Sous-graphique pour la carte de chaleur (x, z)
ax1 = fig.add_subplot(2, 1, 1)
im = ax1.imshow(np.zeros_like(X), extent=[0, t_max * v * 1e6, -100, 100], cmap='jet', vmin=0, vmax=5000)
ax1.set_title("Time = 0.00 µs")
ax1.set_xlabel("Position x (µm)")
ax1.set_ylabel("Position z (µm)")
fig.colorbar(im, ax=ax1, label="Température (°C)")

# Sous-graphique pour la profondeur d'ablation (x)
ax2 = fig.add_subplot(2, 1, 2)
depth_line, = ax2.plot(x * 1e6, np.zeros_like(x), 'b-', label='Profondeur d\'ablation')
ax2.set_xlim(0, t_max * v * 1e6)
ax2.set_ylim(-70, 0)  # Limites pour la profondeur (µm)
ax2.set_xlabel("Position x (µm)")
ax2.set_ylabel("Profondeur (µm)")
ax2.grid(True)
ax2.legend()

# Ajuster la mise en page
plt.tight_layout()

# Fonction de mise à jour pour l'animation
def update(frame):
    # Supprimer les anciens contours s'ils existent
    for coll in ax1.collections:
        coll.remove()
    
    t = times[frame]
    N = N_impulses[frame]
    T_total = np.zeros_like(X)
    
    # Calculer la profondeur d'ablation pour cet instant
    Z0 = ablation_depth(x, t, w0, v, delta_t)  # Utiliser x 1D au lieu de X 2D
    
    # Sommer les contributions de toutes les impulsions
    for j in range(N + 1):
        t_j = t - j * delta_t
        x0_j = j * v * delta_t
        y0_j = 0
        if t_j <= 0:
            continue
        temp_contribution = T_single_pulse(X, Z, t_j, x0_j, y0_j, E_res, w0, rho, c, kappa)
        T_total += temp_contribution

    # Ajuster la température pour inclure l'offset
    T_total += T_offset

    # Appliquer un masque pour simuler la tranchée
    mask = Z < -Z0[np.newaxis, :]  # Étendre Z0 1D à 2D pour correspondre à Z
    T_total[mask] = 0

    # Mettre à jour la carte de chaleur
    im.set_array(T_total)
    ax1.set_title(f"Time = {t*1e6:.2f} µs")
    ax1.contour(X*1e6, Z*1e6, T_total, levels=5, colors='white', alpha=0.5)

    # Mettre à jour la profondeur d'ablation
    depth_line.set_ydata(-Z0 * 1e6)  # Utiliser Z0 1D directement

    print(f"Frame {frame + 1}/{frames}, t = {t*1e6:.2f} µs, Temp max = {T_total.max():.2f} °C, Profondeur max = {Z0.max()*1e6:.2f} µm")
    return [im, depth_line]

# Créer l'animation
ani = FuncAnimation(fig, update, frames=frames, interval=100, blit=True)

# Sauvegarder en GIF
ani.save('heat_and_depth_distribution.gif', writer='pillow', fps=10)

# Afficher l'animation (facultatif)
plt.show()