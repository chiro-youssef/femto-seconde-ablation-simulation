import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image

# Saisie de la couleur pour la courbe Tth
color = input("Choisissez la couleur de la courbe (ex. 'blue', 'red', 'green', etc.) : ")

# Paramètres physiques
w0 = 5e-6  # Rayon du spot (m), 2w0 = 50 µm
f_rep = 800e3  # Fréquence de répétition (Hz)
v = 4  # Vitesse de balayage (m/s)
F = 0.37  # Fluence (J/cm²)
eta_res = 0.4  # Fraction de chaleur résiduelle
T_offset = 0  # Température initiale (°C)
rho = 7920  # Densité de l'acier (kg/m³)
c = 500  # Capacité calorifique (J/kg·K)
kappa = 4e-7  # Diffusivité thermique (m²/s)

# Calcul de l'énergie résiduelle et de l'élévation initiale de température
spot_area = np.pi * w0**2
E_p = F * 1e4 * spot_area  # Conversion de J/cm² à J/m²
E_res = eta_res * E_p
penetration_depth = 19e-9
volume = spot_area * penetration_depth
mass = rho * volume
delta_T_initial = E_res / (mass * c)
print(f"Température initiale par impulsion : {delta_T_initial:.2f} °C")

# Paramètres temporels
delta_t = 1 / f_rep  # Intervalle entre impulsions (s)
N_pulses = 20  # Nombre d'impulsions
t_max = N_pulses * delta_t

# Grille spatiale 2D
nx, ny = 200, 200
x = np.linspace(-20e-6, 250e-6, nx)  # x de -20 µm à 250 µm
y = np.linspace(-100e-6, 100e-6, ny)  # y de -100 µm à 100 µm
X, Y = np.meshgrid(x, y)

# Points temporels pour les courbes et l'animation
t_eval_per_pulse = np.linspace(0, delta_t, 50)  # 50 points par impulsion
t_eval = []  # Temps en µs pour les courbes
Tth = []  # Température au centre de l'impulsion
T_max_values = []  # Température maximale (identique à Tth)

# Fonction pour la température d'une impulsion unique
def T_single_pulse(x, y, z, t, x0, y0, E_res, w0, rho, c, kappa):
    if t <= 1e-12:  # Juste après l'impulsion (impulsion ultra-courte)
        return delta_T_initial
    denom = np.pi * rho * c * np.sqrt(np.pi * kappa * t) * (8 * kappa * t + w0**2)
    num = 2 * E_res
    arg_exp = ((x - x0)**2 + (y - y0)**2)* (w0**2 /(8 * kappa * t + w0**2)  -1 ) / (4 * kappa * t ) - z**2 / (4 * kappa * t)
    return num / denom * np.exp(arg_exp)

# Calcul des courbes Tth et T_max_values
for i in range(N_pulses):
    t_pulse = i * delta_t
    x0_i = i * v * delta_t  # Position x de l'impulsion i
    t_current = t_pulse + t_eval_per_pulse
    t_eval.extend(t_current * 1e6)  # Convertir en µs
    for t in t_eval_per_pulse:
        T = T_offset
        x_eval = x0_i
        y_eval = 0
        for j in range(i + 1):
            t_j = t + (i - j) * delta_t
            x0_j = j * v * delta_t
            T += T_single_pulse(x_eval, y_eval, 0, t_j, x0_j, 0, E_res, w0, rho, c, kappa)
        Tth.append(T)
        T_max_values.append(T)

# Points temporels pour l'animation (sous-ensemble de t_eval)
frames =    50
indices = np.linspace(0, len(t_eval) - 1, frames, dtype=int)  # Sélectionner 10 points
times = [t_eval[i] / 1e6 for i in indices]  # Convertir en secondes
N_impulses = [int(t * f_rep) for t in times]

# Initialisation de la figure pour la carte thermique
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
im = ax.imshow(np.zeros_like(X), extent=[-20, 250, -100, 100], cmap='jet', vmin=0, vmax=1000)
ax.set_title("Time = 0.00 µs")
ax.set_xlabel("Position x (µm)")
ax.set_ylabel("Position y (µm)")
fig.colorbar(im, ax=ax, label="Température (°C)")
plt.tight_layout()

# Fonction de mise à jour pour l'animation
def update(frame):
    for coll in ax.collections:
        coll.remove()
    for line in ax.lines:
        line.remove()  # Supprimer les marqueurs précédents
    
    t = times[frame]
    N = N_impulses[frame]
    T_total = np.zeros_like(X)
    
    # Calculer la température sur la grille 2D
    for j in range(N + 1):
        t_j = t - j * delta_t
        x0_j = j * v * delta_t
        y0_j = 0
        if t_j <= 0:
            continue
        temp_contribution = T_single_pulse(X, Y, 0, t_j, x0_j, y0_j, E_res, w0, rho, c, kappa)
        T_total += temp_contribution
    
    T_total += T_offset
    
    # Position du centre de l'impulsion courante
    x0_current = N * v * delta_t
    idx_x = np.argmin(np.abs(x - x0_current))
    idx_y = np.argmin(np.abs(y - 0))
    T_center = T_total[idx_y, idx_x]
    
    # Température de référence (Tth/T_max_values)
    idx_t_eval = indices[frame]
    T_ref = Tth[idx_t_eval]
    
    # Afficher les informations
    print(f"Frame {frame + 1}/{frames}, t = {t*1e6:.2f} µs")
    print(f"Temp max (grille) = {T_total.max():.2f} °C")
    print(f"Temp au centre (x={x0_current*1e6:.2f} µm, y=0) = {T_center:.2f} °C")
    print(f"Temp de référence (Tth/T_max_values) = {T_ref:.2f} °C")
    
    # Mettre à jour la carte thermique
    im.set_array(T_total)
    ax.set_title(f"Time = {t*1e6:.2f} µs")
    ax.contour(X*1e6, Y*1e6, T_total, levels=5, colors='white', alpha=0.5)
    ax.plot(x0_current*1e6, 0, 'x', color='black', markersize=10, label='Centre du faisceau')  # Marqueur du faisceau
    ax.legend()
    
    return [im]

# Créer et sauvegarder l'animation
ani = FuncAnimation(fig, update, frames=frames, interval=1000, blit=True)
ani.save('heat_distribution_xy.gif', writer='pillow', fps=5)

# Graphique des courbes de température
fig2 = plt.figure(figsize=(8, 6))
ax2 = fig2.add_subplot(111)
ax2.plot(t_eval, T_max_values, label='Température maximale (centre)', color='blue')
ax2.plot(t_eval, Tth, label='Tth (centre de l\'impulsion)', color=color, linestyle='--')
ax2.axhline(y=3000, color='red', linestyle='--', label='Seuil de saturation (3000 °C)')
ax2.set_xlabel('Temps (µs)')
ax2.set_ylabel('Température (°C)')
ax2.set_title('Variation de la température maximale à v = 100 m/s, F = 0.37 J/cm²')
ax2.legend()
ax2.grid(True)
ax2.set_yticks(np.arange(0, 3500, 500))
ax2.set_ylim(0, 3500)
plt.tight_layout()

plt.show()