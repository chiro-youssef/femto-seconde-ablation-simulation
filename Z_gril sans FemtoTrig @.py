import numpy as np
import matplotlib.pyplot as plt
from math import ceil, log, cos, sin, radians

# Fonction pour les effets d'incubation
def varying_params(N_eff, delta0, Fth0, S):
    """
    Ajuste les paramètres delta0 et Fth0 en fonction de N_eff avec l'effet d'incubation.
    """
    power = N_eff**(S - 1)
    return delta0 * power, Fth0 * power

# Paramètres du laser
d = 28  # Diamètre du spot laser (µm)
w0 = d / 2  # Rayon du spot (µm)
F0 = 10  # Fluence de pic (J/cm^2)
Fth0 = 0.055  # Fluence seuil initiale (J/cm^2)
delta0 = 1e-3  # Profondeur de pénétration initiale (µm)
S = 0.8  # Paramètre d'incubation (typique pour métaux)

# Paramètres de la découpe
f = 400e3  # Fréquence de répétition (Hz)
v = 1.5e6  # Vitesse dans la partie droite (µm/s)
Delta_s = v / f  # Distance entre impulsions dans la partie droite (µm)
angle = radians(45)  # Angle de virage (45°)
L_straight = 250  # Longueur de la découpe droite (µm)

# Ajustement pour le virage : maintenir Δx = Delta_s
Delta_x = Delta_s  # Δx constant avant et après le virage
Delta_s_turn = Delta_x / cos(angle)  # Nouvelle distance entre impulsions dans le virage
v_turn = Delta_s_turn * f  # Nouvelle vitesse dans le virage
print(f"Δs (droite) = {Delta_s:.2f} µm, Δs' (virage) = {Delta_s_turn:.2f} µm, v' (virage) = {v_turn:.2f} µm/s")

# Calcul des distances
phi_PO = 1 - Delta_s / d  # Chevauchement des impulsions (partie droite)

# Rayon d'ablation r_th
N_eff = 2 * w0 / Delta_s  # Nombre effectif d'impulsions
delta, Fth = varying_params(N_eff, delta0, Fth0, S)  # Paramètres ajustés
r_th_squared = (w0**2 / 2) * log(F0 / Fth)  # r_th^2 (µm^2)
ln_F0_Fth = log(F0 / Fth)  # Précalcul de ln(F0/Fth)

# Dimensions de la zone simulée (µm)
Lx = 500  # Longueur en x (µm)
Ly = 500  # Longueur en y (µm)

# Grille de points pour x et y (réduite pour optimisation)
nx, ny = 100, 100  # Résolution de la grille
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)

# Initialisation de la matrice Z(x, y)
Z = np.zeros((ny, nx))

# 1. Découpe droite le long de x (y = 250 µm)
n_impulses_straight = ceil(L_straight / Delta_s)
for i in range(n_impulses_straight):
    xi = i * Delta_s
    yi = 250
    r2 = (X - xi)**2 + (Y - yi)**2
    mask = r2 <= r_th_squared
    contribution = delta * (ln_F0_Fth - 2 * r2 / (w0**2))
    Z += np.where(mask, contribution, 0)

# 2. Découpe après le virage à 45°
last_x = (n_impulses_straight - 1) * Delta_s
last_y = 250
x0 = last_x + Delta_x  # Début du virage avec Δx = Delta_s
y0 = last_y + Delta_x * sin(angle) / cos(angle)  # Δy = Δx * tan(45°)
L_remaining = np.sqrt((Lx - x0)**2 + (Ly - y0)**2)
n_impulses_turn = ceil(L_remaining / Delta_s_turn)
for i in range(n_impulses_turn):
    s = i * Delta_s_turn  # Espacement ajusté pour le virage
    xi = x0 + i * Delta_x  # Δx constant
    yi = y0 + i * Delta_x * sin(angle) / cos(angle)  # Δy = Δx * tan(45°)
    if xi > Lx or yi > Ly or xi < 0 or yi < 0:
        continue
    r2 = (X - xi)**2 + (Y - yi)**2
    mask = r2 <= r_th_squared
    contribution = delta * (ln_F0_Fth - 2 * r2 / (w0**2))
    Z += np.where(mask, contribution, 0)

# Calcul du volume ablaté
volume = np.abs(np.trapz(np.trapz(Z, y, axis=0), x))  # Volume en µm³
print(f"Volume ablaté : {volume:.2f} µm³")

# Création des données du tableau
table_data = [
    ["Δs (droite, µm)", f"{Delta_s:.2f}"],
    ["Δs' (virage, µm)", f"{Delta_s_turn:.2f}"],
    ["v' (virage, µm/s)", f"{v_turn:.2f}"],
    ["Volume ablaté (µm³)", f"{volume:.2f}"]
]

# Style professionnel pour les graphiques
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150
})

# Création de la figure principale avec subplots
fig = plt.figure(figsize=(12, 8), constrained_layout=True)
gs = fig.add_gridspec(2, 2, width_ratios=[3, 2], height_ratios=[1, 1])

# Graphique 3D à gauche
ax1 = fig.add_subplot(gs[:, 0], projection='3d')
surf = ax1.plot_surface(X, Y, -Z, cmap='viridis', rstride=1, cstride=1, 
                       linewidth=0.1, antialiased=True, shade=True)
ax1.set_xlabel('x (µm)')
ax1.set_ylabel('y (µm)')
ax1.set_zlabel('Z (µm)')
ax1.set_title(f'Profil 3D (v={v/1e6:.1f} mm/s, f={f/1e3:.0f} kHz)')
cbar = fig.colorbar(surf, ax=ax1, label='Z (µm)', shrink=0.3, pad=0.10, aspect=10)
ax1.view_init(elev=30, azim=135)

# Coupe 1D : z(x) à y = 250 µm (à droite, en haut)
ax2 = fig.add_subplot(gs[0, 1])
y_idx = np.argmin(np.abs(y - 250))
ax2.plot(x, -Z[y_idx, :], color='navy', linewidth=2, marker='o', markersize=4, 
         markevery=10, label='Profondeur (y=250 µm)')
ax2.set_xlabel('x (µm)')
ax2.set_ylabel('Z (µm)')
ax2.set_title('Profil direction X')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Coupe 1D : z(y) à x = 50 µm (à droite, en bas)
ax3 = fig.add_subplot(gs[1, 1])
x_idx = np.argmin(np.abs(x - 50))
ax3.plot(y, -Z[:, x_idx], color='crimson', linewidth=2, marker='s', markersize=2, 
         markevery=10, label='Profondeur (x=50 µm)')
ax3.set_xlabel('y (µm)')
ax3.set_ylabel('Z (µm)')
ax3.set_title('Profil direction Y')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.suptitle('Simulation d’Ablation Laser avec Virage', fontsize=16, fontweight='bold')

# Création du tableau séparé
plt.figure(figsize=(5, 4))
table = plt.table(cellText=table_data, colLabels=['Paramètre', 'Valeur'], 
                  loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)
plt.axis('off')
plt.title('Résultats de la Simulation', fontsize=14, fontweight='bold')
plt.tight_layout()

plt.show()