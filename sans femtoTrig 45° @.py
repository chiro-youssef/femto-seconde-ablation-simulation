# [Le début de votre code reste inchangé : importations, paramètres, calculs, etc.]
import numpy as np
import matplotlib.pyplot as plt
from math import ceil, log, cos, sin, radians
import matplotlib.colors as mcolors

# Fonction pour les effets d'incubation
def varying_params(N_eff, delta0, Fth0, S):
    power = N_eff**(S - 1)
    return delta0 * power, Fth0 * power

# Paramètres du laser
d = 28  # Diamètre du spot laser (µm)
w0 = d / 2  # Rayon du spot (µm)
F0 = 10  # Fluence de pic (J/cm^2)
Fth0 = 0.055  # Fluence seuil initiale (J/cm^2)
delta0 = 18e-3  # Profondeur de pénétration initiale (µm)
S = 0.8  # Paramètre d'incubation (typique pour métaux)

# Paramètres de la découpe
f = 400e3  # Fréquence (Hz)
v = 1.5e6  # Vitesse dans la partie droite (µm/s)
Delta_s = v / f  # Distance entre impulsions dans la partie droite (µm)
angle = radians(45)  # Angle de virage (45°)
L_straight = 60  # Longueur de la découpe droite (µm)

# Ajustement pour le virage : maintenir Δx = Delta_s
Delta_x = Delta_s  # Δx constant avant et après le virage
Delta_s_turn = Delta_x / cos(angle)  # Nouvelle distance entre impulsions dans le virage
v_turn = Delta_s_turn * f  # Nouvelle vitesse dans le virage
print(f"Δs (droite) = {Delta_s} µm, Δs' (virage) = {Delta_s_turn} µm, v' (virage) = {v_turn} µm/s")

# Calcul des distances
phi_PO = 1 - Delta_s / d  # Chevauchement des impulsions (partie droite)

# Rayon d'ablation r_th
r_th_squared = (w0**2 / 2) * log(F0 / Fth0)  # r_th^2 (µm^2)

# Calcul de N_eff pour l'incubation
N_eff = 2 * w0 / Delta_s  # Nombre effectif d'impulsions
delta, Fth = varying_params(N_eff, delta0, Fth0, S)  # Paramètres ajustés

# Dimensions de la zone simulée (µm)
Lx = 150  # Longueur en x (µm)
Ly = 200  # Longueur en y (µm)

# Grille de points pour x et y
nx, ny = 200, 200  # Résolution de la grille
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)

# Calcul de ln(F0/Fth)
ln_F0_Fth = log(F0 / Fth)

# Initialisation de la matrice Z(x, y)
Z = np.zeros((ny, nx))

# 1. Découpe droite le long de x (y = 250 µm)
n_impulses_straight = ceil(L_straight / Delta_s)
for i in range(n_impulses_straight):
    xi = i * Delta_s
    yi = 60
    r2 = (X - xi)**2 + (Y - yi)**2
    mask = r2 <= r_th_squared
    contribution = delta * (ln_F0_Fth - 2 * r2 / (w0**2))
    Z += np.where(mask, contribution, 0)

# 2. Découpe après le virage à 45°
last_x = (n_impulses_straight - 1) * Delta_s
last_y = 60
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

# Style professionnel pour les graphiques (inspiré du premier code)
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 12,
    'axes.titlesize': 5,
    'axes.labelsize': 6,
    'legend.fontsize': 5,
    'xtick.labelsize': 5,
    'ytick.labelsize': 5,
    'figure.dpi': 200
})

# Figure avec subplots (structure du premier code)
fig = plt.figure(figsize=(12, 8), constrained_layout=True)
gs = fig.add_gridspec(2, 2, width_ratios=[3, 2], height_ratios=[1, 1])

# Graphique 3D à gauche
ax1 = fig.add_subplot(gs[:, 0], projection='3d')
surf = ax1.plot_surface(X, Y, Z, cmap='viridis', rstride=1, cstride=1, 
                        linewidth=0.1, antialiased=True, shade=True)
ax1.set_xlabel('x (µm)')
ax1.set_ylabel('y (µm)')
ax1.set_zlabel('z (µm)')
ax1.xaxis.labelpad = -10  # Rapproche l'étiquette "X (µm)"
ax1.yaxis.labelpad = -10  # Rapproche l'étiquette "Y (µm)"
ax1.zaxis.labelpad = -7   # Rapproche l'étiquette "Z (µm)"
ax1.xaxis.set_tick_params(pad=-5)  # Rapproche les chiffres de l'axe X
ax1.yaxis.set_tick_params(pad=-5)  # Rapproche les chiffres de l'axe Y
ax1.zaxis.set_tick_params(pad=-3)  # Rapproche les chiffres de l'axe Z
ax1.invert_zaxis()  # Inverse l'axe Z pour montrer l'ablation vers le bas
cbar = fig.colorbar(surf, ax=ax1, label='z (µm)', shrink=0.5, pad=0.20, aspect=20)
ax1.xaxis.pane.set_edgecolor('black')
ax1.yaxis.pane.set_edgecolor('black')
ax1.zaxis.pane.set_edgecolor('black')
ax1.xaxis.pane.set_linewidth(0.5)
ax1.yaxis.pane.set_linewidth(0.5)
ax1.zaxis.pane.set_linewidth(0.5)

# Graphique 2D (vue de dessus) en haut à droite, remplace la courbe X
ax2 = fig.add_subplot(gs[0, 1])
colors = [(0.7, 0.7, 0.7)]  # Gris pour Z = 0
cmap_hot = plt.get_cmap('hot')
colors += [cmap_hot(i / 255) for i in range(255)]
custom_cmap = mcolors.ListedColormap(colors)
levels = [0] + list(np.linspace(0.001, np.max(Z), 10))
contour = ax2.contourf(X, Y, Z, levels=levels, cmap=custom_cmap)
fig.colorbar(contour, ax=ax2, label='z (µm)', shrink=0.9, aspect=20, pad=0)
ax2.set_xlabel('x (µm)')
ax2.set_ylabel('y (µm)')
ax2.grid(True, alpha=0.3)
ax2.spines['top'].set_visible(True)
ax2.spines['right'].set_visible(True)
ax2.spines['left'].set_visible(True)
ax2.spines['bottom'].set_visible(True)
ax2.spines['top'].set_linewidth(0.5)
ax2.spines['right'].set_linewidth(0.5)
ax2.spines['left'].set_linewidth(0.5)
ax2.spines['bottom'].set_linewidth(0.5)
ax2.spines['top'].set_color('black')
ax2.spines['right'].set_color('black')
ax2.spines['left'].set_color('black')
ax2.spines['bottom'].set_color('black')

# Courbe 2D : z(y) à x = 50 µm (en bas à droite, couleur crimson)
ax3 = fig.add_subplot(gs[1, 1])
y_slice = y
x_idx = np.argmin(np.abs(x - 50))
z_slice = -Z[:, x_idx]  # Négatif pour correspondre à l'ablation (comme dans le premier code)
ax3.plot(y_slice, z_slice, color='crimson', linewidth=1, marker='s', markersize=0, 
         markevery=10, )
ax3.set_xlabel('y (µm)')
ax3.set_ylabel('z (µm)')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.spines['top'].set_visible(True)
ax3.spines['right'].set_visible(True)
ax3.spines['left'].set_visible(True)
ax3.spines['bottom'].set_visible(True)
ax3.spines['top'].set_linewidth(0.5)
ax3.spines['right'].set_linewidth(0.5)
ax3.spines['left'].set_linewidth(0.5)
ax3.spines['bottom'].set_linewidth(0.5)
ax3.spines['top'].set_color('black')
ax3.spines['right'].set_color('black')
ax3.spines['left'].set_color('black')
ax3.spines['bottom'].set_color('black')

plt.show()