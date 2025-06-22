import numpy as np
import matplotlib.pyplot as plt
from math import log

# Paramètres
w0 = 14e-6  # Rayon du spot (m), 2w0 = 50 µm
F0 = 10  # Fluence de pic (J/cm^2)
Fth = 0.055  # Fluence seuil (J/cm^2)
delta = 18e-9  # Profondeur de pénétration (m), 0.112 µm
f_rep = 200e3  # Fréquence de répétition (Hz)
v = 1.0  # Vitesse de balayage (m/s)

# Espacement des impulsions
Delta = v / f_rep  # Espacement entre impulsions (m), 30 µm

# Rayon d'ablation r_th
r_th_squared = (w0**2 / 2) * log(F0 / Fth)  # r_th^2 (m^2)

# Dimensions de la zone simulée (m)
Lx = 150e-6  # Longueur en x (m), 100 µm
Ly = 150e-6  # Longueur en y (m), 100 µm

# Grille de points pour x et y
nx, ny = 200, 200  # Résolution de la grille
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)

# Génération des points de la spirale carrée
spiral_x, spiral_y = [Lx/2], [Ly/2]  # Commencer au centre
current_x, current_y = Lx/2, Ly/2
direction = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # Droite, Haut, Gauche, Bas
n = 10  # Nombre de tours

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
            current_x += dx * Delta
            current_y += dy * Delta
            spiral_x.append(current_x)
            spiral_y.append(current_y)

# Convertir en arrays numpy
spiral_x, spiral_y = np.array(spiral_x), np.array(spiral_y)

# Calcul de ln(F0/Fth)
ln_F0_Fth = log(F0 / Fth)

# Initialisation de la matrice Z(x, y)
Z = np.zeros((ny, nx))

# Simulation du balayage en spirale carrée
for xi, yj in zip(spiral_x, spiral_y):
    if 0 <= xi <= Lx and 0 <= yj <= Ly:
        r2 = (X - xi)**2 + (Y - yj)**2
        mask = r2 <= r_th_squared
        contribution = delta * (ln_F0_Fth - 2 * r2 / (w0**2))
        Z += np.where(mask, contribution, 0)

# Style professionnel
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 12,
    'axes.titlesize': 5,
    'axes.labelsize': 6,
    'legend.fontsize': 5,
    'xtick.labelsize': 5,
    'ytick.labelsize': 5,
    'figure.dpi': 250
})

# Création de la figure avec trois sous-graphiques
fig = plt.figure(figsize=(12, 8), constrained_layout=True)
gs = fig.add_gridspec(2, 2, width_ratios=[3, 2], height_ratios=[1, 1])

# Profil 3D à gauche
ax1 = fig.add_subplot(gs[:, 0], projection='3d')
surf = ax1.plot_surface(X * 1e6, Y * 1e6, -Z * 1e6, cmap='viridis', rstride=1, cstride=1,
                       linewidth=0.1, antialiased=True, shade=True)
ax1.set_xlabel('x (µm)')
ax1.set_ylabel('y (µm)')
ax1.set_zlabel('z (µm)')
ax1.xaxis.labelpad = -10  # Rapproche l'étiquette "X (µm)"
ax1.yaxis.labelpad = -10  # Rapproche l'étiquette "Y (µm)"
ax1.zaxis.labelpad = -7  # Rapproche l'étiquette "Profondeur Z (µm)"
ax1.xaxis.set_tick_params(pad=-5)  # Rapproche les chiffres de l'axe X
ax1.yaxis.set_tick_params(pad=-5)  # Rapproche les chiffres de l'axe Y
ax1.zaxis.set_tick_params(pad=-3)  # Rapproche les chiffres de l'axe Z


cbar = fig.colorbar(surf, ax=ax1, shrink=0.5, pad=0.15, aspect=20)
ax1.view_init(elev=30, azim=135)
ax1.xaxis.pane.set_edgecolor('black')
ax1.yaxis.pane.set_edgecolor('black')
ax1.zaxis.pane.set_edgecolor('black')
ax1.xaxis.pane.set_linewidth(0.5)
ax1.yaxis.pane.set_linewidth(0.5)
ax1.zaxis.pane.set_linewidth(0.5)
# Profil d'ablation suivant X (en haut à droite)
ax2 = fig.add_subplot(gs[0, 1])
y_mid = Ly / 2
y_idx = np.argmin(np.abs(y - y_mid))
ax2.plot(x * 1e6, -Z[y_idx, :] * 1e6, color='navy', linewidth=1, marker='o', markersize=0,
         markevery=20, )
ax2.set_xlabel('x (µm)')
ax2.set_ylabel('z (µm)')

ax2.legend()
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
# Profil d'ablation suivant Y (en bas à droite)
ax3 = fig.add_subplot(gs[1, 1])
x_mid = Lx / 2
x_idx = np.argmin(np.abs(x - x_mid))
ax3.plot(y * 1e6, -Z[:, x_idx] * 1e6, color='crimson', linewidth=1, marker='s', markersize=0,
         markevery=20, )
ax3.set_xlabel('y (µm)')


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

# Tableau pour la profondeur maximale
max_depth = -np.max(Z) * 1e6  # Profondeur maximale en µm
table_data = [['Profondeur maximale (µm)', f'{max_depth:.2f}']]
plt.figure(figsize=(5, 2))
table = plt.table(cellText=table_data, colLabels=['Paramètre', 'Valeur'],
                  loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)
plt.axis('off')
plt.title('Profondeur Maximale', fontsize=14, fontweight='bold')
plt.tight_layout()

plt.show()