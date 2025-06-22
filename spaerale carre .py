import numpy as np
import matplotlib.pyplot as plt
from math import ceil, log

# Paramètres
w0 = 25e-6  # Rayon du spot (m), 2w0 = 50 µm
F0 = 10 # Fluence de pic (J/cm^2)
Fth = 0.055  # Fluence seuil (J/cm^2)
delta = 18e-9  # Profondeur de pénétration (m), 0.112 µm
f_rep = 200e3  # Fréquence de répétition (Hz)
v = 2.0  # Vitesse de balayage (m/s)

# Espacement des impulsions
Delta = v / f_rep  # Espacement entre impulsions (m), 30 µm

# Rayon d'ablation r_th
r_th_squared = (w0**2 / 2) * log(F0 / Fth)  # r_th^2 (m^2)

# Dimensions de la zone simulée (m)
Lx = 200e-6  # Longueur en x (m), 100 µm
Ly = 200e-6  # Longueur en y (m), 100 µm

# Grille de points pour x et y
nx, ny = 200, 200  # Résolution de la grille
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)

# Génération des points de la spirale carrée
spiral_x, spiral_y = [Lx/2], [Ly/2]  # Commencer au centre
current_x, current_y = Lx/2, Ly/2
direction = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # Droite, Haut, Gauche, Bas

# Nombre de tours (ajusté pour couvrir la zone)
n =5  # Ajuster selon la taille de la zone

for k in range(n):
    for i in range(4):  # 4 segments par tour
        dir_idx = i % 4
        dx, dy = direction[dir_idx]
        # Déterminer la longueur du segment en fonction du tour et de la direction
        if i in [0, 1]:  # Droite ou Haut
            steps = 1 + 2 * k
        else:  # Gauche ou Bas
            steps = 2 + 2 * k
        length = steps * Delta  # Longueur du segment (m)
        # Placer des impulsions le long du segment
        num_points = max(2, int(length / Delta))  # Nombre de points (au moins 2)
        for t in np.linspace(0, length, num_points, endpoint=False):
            current_x += dx * Delta
            current_y += dy * Delta
            spiral_x.append(current_x)
            spiral_y.append(current_y)

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

# Création des figures
fig = plt.figure(figsize=(15, 5))

# Profil 3D
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(X * 1e6, Y * 1e6, Z * 1e6, cmap='viridis')  # Convertir en µm
ax1.set_xlabel('X (µm)')
ax1.set_ylabel('Y (µm)')
ax1.set_zlabel('Z (µm)')
ax1.set_title('3D Profile (Square Spiral)')
ax1.invert_zaxis()

# Profil d'ablation suivant X
fig1 = plt.figure(figsize=(15, 5))
ax2 = fig1.add_subplot(132)
y_mid = Ly / 2
y_idx = np.argmin(np.abs(y - y_mid))
ax2.plot(x * 1e6, -Z[y_idx, :] * 1e6,)
ax2.set_xlabel('X (µm)')
ax2.set_ylabel('Z (µm)')
ax2.set_title('Ablation Following X')
ax2.grid(True)
ax2.legend()

# Profil d'ablation suivant Y
fig2 = plt.figure(figsize=(15, 5))
ax3 = fig2.add_subplot(133)
x_mid = Lx / 2
x_idx = np.argmin(np.abs(x - x_mid))
ax3.plot(y * 1e6, -Z[:, x_idx] * 1e6,)
ax3.set_xlabel('Y (µm)')
ax3.set_ylabel('Z (µm)')
ax3.set_title('Ablation Following Y')
ax3.grid(True)
ax3.legend()

# Ajustement de la mise en page
plt.tight_layout()

# Affichage du graphique
plt.show()