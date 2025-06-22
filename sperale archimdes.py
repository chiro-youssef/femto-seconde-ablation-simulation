import numpy as np
import matplotlib.pyplot as plt
from math import log

# Paramètres
w0 = 25e-6  # Rayon du spot (m), 2w0 = 50 µm
F0 = 20.37  # Fluence de pic (J/cm^2)
Fth = 13.33  # Fluence seuil (J/cm^2)
delta = 0.112e-6  # Profondeur de pénétration (m), 0.112 µm
f_rep = 800e3  # Fréquence de répétition (Hz)
v0 = 1.0  # Vitesse initiale au centre (m/s)
k = 0  # Augmentation de la vitesse (m/s par radian)

# Rayon d'ablation r_th
r_th_squared = (w0**2 / 2) * log(F0 / Fth)  # r_th^2 (m^2)

# Dimensions de la zone simulée (m)
Lx = 100e-6  # Longueur en x (m), 100 µm
Ly = 100e-6  # Longueur en y (m), 100 µm

# Grille de points pour x et y
nx, ny = 500, 500  # Résolution augmentée
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)

# Paramètres de la spirale d'Archimède
a = (10e-6) / (5 * np.pi)  # Espacement pour que r_max ≈ 50 µm après 5 tours
theta_max = 10 * np.pi  # 5 tours

# Générer des points pour la spirale d'Archimède (points denses pour approximation)
theta_dense = np.linspace(0, theta_max, 100000)
r_dense = a * theta_dense
x_dense = r_dense * np.cos(theta_dense) + Lx/2
y_dense = r_dense * np.sin(theta_dense) + Ly/2

# Sélectionner les points en fonction de l'espacement variable Delta(theta)
spiral_x, spiral_y = [x_dense[0]], [y_dense[0]]
thetas = [0.0]
last_x, last_y = x_dense[0], y_dense[0]

for i in range(1, len(theta_dense)):
    theta = theta_dense[i]
    v_theta = v0 + k * theta  # Vitesse variable (m/s)
    Delta = v_theta / f_rep  # Espacement variable (m)
    dist = np.sqrt((x_dense[i] - last_x)**2 + (y_dense[i] - last_y)**2)
    if dist >= Delta:
        spiral_x.append(x_dense[i])
        spiral_y.append(y_dense[i])
        thetas.append(theta)
        last_x, last_y = x_dense[i], y_dense[i]

# Convertir en arrays numpy
spiral_x, spiral_y = np.array(spiral_x), np.array(spiral_y)

# Calcul de ln(F0/Fth)
ln_F0_Fth = log(F0 / Fth)

# Initialisation de la matrice Z(x, y)
Z = np.zeros((ny, nx))

# Simulation du balayage en spirale d'Archimède
for xi, yj in zip(spiral_x, spiral_y):
    if 0 <= xi <= Lx and 0 <= yj <= Ly:
        r2 = (X - xi)**2 + (Y - yj)**2
        mask = r2 <= r_th_squared
        contribution = delta * (ln_F0_Fth - 2 * r2 / (w0**2))
        Z += np.where(mask, contribution, 0)

# Création des figures
fig = plt.figure(figsize=(15, 10))

# Profil 3D
ax1 = fig.add_subplot(221, projection='3d')
ax1.plot_surface(X * 1e6, Y * 1e6, Z * 1e6, cmap='viridis')  # Convertir en µm
ax1.set_xlabel('X (µm)')
ax1.set_ylabel('Y (µm)')
ax1.set_zlabel('Z (µm)')
ax1.set_title('3D Profile (Archimedean Spiral)')
ax1.invert_zaxis()

# Profil d'ablation suivant X
fig2 = plt.figure(figsize=(15, 10))
ax2 = fig2.add_subplot(222)
y_mid = Ly / 2
y_idx = np.argmin(np.abs(y - y_mid))
ax2.plot(x * 1e6, -Z[y_idx, :] * 1e6, label=f'Ablation along X (y={(y[y_idx] * 1e6):.2f} µm)')
ax2.set_xlabel('X (µm)')
ax2.set_ylabel('Z (µm)')
ax2.set_title('Ablation Following X')
ax2.grid(True)
ax2.legend()

# Profil d'ablation suivant Y
fig3 = plt.figure(figsize=(15, 10))
ax3 = fig3.add_subplot(223)
x_mid = Lx / 2
x_idx = np.argmin(np.abs(x - x_mid))
ax3.plot(y * 1e6, -Z[:, x_idx] * 1e6, label=f'Ablation along Y (x={(x[x_idx] * 1e6):.2f} µm)')
ax3.set_xlabel('Y (µm)')
ax3.set_ylabel('Z (µm)')
ax3.set_title('Ablation Following Y')
ax3.grid(True)
ax3.legend()

# Visualisation du trajet de la spirale
ax4 = fig.add_subplot(224)
ax4.plot(spiral_x * 1e6, spiral_y * 1e6, 'o-', markersize=3)
ax4.set_xlabel('X (µm)')
ax4.set_ylabel('Y (µm)')
ax4.set_title('Trajet de la spirale d\'Archimède')
ax4.axis('equal')
ax4.grid(True)

# Ajustement de la mise en page
plt.tight_layout()

# Affichage du graphique
plt.show()