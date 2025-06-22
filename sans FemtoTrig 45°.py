import numpy as np
import matplotlib.pyplot as plt
from math import ceil, log, cos, sin, radians

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
L_straight = 300  # Longueur de la découpe droite (µm)

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
Lx = 500  # Longueur en x (µm)
Ly = 500  # Longueur en y (µm)

# Grille de points pour x et y
nx, ny = 500, 500  # Résolution de la grille
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

# 1. Profil 3D
fig1 = plt.figure(figsize=(8, 6))
ax1 = fig1.add_subplot(111, projection='3d')
ax1.plot_surface(X, Y, Z, cmap='viridis')
ax1.set_xlabel('X (µm)')
ax1.set_ylabel('Y (µm)')
ax1.set_zlabel('Z (µm)')
ax1.set_title('Profil 3D (Découpe droite + virage 45° avec incubation)')
ax1.invert_zaxis()
plt.show()

# 2. Profil d'ablation suivant X (pour y=250 µm)
fig2 = plt.figure(figsize=(8, 6))
y_idx = np.argmin(np.abs(y - 250))
ax2 = fig2.add_subplot(111)
ax2.plot(x, Z[y_idx, :], label='Ablation le long de X (y=250 µm)')
ax2.set_xlabel('X (µm)')
ax2.set_ylabel('Z (µm)')
ax2.set_title('Profil d\'ablation suivant X')
ax2.grid(True)
ax2.legend()
plt.show()

# 3. Profil d'ablation suivant Y (pour x=50 µm)
fig3 = plt.figure(figsize=(8, 6))
x_idx = np.argmin(np.abs(x - 50))
ax3 = fig3.add_subplot(111)
ax3.plot(y, -Z[:, x_idx], label='Ablation le long de Y (x=50 µm)')
ax3.set_xlabel('Y (µm)')
ax3.set_ylabel('Z (µm)')
ax3.set_title('Profil d\'ablation suivant Y')
ax3.grid(True)
ax3.legend()
plt.show()

# 4. Graphique 2D (vue de dessus)
import matplotlib.colors as mcolors
colors = [(0.7, 0.7, 0.7)]  # Gris pour Z = 0
cmap_hot = plt.get_cmap('hot')
colors += [cmap_hot(i / 255) for i in range(255)]
custom_cmap = mcolors.ListedColormap(colors)
levels = [0] + list(np.linspace(0.0001, np.max(Z), 10))
fig4 = plt.figure(figsize=(8, 6))
ax4 = fig4.add_subplot(111)
contour = ax4.contourf(X, Y, Z, levels=levels, cmap=custom_cmap)
fig4.colorbar(contour, label='Profondeur (µm)')
ax4.set_xlabel('X (µm)')
ax4.set_ylabel('Y (µm)')
ax4.set_title('Profil d\'ablation 2D (Plan X-Y) - Ablé vs Non Ablé')
plt.show()