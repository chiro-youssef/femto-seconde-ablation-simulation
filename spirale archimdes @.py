import numpy as np
import matplotlib.pyplot as plt
from math import log

# Paramètres du modèle
d = 28  # Diamètre du spot laser (µm)
w0 = d / 2  # Rayon du spot (µm)
F0 = 10  # Fluence de pic (J/cm^2)
Fth_1 = 0.055  # Fluence seuil initiale pour N=1 (J/cm^2)
delta_1 = 18e-3  # Profondeur de pénétration initiale pour N=1 (µm)
S = 0.8  # Paramètre d'incubation pour l'acier inoxydable
v = 1e6  # Vitesse de balayage (µm/s)
f = 200e3  # Fréquence de répétition (Hz)

# Conversion des unités
w0_m = w0 * 1e-6  # Rayon du spot en mètres
v_m = v * 1e-6  # Vitesse en m/s
spot_area = np.pi * w0_m**2  # Surface du spot (m²)
E_p = F0 * 1e4 * spot_area  # Énergie par impulsion (J)

# Dimensions de la zone simulée (µm)
Lx = 160  # Longueur en x (de -80 à 80 µm)
Ly = 160  # Longueur en y (de -80 à 80 µm)

# Grille de points
nx, ny = 100, 100  # Résolution
x = np.linspace(-100, 100, nx)
y = np.linspace(-100, 100, ny)
X, Y = np.meshgrid(x, y)
X_m = X * 1e-6  # Grille x en mètres
Y_m = Y * 1e-6  # Grille y en mètres

# Génération des points de la spirale d'Archimède (inward)
a = (50e-6) / (20 * np.pi)  # Espacement pour r_max ≈ 40 µm après 5 tours
theta_max = 25 * np.pi  # 5 tours
theta_dense = np.linspace(theta_max, 0, 1000000)  # De theta_max à 0 (inward)
r_dense = a * theta_dense  # Rayon décroît de r_max à 0
x_dense = r_dense * np.cos(theta_dense)  # Centré à (0, 0)
y_dense = r_dense * np.sin(theta_dense)  # Centré à (0, 0)

# Sélectionner les points en fonction de l'espacement Delta
spiral_x, spiral_y, times = [x_dense[0] * 1e6], [y_dense[0] * 1e6], [0.0]  # Convertir en µm
last_x, last_y = x_dense[0], y_dense[0]
impulse_idx = 0
delta_t = 1 / f  # Intervalle entre impulsions (s)
Delta = v_m / f  # Espacement entre impulsions (m)

for i in range(1, len(theta_dense)):
    dist = np.sqrt((x_dense[i] - last_x)**2 + (y_dense[i] - last_y)**2)
    if dist >= Delta:
        spiral_x.append(x_dense[i] * 1e6)  # Convertir en µm
        spiral_y.append(y_dense[i] * 1e6)  # Convertir en µm
        times.append(impulse_idx * delta_t)
        last_x, last_y = x_dense[i], y_dense[i]
        impulse_idx += 1

spiral_x, spiral_y, times = np.array(spiral_x), np.array(spiral_y), np.array(times)
N_pulses = len(spiral_x)  # Nombre total d'impulsions

# Calcul de N_eff
N_eff = 2 * w0 / (Delta * 1e6)  # Nombre effectif d'impulsions (Delta converti en µm)

# Fonction pour les paramètres variables avec effet d'incubation
def varying_params(N_eff):
    power = N_eff**(S - 1)
    return delta_1 * power, Fth_1 * power

# Initialisation de la matrice Z(x, y)
Z = np.zeros((ny, nx))  # Profondeur d'ablation (µm)
N = np.zeros((ny, nx))  # Compteur d'impulsions

# Calcul de l'ablation en spirale d'Archimède
for xi, yj in zip(spiral_x, spiral_y):
    if -80 <= xi <= 80 and -80 <= yj <= 80:
        r2 = (X - xi)**2 + (Y - yj)**2
        N += (r2 <= (w0**2)).astype(int)
        delta_N, Fth_N = varying_params(N_eff)
        r_th_squared = (w0**2 / 2) * np.log(F0 / Fth_N)
        mask = (r2 <= r_th_squared) & (r_th_squared > 0)
        ln_F0_Fth_N = np.log(F0 / Fth_N)
        contribution = delta_N * (ln_F0_Fth_N - 2 * r2 / (w0**2))
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

# Profil 3D à gauche (carte de chaleur)
ax1 = fig.add_subplot(gs[:, 0], projection='3d')
surf = ax1.plot_surface(X, Y, -Z, cmap='viridis', rstride=1, cstride=1,
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

ax1.view_init(elev=30, azim=135)
ax1.xaxis.pane.set_edgecolor('black')
ax1.yaxis.pane.set_edgecolor('black')
ax1.zaxis.pane.set_edgecolor('black')
ax1.xaxis.pane.set_linewidth(0.5)
ax1.yaxis.pane.set_linewidth(0.5)
ax1.zaxis.pane.set_linewidth(0.5)
# Profil d'ablation suivant X (en haut à droite)
ax2 = fig.add_subplot(gs[0, 1])
y_mid = 0  # Centré en y = 0
y_idx = np.argmin(np.abs(y - y_mid))
ax2.plot(x, -Z[y_idx, :], color='navy', linewidth=1, marker='o', markersize=0,
         markevery=10, )
ax2.set_xlabel('x (µm)')
ax2.set_ylabel('z (µm)')

ax2.legend()
ax2.grid(True, alpha=0.3)
 # Rend le graphique carré
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
x_mid = 0  # Centré en x = 0
x_idx = np.argmin(np.abs(x - x_mid))
ax3.plot(y, -Z[:, x_idx], color='crimson', linewidth=1, marker='s', markersize=0,
         markevery=10, )
ax3.set_xlabel('x (µm)')
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

# Tableau pour la profondeur maximale
max_depth = -np.max(Z)  # Profondeur maximale en µm
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