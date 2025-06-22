import numpy as np
import matplotlib.pyplot as plt
from math import ceil, log

# Paramètres donnés dans l'article
d = 28  # Diamètre du spot laser (µm)
w0 = d / 2  # Rayon du spot (µm)
F0 = 10  # Fluence de pic (J/cm^2)
Fth_1 = 0.055  # Fluence seuil initiale pour N=1 (J/cm^2)
delta_1 = 18e-3  # Profondeur de pénétration initiale pour N=1 (µm)
S = 0.8  # Paramètre d'incubation pour l'acier inoxydable

# Paramètres pour la vitesse de balayage
v = 1e6  # Vitesse de balayage (µm/s)
f = 200e3  # Fréquence de répétition (Hz)

# Calcul du taux de chevauchement des impulsions (phi_PO)
phi_PO = 1 - v / (2 * w0 * f)  # Formule donnée
if phi_PO < 0 or phi_PO > 1:
    raise ValueError(f"phi_PO = {phi_PO} est hors de la plage [0, 1]. Ajustez v ou f.")

# Calcul des distances
Delta_x = (1 - phi_PO) * d  # Distance entre impulsions consécutives (µm)

# Calcul de N_eff
N_eff = 2 * w0 / Delta_x  # Nombre effectif d'impulsions

# Fonction pour les paramètres variables avec effet d'incubation
def varying_params(N_eff):
    power = N_eff**(S - 1)
    return delta_1 * power, Fth_1 * power

# Dimensions de la zone simulée (µm)
Lx = 100  # Longueur en x (µm)
Ly = 100  # Longueur en y (µm) : de -50 à 50

# Grille de points pour x et y (réduite pour optimisation)
nx, ny = 100, 100  # Résolution de la grille (réduite de 1000 à 100)
x = np.linspace(-30, 130, nx)
y = np.linspace(-50, 50, ny)
X, Y = np.meshgrid(x, y)

# Nombre d'impulsions et de lignes
n_impulses = ceil(Lx / Delta_x)  # Nombre d'impulsions par ligne
n_lines = 1  # Une seule ligne à y=0

# Initialisation des matrices
Z = np.zeros((ny, nx))  # Profondeur d'ablation
N = np.zeros((ny, nx))  # Compteur d'impulsions par position

# Calcul de Z(x, y) pour le balayage à y=0 avec effets d'incubation
yj = 0  # Fixer y à 0
delta_N, Fth_N = varying_params(N_eff)  # Calculer une seule fois en dehors de la boucle
r_th_squared = (w0**2 / 2) * np.log(F0 / Fth_N)  # Rayon d'ablation
ln_F0_Fth_N = np.log(F0 / Fth_N)  # Précalculer le logarithme

for i in range(n_impulses):
    xi = i * Delta_x  # Position x de la i-ème impulsion
    r2 = (X - xi)**2 + (Y - yj)**2  # Distance au centre de l'impulsion
    N += (r2 <= (w0**2)).astype(int)  # Compter l'impulsion si dans le rayon du spot
    mask = (r2 <= r_th_squared) & (r_th_squared > 0)  # Condition d'ablation
    contribution = delta_N * (ln_F0_Fth_N - 2 * r2 / (w0**2))  # Contribution de l'impulsion
    Z += np.where(mask, contribution, 0)

# Calcul numérique du volume ablaté
dx = x[1] - x[0]  # Pas en x
dy = y[1] - y[0]  # Pas en y
V_numeric = np.abs(np.trapz(np.trapz(Z, y, axis=0), x, axis=0))  # Volume en µm³

# Style professionnel pour les graphiques
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

# Création de la figure principale avec subplots
fig = plt.figure(figsize=(12, 8), constrained_layout=True)
gs = fig.add_gridspec(2, 2, width_ratios=[3, 2], height_ratios=[1, 1])

# Graphique 3D à gauche
ax1 = fig.add_subplot(gs[:, 0], projection='3d')
surf = ax1.plot_surface(X, Y, -Z, cmap='viridis', rstride=1, cstride=1, 
                       linewidth=1, antialiased=True, shade=True)
ax1.set_xlabel('x (µm)')
ax1.set_ylabel('y (µm)')
ax1.set_zlabel('z (µm)')
ax1.xaxis.labelpad = -10  # Rapproche l'étiquette "X (µm)"
ax1.yaxis.labelpad = -10  # Rapproche l'étiquette "Y (µm)"
ax1.zaxis.labelpad = -7  # Rapproche l'étiquette "Profondeur Z (µm)"
ax1.xaxis.set_tick_params(pad=-5)  # Rapproche les chiffres de l'axe X
ax1.yaxis.set_tick_params(pad=-5)  # Rapproche les chiffres de l'axe Y
ax1.zaxis.set_tick_params(pad=-3)  # Rapproche les chiffres de l'axe Z
cbar = fig.colorbar(surf, ax=ax1, label='z (µm)', shrink=0.5, pad=0.20, aspect=20)
ax1.view_init(elev=30, azim=135)

ax1.xaxis.pane.set_edgecolor('black')
ax1.yaxis.pane.set_edgecolor('black')
ax1.zaxis.pane.set_edgecolor('black')
ax1.xaxis.pane.set_linewidth(0.5)
ax1.yaxis.pane.set_linewidth(0.5)
ax1.zaxis.pane.set_linewidth(0.5)
# Coupe 1D : z(x) à y = 05 µm (à droite, en haut)
ax2 = fig.add_subplot(gs[0, 1])
y_mid = 0  # Fixer à y=0
y_idx = np.argmin(np.abs(y - y_mid))
ax2.plot(x, -Z[y_idx, :], color='navy', linewidth=1, marker='o', markersize=0, 
         markevery=10,)
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
# Coupe 1D : z(y) à x = 50 µm (à droite, en bas)
ax3 = fig.add_subplot(gs[1, 1])
x_mid = Lx / 2
x_idx = np.argmin(np.abs(x - x_mid))
ax3.plot(y, -Z[:, x_idx], color='crimson', linewidth=1, marker='s', markersize=0, 
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


# Création du tableau séparé
table_data = [['Volume Ablaté (µm³)', f'{V_numeric:.2f}']]

plt.figure(figsize=(5, 2))
table = plt.table(cellText=table_data, colLabels=['Paramètre', 'Valeur'], 
                  loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)
plt.axis('off')
plt.title('Volume Ablaté', fontsize=14, fontweight='bold')
plt.tight_layout()

plt.show()