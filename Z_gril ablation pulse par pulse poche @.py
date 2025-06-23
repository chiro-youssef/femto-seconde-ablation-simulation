import numpy as np
import matplotlib.pyplot as plt
from math import ceil, log

# Paramètres donnés dans l'article
d = 28  # Diamètre du spot laser (µm)
w0 = d / 2  # Rayon du spot (µm)
phi_LO = 0.82  # Taux de chevauchement des lignes
F0 = 10  # Fluence de pic (J/cm^2)
Fth_1 = 0.055  # Fluence seuil initiale pour N=1 (J/cm^2)
delta_1 = 18e-3  # Profondeur de pénétration initiale pour N=1 (µm)
S = 0.8  # Paramètre d'incubation pour l'acier inoxydable

# Paramètres pour la vitesse de balayage
v = 1000000  # Vitesse de balayage (µm/s)
f = 200e3  # Fréquence de répétition (Hz)

# Calcul du taux de chevauchement des impulsions (phi_PO)
phi_PO = 1 - v / (2 * w0 * f)  # Formule donnée
if phi_PO < 0 or phi_PO > 1:
    raise ValueError(f"phi_PO = {phi_PO} est hors de la plage [0, 1]. Ajustez v ou f.")

# Calcul des distances
Delta_x = (1 - phi_PO) * d  # Distance entre impulsions consécutives (µm)
dH = (1 - phi_LO) * d  # Espacement entre les lignes (µm)

# Calcul de N_eff
N_eff = 2 * w0 / Delta_x  # Nombre effectif d'impulsions

# Fonction pour les paramètres variables avec effet d'incubation
def varying_params(N_eff):
    """
    Ajuste les paramètres delta_1 et Fth_1 en fonction de N_eff avec l'effet d'incubation.
    """
    power = N_eff**(S - 1)
    return delta_1 * power, Fth_1 * power

# Dimensions de la zone simulée (µm)
Lx = 200  # Longueur en x (µm)
Ly = 200  # Longueur en y (µm)

# Grille de points pour x et y (réduite pour optimisation)
nx, ny = 100, 100  # Résolution de la grille
x = np.linspace(-20, Lx+20, nx)
y = np.linspace(-20, Ly+20, ny)
X, Y = np.meshgrid(x, y)

# Nombre d'impulsions et de lignes
n_impulses = ceil(Lx / Delta_x)  # Nombre d'impulsions par ligne
n_lines = ceil(Ly / dH)  # Nombre de lignes

# Initialisation des matrices
Z = np.zeros((ny, nx))  # Profondeur d'ablation
N = np.zeros((ny, nx))  # Compteur d'impulsions par position

# Précalcul des paramètres invariants
delta_N, Fth_N = varying_params(N_eff)
r_th_squared = (w0**2 / 2) * np.log(F0 / Fth_N)
ln_F0_Fth_N = np.log(F0 / Fth_N)

# Calcul de Z(x, y) pour le balayage parallèle avec effets d'incubation
for j in range(n_lines):
    yj = j * dH  # Position y de la j-ème ligne
    for i in range(n_impulses):
        xi = i * Delta_x  # Position x de la i-ème impulsion
        r2 = (X - xi)**2 + (Y - yj)**2  # Distance au centre de l'impulsion
        N += (r2 <= (w0**2)).astype(int)  # Compter l'impulsion si dans le rayon du spot
        mask = (r2 <= r_th_squared) & (r_th_squared > 0)  # Condition d'ablation
        contribution = delta_N * (ln_F0_Fth_N - 2 * r2 / (w0**2))  # Contribution de l'impulsion
        Z += np.where(mask, contribution, 0)

# Fonction pour calculer le volume par impulsion
def calculate_V_pulse(delta_N, omega_0, Fc_0, Fth_N):
    """
    Calcule le volume ablaté par impulsion.
    """
    ln_ratio = np.log(Fc_0 / Fth_N)
    V_pulse = (np.pi / 4) * delta_N * (omega_0**2) * (ln_ratio**2)
    return V_pulse  # en µm³

# Calcul du volume ablaté pour Z gril
N_total = n_impulses * n_lines  # Nombre total d'impulsions
V_pulse = calculate_V_pulse(delta_N, w0, F0, Fth_N)  # Volume par impulsion
V_pulse_eff = V_pulse * N_total  # Volume effectif
V_dot_zgril = f * V_pulse_eff  # Taux d'ablation volumique (µm³/s)
V_dot_zgril_mm3 = V_dot_zgril * 1e-9  # Conversion en mm³/s

# Création des données du tableau
table_data = [['Taux d\'ablation volumique (mm³/s)', f'{V_dot_zgril_mm3:.5f}']]

# Style professionnel pour les graphiques
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 12,
    'axes.titlesize': 10,
    'axes.labelsize': 10,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 250
})

# Graphique 1 : Surface 3D
fig1 = plt.figure(figsize=(8, 6), constrained_layout=True)
ax1 = fig1.add_subplot(projection='3d')
surf = ax1.plot_surface(X, Y, -Z, cmap='viridis', rstride=1, cstride=1, 
                        linewidth=0.1, antialiased=True, shade=True)
ax1.set_xlabel('x (µm)')
ax1.set_ylabel('y (µm)')
ax1.set_zlabel('z (µm)')
ax1.xaxis.labelpad = -10
ax1.yaxis.labelpad = -10
ax1.zaxis.labelpad = -7
ax1.xaxis.set_tick_params(pad=-5)
ax1.yaxis.set_tick_params(pad=-5)
ax1.zaxis.set_tick_params(pad=-3)
cbar = fig1.colorbar(surf, ax=ax1, label='z (µm)', shrink=0.5, pad=0.20, aspect=20)
ax1.view_init(elev=30, azim=135)
ax1.xaxis.pane.set_edgecolor('black')
ax1.yaxis.pane.set_edgecolor('black')
ax1.zaxis.pane.set_edgecolor('black')
ax1.xaxis.pane.set_linewidth(0.5)
ax1.yaxis.pane.set_linewidth(0.5)
ax1.zaxis.pane.set_linewidth(0.5)

# Graphique 2 : Coupe z(x) à y = Ly/2 µm
fig2 = plt.figure(figsize=(6, 4), constrained_layout=True)
ax2 = fig2.add_subplot()
y_mid = Ly / 2
y_idx = np.argmin(np.abs(y - y_mid))
ax2.plot(x, -Z[y_idx, :], color='navy', linewidth=1, marker='o', markersize=0, 
         markevery=10)
ax2.set_xlabel('x (µm)')
ax2.set_ylabel('z (µm)')
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

# Graphique 3 : Coupe z(y) à x = Lx/2 µm
fig3 = plt.figure(figsize=(6, 4), constrained_layout=True)
ax3 = fig3.add_subplot()
x_mid = Lx / 2
x_idx = np.argmin(np.abs(x - x_mid))
ax3.plot(y, -Z[:, x_idx], color='crimson', linewidth=2, marker='s', markersize=0, 
         markevery=10)
ax3.set_xlabel('y (µm)')
ax3.set_ylabel('z (µm)')
ax3.grid(True, alpha=0.3)
ax3.spines['top'].set_visible(True)
ax3.spines['right'].set_visible(True)
ax3.spines['left'].set_visible(True)
ax3.spines['bottom'].set_visible(True)
ax3.spines['top'].set_linewidth(1)
ax3.spines['right'].set_linewidth(1)
ax3.spines['left'].set_linewidth(1)
ax3.spines['bottom'].set_linewidth(1)
ax3.spines['top'].set_color('black')
ax3.spines['right'].set_color('black')
ax3.spines['left'].set_color('black')
ax3.spines['bottom'].set_color('black')

# Tableau séparé
fig4 = plt.figure(figsize=(5, 2))
table = plt.table(cellText=table_data, colLabels=['Paramètre', 'Valeur'], 
                  loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)
plt.axis('off')
plt.title('Taux d\'Ablation Volumique', fontsize=14, fontweight='bold')
plt.tight_layout()

plt.show()
