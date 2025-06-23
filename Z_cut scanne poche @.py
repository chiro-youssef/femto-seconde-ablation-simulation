import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import quad

# Paramètres physiques (µm, J/cm², s)
delta0   = 0.018         # Profondeur d'ablation pour une impulsion, δ(1) = 18 nm = 0.018 µm
Fth0     = 0.055         # Fluence seuil pour une impulsion, F_th(1) = 550 J/m² = 0.055 J/cm²
omega_0  = 14            # Rayon du faisceau (µm)
delta_x  = 5             # Espacement entre impulsions fixé à 5 µm
d        = 2 * omega_0   # Diamètre du spot laser (µm)
Fm       = 5          # Fluence moyenne, 4.06 J/cm²
Fc       = 2 * Fm        # Fluence de pic (J/cm²) = 8.12 J/cm²
S        = 0.8           # Exposant d'incubation
Nmax     = 25            # Limite sur le nombre effectif d'impulsions (N_eff)
v        = 1e6           # Vitesse de balayage (µm/s) = 1 m/s
f        = 200e3         # Fréquence de répétition (Hz)
delta_y  = v / f         # Espacement en y (µm) = 5 µm

# Calcul des taux de chevauchement
R_x = 1 - delta_x / d    # Taux de chevauchement en x
R_y = 1 - delta_y / d    # Taux de chevauchement en y
print(f"Taux de chevauchement des impulsions en x (R_x) : {R_x*100:.2f}%")
print(f"Taux de chevauchement des impulsions en y (R_y) : {R_y*100:.2f}%")

# Grille pour x (direction de balayage) et y (transverse)
x = np.linspace(0, 100, 100)      # µm, de 0 à 100 µm
y = np.linspace(-150, 150, 100)   # µm, de -150 µm à 150 µm
X, Y = np.meshgrid(x, y)          # Grille 2D

# Fonctions principales
def ablation_depth(y, Fc_loc, delta_loc, Fth_loc, delta_x_loc):
    """
    Calcule la profondeur d'ablation par impulsion (µm) selon la formule :
      L = ln(Fc_loc / Fth_loc)
      z(y) = (Δx·δ·√2)/(3·ω₀) · √([L − 2y²/ω₀²]) · [ (2ω₀²/Δx²)·(L − 2y²/ω₀²) − 1 ]
    """
    L = np.log(Fc_loc / Fth_loc)
    arg = L - 2 * (y**2) / omega_0**2
    positive = arg > 0
    fac = (delta_x_loc * delta_loc * np.sqrt(2)) / (3 * omega_0)
    sqrt_term = np.sqrt(np.where(positive, arg, 0.0))
    bracket = (2 * omega_0**2 / delta_x_loc**2) * arg - 1
    z = fac* sqrt_term * bracket
    return np.where(positive, z, 0.0)

def varying_params(N_eff):
    """
    Ajuste les paramètres delta0 et Fth0 en fonction de N_eff avec l'effet d'incubation.
    """
    power = N_eff**(S - 1)
    return delta0 * power, Fth0 * power

def compute_y_scan_profile(X, Y, delta_x_loc, delta_y_loc, Nmax=None):
    """
    Calcule le profil d'ablation pour un balayage le long de y avec compteur.
    """
    Z = np.zeros_like(X)
    y_positions = np.arange(-100.5, 100.5 + delta_y_loc, delta_y_loc)
    N_eff = 2 * omega_0 / delta_y_loc
    if Nmax is not None:
        N_eff = min(N_eff, Nmax)
    delta_N, Fth_N = varying_params(N_eff)
    pulse_count = 0
    for y_i in y_positions:
        pulse_count += 1
        y_relative = Y - y_i
        Z += ablation_depth(y_relative, Fc, delta_N, Fth_N, delta_x_loc)
    print(f"Nombre total d'impulsions : {pulse_count}")
    return Z, pulse_count

def calculate_A_cross(delta_x_loc, delta_loc, omega_0_loc, Fc_loc, Fth_loc, y_max_loc):
    """
    Calcule la section transversale A_cross en intégrant la profondeur d'ablation le long de y.
    """
    def z_y(y):
        return ablation_depth(y, Fc_loc, delta_loc, Fth_loc, delta_x_loc)  # Déjà en µm
    integral, _ = quad(z_y, -y_max_loc, y_max_loc)  # Intégration en µm²
    return integral

# Calcul de Z pour la trajectoire de balayage en y
Z, pulse_count = compute_y_scan_profile(X, Y, delta_x, delta_y, Nmax=Nmax)  # Z en µm
Z = -Z  # Profondeur positive pour l'affichage

# Calcul du taux d'ablation volumique (un passage)
N_eff = min(2 * omega_0 / delta_y, Nmax)  # Calcul de N_eff pour le balayage
delta_N, Fth_N = varying_params(N_eff)
y_max = (omega_0 / np.sqrt(2)) * np.sqrt(np.log(Fc / Fth_N))  # Limite d'intégration dynamique
A_cross = calculate_A_cross(delta_x, delta_N, omega_0, Fc, Fth_N, y_max)  # Section transversale en µm²
V_dot_zcut = v * A_cross  # Taux d'ablation volumique en µm³/s
V_dot_zcut_mm3 = V_dot_zcut * 1e-9  # Conversion en mm³/s
print(f"Section transversale (A_cross) : {A_cross:.5f} µm²")
print(f"Taux d'ablation volumique (Zcut) : {V_dot_zcut_mm3:.5f} mm³/s")

# Création des données du tableau
table_data = [['Taux d\'ablation volumique (mm³/s)', f'{V_dot_zcut_mm3:.5f}']]

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
surf = ax1.plot_surface(X, Y, Z, cmap='viridis', rstride=1, cstride=1, 
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
cbar = fig.colorbar(surf, ax=ax1, label='z (µm)', shrink=0.5, pad=0.20, aspect=20)
ax1.view_init(elev=30, azim=135)
ax1.view_init(elev=30, azim=135)
ax1.xaxis.pane.set_edgecolor('black')
ax1.yaxis.pane.set_edgecolor('black')
ax1.zaxis.pane.set_edgecolor('black')
ax1.xaxis.pane.set_linewidth(0.5)
ax1.yaxis.pane.set_linewidth(0.5)
ax1.zaxis.pane.set_linewidth(0.5)
# Coupe 1D : z(x) à y = 0 µm (à droite, en haut)
ax2 = fig.add_subplot(gs[0, 1])
x_slice = x
y_index = len(y) // 2  # Correspond à y = 0 µm
z_slice = Z[y_index, :]
ax2.plot(x_slice, z_slice, color='navy', linewidth=1, marker='o', markersize=0, 
         markevery=10, )
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
y_slice = y
x_index = len(x) // 2  # Correspond à x = 50 µm
z_slice = Z[:, x_index]
ax3.plot(y_slice, z_slice, color='crimson', linewidth=1, marker='s', markersize=0, 
         markevery=10, )
ax3.set_xlabel('y (µm)')
ax3.set_ylabel('z (µm)')

ax3.legend()
ax3.grid(True, alpha=0.3)
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
plt.figure(figsize=(5, 2))
table = plt.table(cellText=table_data, colLabels=['Paramètre', 'Valeur'], 
                  loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)
plt.axis('off')
plt.title('Taux d\'Ablation Volumique', fontsize=14, fontweight='bold')
plt.tight_layout()

plt.show()
