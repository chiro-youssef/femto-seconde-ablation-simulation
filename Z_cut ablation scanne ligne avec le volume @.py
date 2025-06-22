import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Paramètres physiques (unités SI)
delta0   = 18e-9         # Profondeur d'ablation pour une impulsion, δ(1) = 18 nm
Fth0     = 550           # Fluence seuil pour une impulsion, F_th(1) = 550 J/m² (0.055 J/cm²)
omega_0  = 14e-6         # Rayon du faisceau laser (m)
d        = 2 * omega_0   # Diamètre du spot laser (m)
Fm       = 5e4           # Fluence moyenne, J/m² (4.06 J/cm²)
Fc       = 2 * Fm        # Fluence de pic (J/m²)
S        = 0.8           # Exposant d'incubation
Nmax     = 25            # Limite sur le nombre effectif d'impulsions (N_eff)
v        = 1             # Vitesse de balayage (m/s)
f        = 200e3         # Fréquence de répétition (Hz)
delta_x  = v / f         # Distance entre impulsions (m)

# Calcul du taux de chevauchement en x
R_x = 1 - delta_x / d    # Taux de chevauchement en x
print(f"Taux de chevauchement des impulsions en x (R_x) : {R_x*100:.2f}%")

# Grille pour x (direction de balayage) et y (transverse)
x = np.linspace(0, 100e-6, 100)  # m, de 0 à 100 µm
y = np.linspace(-60e-6, 60e-6, 100)  # m, de -60 µm à 60 µm
X, Y = np.meshgrid(x, y)  # Grille 2D

# Fonctions principales
def ablation_depth(y, Fc_loc, delta_loc, Fth_loc, delta_x_loc):
    """
    Calcule la profondeur d'ablation par impulsion selon la formule :
      L = ln(Fc_loc / Fth_loc)
      z(y) = (Δx·δ·√2)/(3·ω₀) · √([L − 2y²/ω₀²]) · [ (2ω₀²/Δx²)·(L − 2y²/ω₀²) − 1 ]
    """
    L = np.log(Fc_loc / Fth_loc)
    arg = L - 2 * (y**2) / omega_0**2
    positive = arg > 0
    fac = (delta_x_loc * delta_loc * np.sqrt(2)) / (3 * omega_0)
    sqrt_term = np.sqrt(np.where(positive, arg, 0.0))
    bracket = (2 * omega_0**2 / delta_x_loc**2) * arg - 1
    z = fac * sqrt_term * bracket
    return np.where(positive, z, 0.0)

def varying_params(N_eff):
    """
    Ajuste les paramètres delta0 et Fth0 en fonction de N_eff avec l'effet d'incubation.
    """
    power = N_eff**(S - 1)
    return delta0 * power, Fth0 * power

def compute_y_fixed_profile(X, Y, delta_x_loc, Nmax=None, fixed_y=0.0):
    """
    Calcule le profil d'ablation pour une position y fixe.
    Retourne la profondeur Z et le nombre total d'impulsions (1 pour y fixe).
    """
    Z = np.zeros_like(X)
    N_eff = 2 * omega_0 / delta_x_loc  # Calcul de N_eff
    if Nmax is not None:
        N_eff = min(N_eff, Nmax)
    delta_N, Fth_N = varying_params(N_eff)
    y_relative = Y - fixed_y
    Z = ablation_depth(y_relative, Fc, delta_N, Fth_N, delta_x_loc)
    pulse_count = 1
    print(f"Nombre total d'impulsions (y fixe) : {pulse_count}")
    print(f"Profondeur max (µm) : {np.max(np.abs(Z))*1e6:.2f}")
    return Z, pulse_count

# Calcul de Z et pulse_count pour la trajectoire à y fixé
fixed_y = 0.0  # Fixer y à 0 µm
Z, pulse_count = compute_y_fixed_profile(X, Y, delta_x, Nmax=Nmax, fixed_y=fixed_y)
Z_m = Z  # Garder Z en mètres pour le calcul du volume
Z = -Z * 1e6  # Convertir en µm pour l'affichage (profondeur positive)

# Calcul numérique du volume ablaté
volume = np.abs(np.trapezoid(np.trapezoid(Z_m, y, axis=0), x))  # Intégration double
volume_um3 = volume * 1e18  # Convertir de m³ à µm³
print(f"Volume ablaté (µm³) : {volume_um3:.2f}")

# Création des données du tableau
table_data = [['Volume Ablaté (µm³)', f'{volume_um3:.2f}']]

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
surf = ax1.plot_surface(X * 1e6, Y * 1e6, Z, cmap='viridis', rstride=1, cstride=1, 
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
ax1.xaxis.pane.set_edgecolor('black')
ax1.yaxis.pane.set_edgecolor('black')
ax1.zaxis.pane.set_edgecolor('black')
ax1.xaxis.pane.set_linewidth(0.5)
ax1.yaxis.pane.set_linewidth(0.5)
ax1.zaxis.pane.set_linewidth(0.5)
# Coupe 1D : z(x) à y = 0 µm (à droite, en haut)
ax2 = fig.add_subplot(gs[0, 1])
x_slice = x * 1e6
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
y_slice = y * 1e6
x_index = len(x) // 2
z_slice = Z[:, x_index]
ax3.plot(y_slice, z_slice, color='crimson', linewidth=1, marker='s', markersize=0, 
         markevery=10,)
ax3.set_xlabel('y (µm)')
ax3.set_ylabel('Z (µm)')

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
plt.title('Volume Ablaté', fontsize=14, fontweight='bold')
plt.tight_layout()

plt.show()