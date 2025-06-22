import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import ceil

# Paramètres physiques (SI)
delta0   = 18e-9         # δ(1) = 18 nm
Fth0     = 550           # F_th(1) = 550 J/m²
omega_0  = 14e-6         # rayon du faisceau (m)
d        = 2 * omega_0   # diamètre du spot (m)
Fm       = 5e4           # J/m²
Fc       = 2 * Fm        # fluence de pic (J/m²)
S        = 0.8           # exposant d’incubation
Nmax     = 25            # limite sur N_eff
v        = 1.5           # vitesse de balayage (m/s)
f        = 400e3         # fréquence de répétition (Hz)
delta_x  = v/f           # delta_x initial (m)
change_point = 50        # point où delta_x change (index dans x)
delta_x_new = delta_x / 0.77  # nouvelle valeur de delta_x

# Calculer le taux de chevauchement en x
R_x = 1 - delta_x / d
R_x_new = 1 - delta_x_new / d
print(f"Taux de chevauchement initial en x (R_x): {R_x*100:.2f}%")
print(f"Taux de chevauchement après changement en x (R_x_new): {R_x_new*100:.2f}%")

# Grille pour x et y
x = np.linspace(0, 100e-6, 100)  # m, de 0 à 100 µm
y = np.linspace(-60e-6, 60e-6, 100)  # m, de -60 µm à 60 µm
X, Y = np.meshgrid(x, y)

# Créer un tableau delta_x variant
delta_x_loc = np.full_like(x, delta_x)
delta_x_loc[change_point:] = delta_x_new

# Fonctions principales
def ablation_depth(y, Fc_loc, delta_loc, Fth_loc, delta_x_loc):
    L = np.log(Fc_loc / Fth_loc)
    arg = L - 2 * (y**2) / omega_0**2
    positive = arg > 0
    fac = (delta_x_loc * delta_loc * np.sqrt(2)) / (3 * omega_0)
    sqrt_term = np.sqrt(np.where(positive, arg, 0.0))
    bracket = (2 * omega_0**2 / delta_x_loc**2) * arg - 1
    z = fac * sqrt_term * bracket
    return np.where(positive, z, 0.0)

def varying_params(N_eff):
    power = N_eff**(S - 1)
    return delta0 * power, Fth0 * power

def compute_y_fixed_profile(X, Y, delta_x_loc, Nmax=None, fixed_y=0.0):
    Z = np.zeros_like(X)
    N_eff = 2 * omega_0 / delta_x_loc
    if Nmax is not None:
        N_eff = np.minimum(N_eff, Nmax)
    
    x_change = x[change_point]
    L1 = x_change
    L2 = x[-1] - x_change
    N1 = ceil(L1 / delta_x)
    N2 = ceil(L2 / delta_x_new)
    pulse_count = N1 + N2
    
    y_relative = Y - fixed_y
    for i in range(X.shape[1]):
        delta_N, Fth_N = varying_params(N_eff[i])
        Z[:, i] = ablation_depth(y_relative[:, i], Fc, delta_N, Fth_N, delta_x_loc[i])
    
    print(f"Nombre d'impulsions première partie (0 à {x_change*1e6:.1f} µm) : {N1}")
    print(f"Nombre d'impulsions deuxième partie ({x_change*1e6:.1f} à {x[-1]*1e6:.1f} µm) : {N2}")
    print(f"Nombre total d'impulsions : {pulse_count}")
    print(f"Profondeur max globale (µm) : {np.max(np.abs(Z))*1e6:.2f}")
    print(f"Profondeur max avant x = {x_change*1e6:.1f} µm : {np.max(np.abs(Z[:, :change_point]))*1e6:.2f} µm")
    print(f"Profondeur max après x = {x_change*1e6:.1f} µm : {np.max(np.abs(Z[:, change_point:]))*1e6:.2f} µm")

    return Z, pulse_count

# Calculer Z et pulse_count
fixed_y = 0.0
Z, pulse_count = compute_y_fixed_profile(X, Y, delta_x_loc, Nmax=Nmax, fixed_y=fixed_y)
Z_m = Z
Z = -Z * 1e6

# Calculer le volume ablaté
volume = np.abs(np.trapezoid(np.trapezoid(Z_m, y, axis=0), x))
volume_um3 = volume * 1e18
print(f"Volume ablaté (µm³) : {volume_um3:.2f}")

# Données du tableau
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

# Figure avec subplots
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

cbar = fig.colorbar(surf, ax=ax1, label='Z (µm)', shrink=0.5, pad=0.20, aspect=20)
ax1.xaxis.pane.set_edgecolor('black')
ax1.yaxis.pane.set_edgecolor('black')
ax1.zaxis.pane.set_edgecolor('black')
ax1.xaxis.pane.set_linewidth(0.5)
ax1.yaxis.pane.set_linewidth(0.5)
ax1.zaxis.pane.set_linewidth(0.5) 

# Coupe 1D : z(x) à y = 0 µm (à droite, en haut)
ax2 = fig.add_subplot(gs[0, 1])
x_slice = x * 1e6
y_index = len(y) // 2
z_slice = Z[y_index, :]
ax2.plot(x_slice, z_slice, color='navy', linewidth=1, marker='o', markersize=0, 
         markevery=10,)

ax2.set_xlabel('x (µm)')
ax2.set_ylabel('Z(µm)')

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
         markevery=10, )
ax3.set_xlabel('y (µm)')
ax3.set_ylabel('Z(µm)')

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


# Tableau séparé
plt.figure(figsize=(5, 2))
table = plt.table(cellText=table_data, colLabels=['Paramètre', 'Valeur'], loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)
plt.axis('off')
plt.title('Volume Ablaté', fontsize=14, fontweight='bold')
plt.tight_layout()

plt.show()