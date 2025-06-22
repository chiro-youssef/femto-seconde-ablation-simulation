import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set a modern plotting style


# -------------------------------------------------------------------
# 1) Physical parameters (SI)
# -------------------------------------------------------------------
delta0   = 18e-9         # δ(1) = 18 nm
Fth0     = 550           # F_th(1) = 550 J/m² (0.055 J/cm²)
omega_0  = 14e-6         # beam radius (m)
delta_x  = 5e-6          # pulse spacing fixed at 5 µm (used in equation)         # pulse spacing for y-scan (µm)
d        = 2* omega_0    # spot diameter (m)
Fm       = 4.06e4        # J/m² (4.06 J/cm²)
Fc       = 2 * Fm        # peak fluence (J/m²)
S        = 0.8           # exponent d’incubation fixé
Nmax     = 25            # limite sur N_eff
v        = 1             # la vitesse de palayege
f        = 200e3          # frequance de repetetion 
delta_y  =v/f
# Calculate overlap rates
R_x = 1 - delta_x / d    # Taux de chevauchement en x (pour l'équation)
R_y = 1 - delta_y / d    # Taux de chevauchement en y (pour le balayage)
print(f"Taux de chevauchement des impulsions en x (R_x): {R_x*100:.2f}%")
print(f"Taux de chevauchement des impulsions en y (R_y): {R_y*100:.2f}%")

# -------------------------------------------------------------------
# 2) Grid for x (scan direction) and y (transverse)
# -------------------------------------------------------------------
x = np.linspace(0, 100e-6, 100)  # m, de 0 à 100 µm, higher resolution
y = np.linspace(-150e-6, 150e-6, 100)  # m, de -50 µm à 50 µm
X, Y = np.meshgrid(x, y)  # grille 2D

# -------------------------------------------------------------------
# 3) Core functions
# -------------------------------------------------------------------
def ablation_depth(y, Fc_loc, delta_loc, Fth_loc, delta_x_loc):
    """
    Profondeur d’ablation par impulsion selon la formule :
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
    power = N_eff**(S - 1)
    return delta0 * power, Fth0 * power

def compute_y_scan_profile(X, Y, delta_x_loc, delta_y_loc, Nmax=None):
    """
    Profil d’ablation pour un balayage le long de y avec compteur.
    """
    Z = np.zeros_like(X)
    y_positions = np.arange(-100.5e-6, 100.5e-6 + delta_y_loc, delta_y_loc)  # de -47.5 µm à 47.5 µm
    N_eff = 2 * omega_0 / delta_y_loc  # N_eff basé sur delta_y pour le balayage
    if Nmax is not None:
        N_eff = min(N_eff, Nmax)
    delta_N, Fth_N = varying_params(N_eff)

    pulse_count = 0  # Compteur d'impulsions
    for y_i in y_positions:
        pulse_count += 1
        y_relative = Y - y_i  # Distance transversale par rapport à y_i
        Z += ablation_depth(y_relative, Fc, delta_N, Fth_N, delta_x_loc)
    print(f"Nombre total d'impulsions : {pulse_count}")

    return Z


# -------------------------------------------------------------------
# 4) Compute Z for the y-scan trajectory
# -------------------------------------------------------------------
Z = -compute_y_scan_profile(X, Y, delta_x, delta_y, Nmax=Nmax) * 1e6  # Convertir en µm

# -------------------------------------------------------------------
# 5) Plot 3D
# -------------------------------------------------------------------
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Tracer la surface avec une palette moderne
surf = ax.plot_surface(X * 1e6, Y * 1e6, Z, cmap='magma', rstride=1, cstride=1, 
                       linewidth=0.1, antialiased=True, shade=True)

# Labels et titre avec style amélioré
ax.set_xlabel('Position x (µm)', fontsize=12, fontweight='bold')
ax.set_ylabel('Position y (µm)', fontsize=12, fontweight='bold')
ax.set_zlabel('Profondeur Z (µm)', fontsize=12, fontweight='bold')
ax.set_title(f'Profondeur d’ablation (balayage le long de y, 20 impulsions)\n'
             f'Δy = {delta_y*1e6:.1f} µm, R_y = {R_y*100:.1f}% (S={S}, N_eff≤{Nmax})',
             fontsize=13, fontweight='bold', pad=15)

# Barre de couleur
fig.colorbar(surf, ax=ax, label='Profondeur (µm)', shrink=0.6, pad=0.1)

plt.tight_layout()
plt.show()

# -------------------------------------------------------------------
# 6) Plot 1D slice: z(x) at y = 0 µm
# -------------------------------------------------------------------
plt.figure(figsize=(12, 8))
x_slice = x * 1e6
y_index = len(y) // 2  # Correspond à y ≈ 0 µm
z_slice = Z[y_index, :]  # Profondeur le long de x à y fixe
plt.plot(x_slice, z_slice, color='navy', linewidth=2, marker='o', markersize=4, 
         markevery=10, label=f'Profil à y = 0 µm')
plt.xlabel('Position x (µm)', fontsize=12, fontweight='bold')
plt.ylabel('Profondeur Z (µm)', fontsize=12, fontweight='bold')
plt.title(f'Profondeur d’ablation pour Δy = {delta_y*1e6:.1f} µm, R_y = {R_y*100:.1f}%', 
          fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# -------------------------------------------------------------------
# 7) Plot 1D slice: z(y) at x = 50 µm
# -------------------------------------------------------------------
plt.figure(figsize=(12, 8))
y_slice = y * 1e6
x_index = len(x) // 2  # Correspond à x ≈ 50 µm
z_slice = Z[:, x_index]  # Profondeur le long de y à x fixe
plt.plot(y_slice, z_slice, color='crimson', linewidth=2, marker='s', markersize=2, 
         markevery=10, label=f'Profil à x = 50 µm')
plt.xlabel('Position y (µm)', fontsize=12, fontweight='bold')
plt.ylabel('Profondeur Z (µm)', fontsize=12, fontweight='bold')
plt.title(f'Profondeur d’ablation pour Δy = {delta_y*1e6:.1f} µm, R_y = {R_y*100:.1f}%', 
          fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()