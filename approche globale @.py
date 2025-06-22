import numpy as np
import matplotlib.pyplot as plt

# Paramètres
delta = 18e-9  # m (18 nm)
F_th = 550  # J/m^2 (0.055 J/cm^2)
omega_0 = 12.5e-6  # m (12.5 µm)
Fc_0 = 1e5  # J/m^2 (10 J/cm^2)
Fm_0 = Fc_0 / 2  # Fluence moyenne
E_p = np.pi * omega_0**2 * Fm_0  # Énergie par impulsion
x_0 = (omega_0 / np.sqrt(2)) * np.sqrt(np.log(Fc_0 / F_th))  # Rayon du cratère

# Grille spatiale
r = np.linspace(-x_0*2, x_0*2, 1000)  # m

# Fonction pour calculer la profondeur d'ablation par impulsion
def z_k(r, Fc_k, delta, F_th, omega_0):
    ln_term = np.log(Fc_k / F_th) - 2 * (r**2 / omega_0**2)
    z = delta * ln_term
    z = np.where(ln_term > 0, z, 0)
    return z

# Fonction pour calculer la surface du cratère
def surface_area(Z_n_center, x_0):
    if Z_n_center <= 0:
        return np.pi * x_0**2
    return (np.pi / 6.0) * (x_0 / (Z_n_center**2)) * (
        (x_0**2 + 4 * Z_n_center**2)**1.5 - x_0**3
    )

# Simulation avec saturation globale
def simulate_with_saturation(N, r, delta, F_th, omega_0, Fc_0, E_p, x_0):
    Z_n = np.zeros_like(r)
    Fc_k = Fc_0
    for k in range(1, N + 1):
        z_k_r = z_k(r, Fc_k, delta, F_th, omega_0)
        Z_n += z_k_r
        Z_n_center = delta * np.log(Fc_k / F_th)
        if Z_n_center > 0:
            S_n = surface_area(Z_n_center, x_0)
            Fm_k = E_p / S_n
            Fc_k = 2 * Fm_k
            if Fc_k < F_th:
                Fc_k = F_th
                break
    return Z_n

# Simulation sans saturation
def simulate_without_saturation(N, r, delta, F_th, omega_0, Fc_0):
    Z_n = np.zeros_like(r)
    for k in range(1, N + 1):
        Z_n += z_k(r, Fc_0, delta, F_th, omega_0)
    return Z_n

# Style professionnel
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 12,
    'axes.titlesize': 2,
    'axes.labelsize': 5,
    'legend.fontsize': 3,
    'xtick.labelsize': 5,
    'ytick.labelsize': 5,
    'figure.dpi': 300
})

# Création de la figure avec deux sous-graphiques
fig = plt.figure(figsize=(12, 5), constrained_layout=True)
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])

# Figure 3a : Profils pour différents nombres d'impulsions
ax1 = fig.add_subplot(gs[0, 0])
pulse_numbers = [100, 1000, 2500, 5000]
colors = ['navy', 'crimson', 'darkgreen', 'purple']
max_depths = []
for N, color in zip(pulse_numbers, colors):
    Z_n = simulate_with_saturation(N, r, delta, F_th, omega_0, Fc_0, E_p, x_0)
    ax1.plot(r * 1e6, -Z_n * 1e6, color=color, linewidth=1, marker='o', markersize=0,
             markevery=100, label=f'{N} impulsions')
    max_depths.append([f'{N} impulsions', f'{-np.max(Z_n)*1e6:.2f}'])
ax1.set_xlabel('x (µm)')
ax1.set_ylabel('z (µm)')
ax1.legend()
ax1.grid(True, alpha=0.3)
# Ajout d'un cadre carré autour du graphique 3a
ax1.set_box_aspect(1)  # Rend le graphique carré
ax1.spines['top'].set_visible(True)
ax1.spines['right'].set_visible(True)
ax1.spines['left'].set_visible(True)
ax1.spines['bottom'].set_visible(True)
ax1.spines['top'].set_linewidth(0.5)
ax1.spines['right'].set_linewidth(0.5)
ax1.spines['left'].set_linewidth(0.5)
ax1.spines['bottom'].set_linewidth(0.5)
ax1.spines['top'].set_color('black')

ax1.spines['right'].set_color('black')
ax1.spines['left'].set_color('black')
ax1.spines['bottom'].set_color('black')
# Figure 3b : Comparaison pour 5000 impulsions
ax2 = fig.add_subplot(gs[0, 1])
N_5000 = 5000
Z_with_saturation = simulate_with_saturation(N_5000, r, delta, F_th, omega_0, Fc_0, E_p, x_0)
Z_without_saturation = simulate_without_saturation(N_5000, r, delta, F_th, omega_0, Fc_0)
ax2.plot(r * 1e6, -Z_with_saturation * 1e6, color='navy', linewidth=1, marker='o', markersize=0,
         markevery=100, label='Avec saturation')
ax2.plot(r * 1e6, -Z_without_saturation * 1e6, color='crimson', linestyle='--', linewidth=1,
         marker='s', markersize=0, markevery=100, label='Sans saturation')
ax2.set_xlabel('x (µm)')
ax2.set_ylabel('z (µm)')
ax2.legend()
ax2.grid(True, alpha=0.3)
# Ajout d'un cadre carré autour du graphique 3b
ax2.set_box_aspect(1)  # Rend le graphique carré
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
# Tableau pour les profondeurs maximales
plt.figure(figsize=(5, 3))
table = plt.table(cellText=max_depths, colLabels=['Impulsions', 'Profondeur max (µm)'],
                  loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)
plt.axis('off')
plt.title('Profondeurs Maximales', fontsize=14, fontweight='bold')
plt.tight_layout()

plt.show()