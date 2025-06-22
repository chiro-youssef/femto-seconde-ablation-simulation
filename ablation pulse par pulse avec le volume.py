import numpy as np
import matplotlib.pyplot as plt
from math import ceil, log
import pandas as pd

# Paramètres donnés dans l'article
d = 28  # Diamètre du spot laser (µm)
w0 = d / 2  # Rayon du spot (µm)
phi_LO = 0.9  # Taux de chevauchement des lignes
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
    power = N_eff**(S - 1)
    return delta_1 * power, Fth_1 * power

# Dimensions de la zone simulée (µm)
Lx = 200  # Longueur en x (µm)
Ly = 200  # Longueur en y (µm)

# Grille de points pour x et y
nx, ny = 200, 200  # Résolution de la grille
x = np.linspace(-20, Lx+20, nx)
y = np.linspace(-20, Ly+20, ny)
X, Y = np.meshgrid(x, y)

# Nombre d'impulses et de lignes
n_impulses = ceil((Lx) / Delta_x)  # Nombre d'impulsions par ligne
n_lines = ceil(Ly / dH)  # Nombre de lignes

# Initialisation des matrices
Z = np.zeros((ny, nx))  # Profondeur d'ablation
N = np.zeros((ny, nx))  # Compteur d'impulsions par position

# Calcul de Z(x, y) pour le balayage parallèle avec effets d'incubation
for j in range(n_lines):
    yj = j * dH  # Position y de la j-ème ligne
    for i in range(n_impulses):
        xi = i * Delta_x   # Position x de la i-ème impulsion
        # Distance au centre de l'impulsion
        r2 = (X - xi)**2 + (Y - yj)**2
        # Mettre à jour le compteur d'impulsions
        N += (r2 <= (w0**2)).astype(int)  # Compter l'impulsion si dans le rayon du spot
        # Calcul de Fth_N et delta_N avec N_eff
        delta_N, Fth_N = varying_params(N_eff)
        # Rayon d'ablation r_th basé sur Fth_N
        r_th_squared = (w0**2 / 2) * np.log(F0 / Fth_N)
        # Condition d'ablation
        mask = (r2 <= r_th_squared) & (r_th_squared > 0)
        # Contribution de l'impulsion
        ln_F0_Fth_N = np.log(F0 / Fth_N)
        contribution = delta_N * (ln_F0_Fth_N - 2 * r2 / (w0**2))
        # Ajouter la contribution
        Z += np.where(mask, contribution, 0)

# Fonction pour calculer le volume par impulsion
def calculate_V_pulse(delta_N, omega_0, Fc_0, Fth_N):
    ln_ratio = np.log(Fc_0 / Fth_N)
    V_pulse = (np.pi / 4) * delta_N * (omega_0**2) * (ln_ratio**2)
    return V_pulse  # en µm³

# Calcul du volume ablaté pour Z gril
N_total = n_impulses * n_lines  # Nombre total d'impulsions
delta_N, Fth_N = varying_params(N_eff)  # Utiliser N_eff pour le calcul du volume
V_pulse = calculate_V_pulse(delta_N, w0, F0, Fth_N)  # Volume par impulsion
V_pulse_eff = V_pulse * N_total  # Volume effectif
V_dot_zgril = f * V_pulse_eff  # Taux d'ablation volumique (µm³/s)
V_dot_zgril_mm3 = V_dot_zgril * 1e-9  # Conversion en mm³/s

# Création du tableau avec pandas
data = {
    'Cas ablation pulse par pulse': ['Z gril (Gravure bidimensionnelle)'],
    'Taux d\'ablation volumique (mm³/s)': [f'{V_dot_zgril_mm3:.5f}']
}
df = pd.DataFrame(data)

# Création de la figure pour le tableau
fig4 = plt.figure(figsize=(8, 3))
ax4 = fig4.add_subplot(111)
ax4.axis('off')  # Désactiver les axes

# Création du tableau dans Matplotlib
table_data = [df.columns] + df.values.tolist()
table = plt.table(cellText=table_data,
                  colLabels=None,
                  cellLoc='center',
                  loc='center',
                  bbox=[0, 0, 1, 1])

# Mise en forme du tableau
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)  # Ajuster la taille des cellules
for (i, j), cell in table.get_celld().items():
    if i == 0:
        cell.set_text_props(weight='bold')  # En-tête en gras
    cell.set_edgecolor('black')
    cell.set_facecolor('white' if i == 0 else '#f0f0f0')

plt.title('Tableau des résultats', fontsize=12, pad=20)
plt.tight_layout()

# Figure 1 : Profil 3D
fig1 = plt.figure(figsize=(8, 6))
ax1 = fig1.add_subplot(111, projection='3d')
surf = ax1.plot_surface(X, Y, -Z, cmap='viridis', edgecolor='none', antialiased=True)
ax1.set_xlabel('X (µm)')
ax1.set_ylabel('Y (µm)')
ax1.set_zlabel('Z (µm)')
ax1.set_title(f'Profil 3D (v = {v} µm/s, f = {f} Hz)')
fig1.colorbar(surf, ax=ax1, shrink=0.5, aspect=10, pad=0.1, label='Profondeur (µm)')
ax1.view_init(elev=30, azim=135)
plt.tight_layout()

# Figure 2 : Coupe le long de X
fig2 = plt.figure(figsize=(8, 6))
ax2 = fig2.add_subplot(111)
y_mid = Ly / 2
y_idx = np.argmin(np.abs(y - y_mid))
ax2.plot(x, -Z[y_idx, :], label='Section A-A (X-direction)')
ax2.set_xlabel('X (µm)')
ax2.set_ylabel('Z (µm)')
ax2.set_title('Profil direction X')
ax2.grid(True)
ax2.legend()
plt.tight_layout()

# Figure 3 : Coupe le long de Y
fig3 = plt.figure(figsize=(8, 6))
ax3 = fig3.add_subplot(111)
x_mid = Lx / 2
x_idx = np.argmin(np.abs(x - x_mid))
ax3.plot(y, -Z[:, x_idx], label='Section B-B (Y-direction)')
ax3.set_xlabel('Y (µm)')
ax3.set_ylabel('Z (µm)')
ax3.set_title('Profil direction Y')
ax3.grid(True)
ax3.legend()
plt.tight_layout()

# Affichage des graphiques
plt.show()