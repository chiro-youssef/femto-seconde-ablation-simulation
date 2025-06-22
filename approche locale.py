import numpy as np
import matplotlib.pyplot as plt

# Paramètres
delta = 18e-9  # m (18 nm)
F_th = 550  # J/m^2 (0.055 J/cm^2)
omega_0 = 12.5e-6  # m (12.5 µm)
Fc_0 = 1e5  # J/m^2 (10 J/cm^2)
Fm_0 = Fc_0 / 2  # Fluence moyenne initiale
E_p = np.pi * omega_0**2 * Fm_0  # Énergie par impulsion
x_0 = (omega_0 / np.sqrt(2)) * np.sqrt(np.log(Fc_0 / F_th))  # Rayon du cratère

# Grille spatiale
r = np.linspace(-x_0 * 1.5, x_0 * 1.5, 1000)  # m

# Fonction pour calculer la profondeur d'ablation par impulsion
def z_k(r, Fc_k, delta, F_th, omega_0):
    # Fluence spatiale : F(r) = Fc_k * exp(-2*r^2/omega_0^2)
    ln_term = np.log(Fc_k / F_th) - 2 * (r**2 / omega_0**2)
    z = delta * ln_term
    # Ablation uniquement si F(r) > F_th
    z = np.where(ln_term > 0, z, 0)
    return z

# Fonction pour calculer la surface locale
def local_surface_area(dZ_dr, x_0):
    # Approximation de la surface locale : S(r) = pi * x_0^2 * sqrt(1 + (dZ/dr)^2)
    S_n = np.pi * x_0**2 * np.sqrt(1 + dZ_dr**2)
    return S_n

# Simulation avec approche locale
def simulate_local_approach(N, r, delta, F_th, omega_0, Fc_0, E_p, x_0):
    Z_n = np.zeros_like(r)
    Fc_k = np.ones_like(r) * Fc_0  # Fluence initiale uniforme
    dr = r[1] - r[0]  # Pas spatial pour la dérivée
    for k in range(1, N + 1):
        # Calculer la profondeur pour l'impulsion k
        z_k_r = z_k(r, Fc_k, delta, F_th, omega_0)
        Z_n += z_k_r
        # Calculer la dérivée spatiale de Z_n pour estimer la surface locale
        dZ_dr = np.gradient(Z_n, dr)
        # Calculer la surface locale à chaque r
        S_n = local_surface_area(dZ_dr, x_0)
        # Mettre à jour la fluence locale
        Fm_k = E_p / S_n
        Fc_k = 2 * Fm_k
        # S'assurer que Fc_k ne tombe pas en dessous de F_th
        Fc_k = np.where(Fc_k < F_th, F_th, Fc_k)
    return Z_n

# Simulation sans saturation (identique à l'approche globale sans ajustement)
def simulate_without_saturation(N, r, delta, F_th, omega_0, Fc_0):
    Z_n = np.zeros_like(r)
    for k in range(1, N + 1):
        Z_n += z_k(r, Fc_0, delta, F_th, omega_0)
    return Z_n

# Figure 5 : Profils pour différents nombres d'impulsions (approche locale)
pulse_numbers = [100, 1000, 2500, 5000]
plt.figure(figsize=(10, 6))
for N in pulse_numbers:
    Z_n = simulate_local_approach(N, r, delta, F_th, omega_0, Fc_0, E_p, x_0)
    plt.plot(r * 1e6, -Z_n * 1e6, label=f'{N} impulsions')  # Inverser Z_n pour concavité vers le bas
plt.xlabel('Rayon (µm)')
plt.ylabel('Profondeur (µm)')
plt.title('Figure 5 : Profil du cratère d\'ablation avec approche locale')
plt.legend()
plt.grid(True)
plt.show()

# Figure 5b (optionnelle) : Comparaison pour 5000 impulsions
N_5000 = 5000
Z_local = simulate_local_approach(N_5000, r, delta, F_th, omega_0, Fc_0, E_p, x_0)
Z_without_saturation = simulate_without_saturation(N_5000, r, delta, F_th, omega_0, Fc_0)

plt.figure(figsize=(10, 6))
plt.plot(r * 1e6, -Z_local * 1e6, 'k-', label='Approche locale')  # Inverser Z_n
plt.plot(r * 1e6, -Z_without_saturation * 1e6, 'k--', label='Sans saturation')  # Inverser Z_n
plt.xlabel('Rayon (µm)')
plt.ylabel('Profondeur (µm)')
plt.title('Figure 5b : Profil du cratère pour 5000 impulsions (approche locale vs sans saturation)')
plt.legend()
plt.grid(True)
plt.show()