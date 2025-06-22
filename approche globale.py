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
r = np.linspace(-x_0*2, x_0*2,1000)  # m

# Fonction pour calculer la profondeur d'ablation par impulsion
def z_k(r, Fc_k, delta, F_th, omega_0):
    # Fluence spatiale : F(r) = Fc_k * exp(-2*r^2/omega_0^2)
    ln_term = np.log(Fc_k / F_th) - 2 * (r**2 / omega_0**2)
    z = delta * ln_term
    # Ablation uniquement si F(r) > F_th
    z = np.where(ln_term > 0, z, 0)
    return z

# Fonction pour calculer la surface du cratère
def surface_area(Z_n_center, x_0):
    """
    Calcule la surface effective du cratère à partir de la profondeur centrale Z_n_center,
    selon la formule analytique : 
      S_n = (π/6) * (x0 / Z_c^2) * [ (x0^2 + 4 Z_c^2)^(3/2) - x0^3 ]
    Si pas d'ablation (Z_c <= 0), on retourne la surface initiale π x0^2.
    """
    if Z_n_center <= 0:
        return np.pi * x_0**2
    # Formule fermée pour la surface d'un profil de cratère
    return (np.pi / 6.0) * (x_0 / (Z_n_center**2)) * (
        (x_0**2 + 4 * Z_n_center**2)**1.5 - x_0**3
    )

# Simulation avec saturation globale
def simulate_with_saturation(N, r, delta, F_th, omega_0, Fc_0, E_p, x_0):
    Z_n = np.zeros_like(r)
    Fc_k = Fc_0
    for k in range(1, N + 1):
        # Calculer la profondeur pour l'impulsion k
        z_k_r = z_k(r, Fc_k, delta, F_th, omega_0)
        Z_n += z_k_r
        # Calculer la profondeur centrale pour estimer la surface
        Z_n_center = delta * np.log(Fc_k / F_th)
        if Z_n_center > 0:
            S_n = surface_area(Z_n_center, x_0)
            Fm_k = E_p / S_n
            Fc_k = 2 * Fm_k
            # S'assurer que Fc_k ne tombe pas en dessous de F_th
            if Fc_k < F_th:
                Fc_k = F_th
                break  # Arrêter l'ablation si la fluence est trop faible
    return Z_n

# Simulation sans saturation
def simulate_without_saturation(N, r, delta, F_th, omega_0, Fc_0):
    Z_n = np.zeros_like(r)
    for k in range(1, N + 1):
        Z_n += z_k(r, Fc_0, delta, F_th, omega_0)
    return Z_n

# Figure 3a : Profils pour différents nombres d'impulsions
pulse_numbers = [100, 1000, 2500, 5000]
plt.figure(figsize=(10, 6))
for N in pulse_numbers:
    Z_n = simulate_with_saturation(N, r, delta, F_th, omega_0, Fc_0, E_p, x_0)
    plt.plot(r * 1e6, -Z_n * 1e6, label=f'{N} impulsions')  # Inverser Z_n pour concavité vers le bas
plt.xlabel('Rayon (µm)')
plt.ylabel('Profondeur (µm)')
plt.title('Figure 3a : Profil du cratère d\'ablation avec saturation globale')
plt.legend()
plt.grid(True)
plt.show()

# Figure 3b : Comparaison pour 5000 impulsions
N_5000 = 5000
Z_with_saturation = simulate_with_saturation(N_5000, r, delta, F_th, omega_0, Fc_0, E_p, x_0)
Z_without_saturation = simulate_without_saturation(N_5000, r, delta, F_th, omega_0, Fc_0)

plt.figure(figsize=(10, 6))
plt.plot(r * 1e6, -Z_with_saturation * 1e6, 'k-', label='Avec saturation')  # Inverser Z_n
plt.plot(r * 1e6, -Z_without_saturation * 1e6, 'k--', label='Sans saturation')  # Inverser Z_n
plt.xlabel('Rayon (µm)')
plt.ylabel('Profondeur (µm)')
plt.title('Figure 3b : Profil du cratère pour 5000 impulsions')
plt.legend()
plt.grid(True)
plt.show()