import numpy as np
import matplotlib.pyplot as plt

# Paramètres de l'expérience
w0 = 25e-6  # Rayon du spot (m), 2w0 = 50 µm
f_rep = 800e3  # Fréquence de répétition (Hz)
v = 4  # Vitesse de balayage (m/s)
F = 0.37  # Fluence (J/cm²)
eta_res = 0.4  # Fraction de chaleur résiduelle
T_offset = 23  # Température initiale (°C)
rho = 7920  # Densité de l'acier (kg/m³)
c = 500  # Capacité calorifique (J/kg·K), approximation à 20 °C
kappa = 4e-6  # Diffusivité thermique (m²/s), approximation pour acier

# Calcul de l'énergie résiduelle
spot_area = np.pi * w0**2  # Surface du spot (m²)
E_p = F * 1e4 * spot_area  # Énergie par impulsion (J), conversion J/cm² -> J/m²
E_res = eta_res * E_p  # Chaleur résiduelle par impulsion (J)

# Volume chauffé pour estimer la température initiale
penetration_depth = 0.1e-6  # Profondeur de pénétration (m), ajustée pour atteindre ~3000 °C
volume = spot_area * penetration_depth  # Volume chauffé (m³)
mass = rho * volume  # Masse du volume chauffé (kg)
delta_T_initial = E_res / (mass * c)  # Hausse de température initiale (°C)

# Paramètres temporels
delta_t = 1 / f_rep  # Intervalle entre impulsions (s)
N_pulses = 20  # Nombre d'impulsions (pour atteindre ~20 µs)
t_max = N_pulses * delta_t  # Temps total simulé (s)
t_eval_per_pulse = np.linspace(1e-8, delta_t, 50)  # Points temporels par impulsion
t_eval = []
T_max = []

# Fonction pour la température d'une impulsion unique (Eq. 4)
def T_single_pulse(x, y, z, t, x0, y0, E_res, w0, rho, c, kappa):
    if t <= 1e-12:  # Juste après l'impulsion (impulsion ultra-courte)
        return delta_T_initial
    denom = np.pi * rho * c * np.sqrt(np.pi * kappa * t) * (8 * kappa * t + w0**2)
    num = 2 * E_res
    arg_exp = ((x - x0)**2 + (y - y0)**2)* (w0**2 /(8 * kappa * t + w0**2)  -1 ) / (4 * kappa * t ) - z**2 / (4 * kappa * t)
    return num / denom * np.exp(arg_exp)

# Calcul de la température maximale à la surface en suivant le centre de l'impulsion
for i in range(N_pulses):
    t_pulse = i * delta_t  # Moment de l'impulsion i
    x0_i = i * v * delta_t  # Position x de l'impulsion i
    # Ajouter les points temporels pour cette impulsion
    t_current = t_pulse + t_eval_per_pulse
    t_eval.extend(t_current * 1e6)  # Convertir en µs
    
    # Calculer la température pour cette période
    for t in t_eval_per_pulse:
        T = T_offset
        # Suivre le centre de la dernière impulsion
        x_eval = x0_i  # On évalue la température au centre de l'impulsion courante
        y_eval = 0
        for j in range(i + 1):  # Superposer toutes les impulsions précédentes
            t_j = t + (i - j) * delta_t  # Temps écoulé depuis l'impulsion j
            x0_j = j * v * delta_t  # Position x de l'impulsion j
        
            
            T += T_single_pulse(x_eval, y_eval, 0, t_j, x0_j, 0, E_res, w0, rho, c, kappa)
        T_max.append(T)

# Tracé de la Figure 7
plt.plot(t_eval, T_max, label='T_max (surface)')
plt.xlabel('Temps (µs)')
plt.ylabel('Température (°C)')

plt.grid(True)
plt.legend()
plt.yticks(np.arange(0, 1400, 250))
plt.ylim(0, 1400)
plt.show()