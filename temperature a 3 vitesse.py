import numpy as np
import matplotlib.pyplot as plt

# Paramètres de l'expérience
w0 = 25e-6  # Rayon du spot laser (m), diamètre = 50 µm
f_rep = 800e3  # Fréquence de répétition (Hz)
speeds = [3, 6]  # Vitesses du laser (m/s)
F = 0.37  # Fluence (J/cm²)
eta_res = 0.4  # Fraction de chaleur résiduelle
T_offset = 250  # Température initiale (°C)
rho = 7920  # Densité de l'acier (kg/m³)
c = 530  # Capacité thermique (J/kg·K), approximation à 20 °C
kappa = 4.7e-6  # Diffusivité thermique (m²/s), approximation pour l'acier

# Calcul de l'énergie résiduelle
spot_area = np.pi * w0**2  # Surface du spot (m²)
E_p = F * 1e4 * spot_area  # Énergie par impulsion (J), conversion J/cm² -> J/m²
E_res = eta_res * E_p  # Chaleur résiduelle par impulsion (J)

# Volume chauffé pour estimation initiale de la température
penetration_depth = 0.1e-6  # Profondeur de pénétration (m)
volume = spot_area * penetration_depth  # Volume chauffé (m³)
mass = rho * volume  # Masse du volume chauffé (kg)
delta_T_initial = E_res / (mass * c)  # Montée initiale de température (°C)

# Paramètres temporels
delta_t = 1 / f_rep  # Temps entre impulsions (s)
N_pulses = 20  # Nombre d'impulsions
t_max = N_pulses * delta_t  # Temps total simulé (s)
t_eval_per_pulse = np.linspace(1e-8, delta_t, 50)  # Points temporels par impulsion

# Fonction de température pour une impulsion unique
def T_single_pulse(x, y, z, t, x0, y0, E_res, w0, rho, c, kappa):
    if t <= 1e-12:  # Juste après l'impulsion (ultra-court)
        return delta_T_initial
    denom = np.pi * rho * c * np.sqrt(np.pi * kappa * t) * (8 * kappa * t + w0**2)
    num = 2 * E_res
    arg_exp = ((x - x0)**2 + (y - y0)**2) * (w0**2 / (8 * kappa * t + w0**2) - 1) / (4 * kappa * t) - z**2 / (4 * kappa * t)
    return num / denom * np.exp(arg_exp)   

# Configuration du style professionnel avec fond blanc
plt.rcParams['figure.facecolor'] = 'white'  # Fond blanc pour la figure
plt.rcParams['axes.facecolor'] = 'white'  # Fond blanc pour les axes
plt.figure(figsize=(14, 8), dpi=150)  # Taille plus grande et résolution augmentée

# Palette de couleurs pour les courbes
colors = ["#f7051d", "#0e0e0e"]  # Rouge et noir pour une bonne distinction

# Style professionnel
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 14,  # Taille de police de base augmentée
    'axes.titlesize': 18,  # Taille du titre augmentée
    'axes.labelsize': 16,  # Taille des labels des axes augmentée
    'legend.fontsize': 14,  # Taille de la légende augmentée
    'xtick.labelsize': 14,  # Taille des chiffres sur l'axe x augmentée
    'ytick.labelsize': 14,  # Taille des chiffres sur l'axe y augmentée
    'figure.dpi': 300  # Résolution élevée pour une meilleure qualité
})

# Liste pour stocker les températures de saturation
saturation_temps = []

# Simulation pour chaque vitesse
for idx, v in enumerate(speeds):
    t_eval = []
    T_max = []
    pulse_temps = []  # Stocke la température à la fin de chaque impulsion
    for i in range(N_pulses):
        t_pulse = i * delta_t  # Temps de l'impulsion i
        x0_i = i * v * delta_t  # Position x de l'impulsion i
        t_current = t_pulse + t_eval_per_pulse
        t_eval.extend(t_current * 1e6)  # Convertit en µs
        
        for t in t_eval_per_pulse:
            T = T_offset
            x_eval = x0_i  # Évalue au centre de l'impulsion actuelle
            y_eval = 0
            for j in range(i + 1):  # Superpose toutes les impulsions précédentes
                t_j = t + (i - j) * delta_t  # Temps écoulé depuis l'impulsion j
                x0_j = j * v * delta_t  # Position x de l'impulsion j
                T += T_single_pulse(x_eval, y_eval, 0, t_j, x0_j, 0, E_res, w0, rho, c, kappa)
            T_max.append(T)
        
        # Enregistre la température à la fin de l'impulsion actuelle
        final_t_idx = -1
        pulse_temps.append(T_max[final_t_idx])
    
    # Calcul de la température de saturation (T_sat) pour la vitesse actuelle
    T_max_array = np.array(T_max)
    minima = []
    for i in range(1, N_pulses):
        start_idx = i * len(t_eval_per_pulse) - 1
        minima.append(np.min(T_max_array[(i-1)*len(t_eval_per_pulse):start_idx]))
    minima_after_10_pulses = minima[9:]  # Prend les minima après 10 impulsions
    T_sat = np.mean(minima_after_10_pulses)  # Moyenne pour la température de saturation
    saturation_temps.append(T_sat)
    print(f"Pour v = {v} m/s, T_sat = {T_sat:.1f} °C")
    
    # Affiche la température à la fin de chaque impulsion
    print(f"\nTempératures à la fin de chaque impulsion pour vitesse {v} m/s:")
    for pulse, temp in enumerate(pulse_temps, 1):
        print(f"Impulsion {pulse}: {temp:.2f} °C")
    
    # Annote la température maximale
    max_temp = max(T_max)
    print(f"Température maximale pour vitesse {v} m/s: {max_temp:.2f} °C")
    plt.plot(t_eval, T_max, label=f'Vitesse = {v} m/s', color=colors[idx], linewidth=2, marker='o', markersize=5, markevery=50)
    plt.annotate(f'{max_temp:.0f} °C', 
                 xy=(t_eval[T_max.index(max(T_max))], max_temp), 
                 xytext=(t_eval[T_max.index(max(T_max))]+2, max_temp-150),
                 textcoords='data',
                 bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white", alpha=0.9),
                 arrowprops=dict(facecolor='black', arrowstyle='->', connectionstyle="arc3,rad=0.2"))
    
    # Trace la ligne de saturation
    plt.axhline(saturation_temps[idx], color='green', linestyle=':', linewidth=2, label=f'T_sat v={v} m/s')

# Configuration du graphique

plt.xlabel('Temps (µs)', fontsize=16, labelpad=10)
plt.ylabel('Température (°C)', fontsize=16, labelpad=10)
plt.grid(True, linestyle='--', alpha=0.5, linewidth=1)  # Grille plus épaisse et moins opaque
plt.legend(fontsize=14, loc='lower right')
plt.yticks(np.arange(0, 1400, 200), fontsize=14)
plt.xticks(fontsize=14)
plt.ylim(0, 1400)
plt.xlim(0, max(t_eval) * 1.05)  # Ajoute une marge à droite
plt.tight_layout()  # Ajuste les marges pour un rendu propre
plt.show()