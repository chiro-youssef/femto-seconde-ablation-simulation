import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image

# Paramètres physiques (inchangés)
w0 = 25e-6
f_rep = 800e3
v = 4.6
F = 0.37
eta_res = 0.4
T_offset = 0
rho = 7920
c = 500
kappa = 4e-6

# Calculs physiques (inchangés)
spot_area = np.pi * w0**2
E_p = F * 1e4 * spot_area
E_res = eta_res * E_p
penetration_depth = 19e-9
volume = spot_area * penetration_depth
mass = rho * volume
delta_T_initial = 0
print(f"Température initiale par impulsion : {delta_T_initial:.2f} °C")

# Paramètres temporels et grille spatiale (inchangés)
delta_t = 1 / f_rep
N_pulses = 30
t_max = N_pulses * delta_t
nx, ny = 1000, 1000
x = np.linspace(-20e-6, 250e-6, nx)
y = np.linspace(-100e-6, 100e-6, ny)
X, Y = np.meshgrid(x, y)

# Points temporels pour courbes et animation (inchangés)
t_eval_per_pulse = np.linspace(10e-12, delta_t, 100)
t_eval = []
Tth = []
T_max_values = []

# Fonction T_single_pulse (inchangée)
def T_single_pulse(x, y, z, t, x0, y0, E_res, w0, rho, c, kappa):
    if t <= 1e-12:
        return 0
    denom = np.pi * rho * c * np.sqrt(np.pi * kappa * t) * (8 * kappa * t + w0**2)
    num = 2 * E_res
    arg_exp = ((x - x0)**2 + (y - y0)**2) * (w0**2 / (8 * kappa * t + w0**2) - 1) / (4 * kappa * t) - z**2 / (4 * kappa * t)
    return num / denom * np.exp(arg_exp)

# Calcul des courbes Tth et T_max_values (inchangé)
for i in range(N_pulses):
    t_pulse = i * delta_t
    x0_i = i * v * delta_t
    t_current = t_pulse + t_eval_per_pulse
    t_eval.extend(t_current * 1e6)
    for t in t_eval_per_pulse:
        T = T_offset
        x_eval = x0_i
        y_eval = 0
        for j in range(i + 1):
            t_j = t + (i - j) * delta_t
            x0_j = j * v * delta_t
            if t_j <= 0:
                continue
            T += T_single_pulse(x_eval, y_eval, 0, t_j, x0_j, 0, E_res, w0, rho, c, kappa)
        Tth.append(T)
        T_max_values.append(T)

# Points temporels pour l'animation (inchangés)
frames = 100
indices = np.linspace(0, len(t_eval) - 1, frames, dtype=int)
times = [t_eval[i] / 1e6 for i in indices]
N_impulses = [int(t * f_rep) for t in times]

# Variable globale pour l'état de pause
paused = False
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 12,
    'axes.titlesize': 9,
    'axes.labelsize': 9,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 200
})
# Initialisation de la figure pour la carte 2D
fig = plt.figure(figsize=(10, 6))  # Taille augmentée pour meilleure lisibilité
ax = fig.add_subplot(111, aspect='equal')  # Aspect ratio égal pour éviter la distorsion
im = ax.imshow(np.zeros_like(X), extent=[-20, 250, -100, 100], cmap='jet', vmin=0, vmax=1200)  # Colormap jet

ax.set_xlabel(" x (µm)", fontsize=12)
ax.set_ylabel(" y (µm)", fontsize=12)
ax.set_xticks(np.arange(-20, 251, 50))  # Graduations claires pour x
ax.set_yticks(np.arange(-100, 101, 50))  # Graduations claires pour y
ax.grid(False)  # Pas de grille pour plus de clarté
cbar = fig.colorbar(im, ax=ax, label="Température (°C)", pad=0.02)  # Barre de couleur plus proche
cbar.set_ticks(np.arange(0, 1201, 200))  # Ticks espacés pour la barre de couleur
plt.tight_layout()

# Fonction de mise à jour pour l'animation
def update(frame):
    if paused:  # Ne met pas à jour si l'animation est en pause
        return [im]
    for coll in ax.collections:
        coll.remove()
    for line in ax.lines:
        line.remove()
    t = times[frame]
    N = N_impulses[frame]
    T_total = np.zeros_like(X)
    for j in range(N + 1):
        t_j = t - j * delta_t
        x0_j = j * v * delta_t
        y0_j = 0
        if t_j <= 0:
            continue
        temp_contribution = T_single_pulse(X, Y, 0, t_j, x0_j, y0_j, E_res, w0, rho, c, kappa)
        T_total += temp_contribution
    T_total += T_offset
    x0_current = N * v * delta_t
    idx_x = np.argmin(np.abs(x - x0_current))
    idx_y = np.argmin(np.abs(y - 0))
    T_center = T_total[idx_y, idx_x]
    idx_t_eval = indices[frame]
    T_ref = Tth[idx_t_eval]
    print(f"Frame {frame + 1}/{frames}, t = {t*1e6:.2f} µs")
    print(f"Temp max (grille) = {T_total.max():.2f} °C")
    print(f"Temp au centre (x={x0_current*1e6:.2f} µm, y=0) = {T_center:.2f} °C")
    print(f"Temp de référence (Tth/T_max_values) = {T_ref:.2f} °C")
    im.set_array(T_total)
    ax.plot(x0_current*1e6, 0, 'x', color='red', markersize=8, label='Centre du faisceau')  # Marqueur visible
    ax.set_title(f"{' (En pause)' if paused else ''}", fontsize=12)
    ax.legend(loc='upper right')
    return [im]

# Gestionnaire d'événements pour pause/reprise
def on_key_press(event):
    global paused
    if event.key == ' ':  # Appuyer sur la barre d'espace pour basculer pause/reprise
        paused = not paused
        ax.set_title(f"Distribution de température à t = {times[ani.frame_seq.frame]*1e6:.2f} µs{' (En pause)' if paused else ''}", fontsize=12)
        fig.canvas.draw()

# Connecter le gestionnaire d'événements
fig.canvas.mpl_connect('key_press_event', on_key_press)

# Animation
ani = FuncAnimation(fig, update, frames=frames, interval=50, blit=True)
ani.save('improved_heat_distribution_with_pause.gif', writer='pillow', fps=10)

# Graphique des courbes (inchangé)
fig2 = plt.figure(figsize=(8, 6))
ax2 = fig2.add_subplot(111)
ax2.plot(t_eval, T_max_values, label='Température maximale (centre)', color='blue')
ax2.plot(t_eval, Tth, label='Tth (centre de l\'impulsion)', color='green', linestyle='--')
ax2.axhline(y=3000, color='red', linestyle='--', label='Seuil de saturation (3000 °C)')
ax2.set_xlabel('Temps (µs)')
ax2.set_ylabel('Température (°C)')
ax2.set_title('Variation de la température maximale à v = 4.6 m/s, F = 0.37 J/cm²')
ax2.legend()
ax2.grid(True)
ax2.set_yticks(np.arange(0, 1400, 500))
ax2.set_ylim(0, 1400)
plt.tight_layout()
plt.show()