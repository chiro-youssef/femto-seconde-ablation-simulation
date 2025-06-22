import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

# Paramètres
N_max = 50  # Nombre maximum d'impulsions
N = np.arange(1, N_max + 1)
F_th1 = 10.0  # Seuil initial de fluence
F_th_inf = 2.0  # Seuil asymptotique de fluence
S = 0.5  # Exposant fictif
F_th_N = F_th_inf + (F_th1 - F_th_inf) * N**(-S - 1)

# Configuration de la figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [2, 1]})
plt.subplots_adjust(wspace=0.4)

# Graphique de l'évolution de F_th(N)
line, = ax1.plot([], [], 'b-', label='Seuil d\'ablation \( F_{th}(N) \)')
ax1.set_xlim(0, N_max)
ax1.set_ylim(0, 12)
ax1.set_xlabel('Nombre d\'impulsions \( N \)')
ax1.set_ylabel('Fluence seuil \( F_{th}(N) \)')
ax1.legend()
ax1.grid(True)

# Représentation visuelle du matériau et de l'ablation
material = Rectangle((0, 0), 1, 1, linewidth=1, edgecolor='black', facecolor='gray')
fragile_zone = Rectangle((0.1, 0.1), 0, 0, linewidth=1, edgecolor='yellow', facecolor='yellow', alpha=0.5)  # Zone fragilisée
ablation_zone = Rectangle((0.1, 0.1), 0, 0, linewidth=1, edgecolor='red', facecolor='red')  # Zone ablatée
ax2.add_patch(material)
ax2.add_patch(fragile_zone)
ax2.add_patch(ablation_zone)
ax2.set_xlim(0, 1.2)
ax2.set_ylim(0, 1.2)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_title('Interaction laser-matériau')

# Texte pour le nombre d'impulsions
text = ax1.text(0.5, 0.9, '', transform=ax1.transAxes, ha='center')

# Initialisation de l'animation
def init():
    line.set_data([], [])
    fragile_zone.set_width(0)
    fragile_zone.set_height(0)
    ablation_zone.set_width(0)
    ablation_zone.set_height(0)
    text.set_text('')
    return line, fragile_zone, ablation_zone, text

# Mise à jour de l'animation
def update(frame):
    # Mise à jour du graphique
    line.set_data(N[:frame + 1], F_th_N[:frame + 1])
    
    # Première impulsion : création d'une zone fragilisée
    if frame == 0:
        fragile_zone.set_width(0.3)
        fragile_zone.set_height(0.3)
    # Impulsions suivantes : transformation en ablation profonde
    elif frame > 0:
        ablation_width = 0.8 * (1 - F_th_N[frame] / F_th1)  # Taille de l'ablation augmente
        ablation_zone.set_width(ablation_width)
        ablation_zone.set_height(ablation_width)
        # Réduire la zone fragilisée progressivement
        fragile_width = max(0, 0.3 - 0.02 * frame)  # Décroissance de la fragilisation
        fragile_zone.set_width(fragile_width)
        fragile_zone.set_height(fragile_width)
    
    # Mise à jour du texte
    text.set_text(f'Impulsions: {frame + 1}')
    
    return line, fragile_zone, ablation_zone, text

# Création de l'animation
ani = FuncAnimation(fig, update, frames=N_max, init_func=init, blit=True, interval=100)

# Affichage
plt.show()

# Option pour sauvegarder (nécessite ffmpeg installé)
# ani.save('laser_ablation_animation.mp4', writer='ffmpeg', fps=10)