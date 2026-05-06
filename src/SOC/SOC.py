# Simulation of a sandpile, showing how this system achieves a critical point 
# and how the size of avalanches depends on their frequency
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numba import njit
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "Results")

# Creates the Results folder if it doesn't exist yet
os.makedirs(RESULTS_DIR, exist_ok=True)

hmax = 4
L = 31
tmax = 50000
S_min = 1
S_max = 300

grid = np.zeros((L, L), dtype=np.int64)
times = []
grains = []
avalanche_sizes = []
im_collection = [] 

@njit
def relax(grid, hmax):
    avalanche_size = 0
    L = grid.shape[0]
    while True:
        over = np.where(grid > hmax)
        if over[0].size == 0:
            break
        for x, y in zip(over[0], over[1]):
            grid[x, y] -= 4
            if x+1 < L: grid[x+1, y] += 1
            if x-1 >= 0: grid[x-1, y] += 1
            if y+1 < L: grid[x, y+1] += 1
            if y-1 >= 0: grid[x, y-1] += 1
            avalanche_size += 1
    return avalanche_size

@njit
def random_add(grid):
    x = np.random.randint(0, grid.shape[0])
    y = np.random.randint(0, grid.shape[1])
    grid[x, y] += 1

for t in range(tmax):
    random_add(grid)
    
    s = relax(grid, hmax)
    
    avalanche_sizes.append(s)
    grains.append(grid.sum())
    times.append(t)
    
    if t % 100 == 0:
        im_collection.append(np.copy(grid))

sizes, counts = np.unique(avalanche_sizes, return_counts=True)

mask = (sizes >= S_min) & (sizes <= S_max)
sizes_f = sizes[mask]
counts_f = counts[mask]

logS = np.log10(sizes_f + 1) 
logN = np.log10(counts_f + 1)
coeffs = np.polyfit(logS, logN, 1)
tau = -coeffs[0]
A = 10**coeffs[1]

plt.figure()
plt.plot(times, grains, color='blue')
plt.xlabel("Time (iterations)")
plt.ylabel("Total number of grains")
plt.title("Growth of grain count over time")
plt.grid()
plt.savefig(os.path.join(RESULTS_DIR, "Total_grains_over_time.png"))
plt.close()

plt.figure()
plt.loglog(sizes, counts, 'o', markersize=4, label="Simulation data")
plt.loglog(sizes_f, A * sizes_f**(-tau), '-', label=f"Fit: τ={tau:.2f}")
plt.xlabel("Avalanche size (S)")
plt.ylabel("Frequency N(S)")
plt.title("Avalanche size distribution (power law)")
plt.legend()
plt.grid(True, which="both", ls="--")
plt.savefig(os.path.join(RESULTS_DIR, "Avalanche_distribution.png"))
plt.close()

# --- ANIMATION: SANDPILE EVOLUTION ---
fig, ax = plt.subplots()
img = ax.imshow(im_collection[0], cmap='plasma', vmin=0, vmax=hmax)
ax.axis('off')
ax.set_title("Sandpile evolution")

def update(i):
    img.set_data(im_collection[i])
    return img,

anim = FuncAnimation(fig, update, frames=len(im_collection), interval=50, blit=True)
# Saving animation to folder Results
anim.save(os.path.join(RESULTS_DIR, 'sandpile_animation.gif'), writer='pillow', fps=10)
plt.close()