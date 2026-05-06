#An interesting case of animation in critical state
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numba import njit
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "Results")

os.makedirs(RESULTS_DIR, exist_ok=True)

hmax = 4
L = 51
tmax = 1
S_min = 1
S_max = 300

grid = np.full((L, L), 7, dtype=np.int64)
times = []
grains = []
avalanche_sizes = []
im_collection = []

def relax(grid, hmax):
    L = grid.shape[0]
    a = 0 
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

        im_collection.append(np.copy(grid))

@njit
def random_add(grid):
    x = np.random.randint(0, grid.shape[0])
    y = np.random.randint(0, grid.shape[1])
    A = int((L-1) / 2) 
    grid[A, A] += 1

for t in range(tmax):
    
    s = relax(grid, hmax)
    
    if t % 100 == 0:
        im_collection.append(np.copy(grid))

fig, ax = plt.subplots()
img = ax.imshow(im_collection[0], cmap='plasma', vmin=0, vmax=8)
ax.axis('off')
ax.set_title("Sandpile evolution (critical state)")
cbar = fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Height / number of grains")

def update(i):
    img.set_data(im_collection[i])
    return img,

anim = FuncAnimation(fig, update, frames=len(im_collection), interval=10, blit=True)
anim.save(os.path.join(RESULTS_DIR, 'sandpile_animation1.gif'), writer='pillow', fps=10)
plt.close()