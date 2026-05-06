# A two dimensional simulation of Grey-Scott system that describes many physical and biological systems
import numpy as np 
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "Results")
os.makedirs(RESULTS_DIR, exist_ok=True)

N = 100
L = 2 
dx = 0.02
x = np.linspace(dx, L, N)
dt = 1 
T = 10000
Du = 2 * 10**(-5)
Dv = 10**(-5)
F = 0.037
k = 0.06
im_collection = []

t_snapshots = [0, 500, 1500, 2500, 10000]
u = np.ones((N, N))
v = np.zeros((N, N))
N1 = N // 4; N2 = 3 * N // 4
u[N1:N2+1, N1:N2+1] = 0.4 + np.random.rand(N2-N1+1, N2-N1+1) * 0.2
v[N1:N2+1, N1:N2+1] = 0.2 + np.random.rand(N2-N1+1, N2-N1+1) * 0.2

def Laplace2d(f):
    return (np.roll(f, 1, 0) + np.roll(f, -1, 0) + np.roll(f, 1, 1) + np.roll(f, -1, 1) - 4 * f) / dx**2

def u_prym(u, v):
    return Du * Laplace2d(u) - u * v**2 + F - F * u 

def v_prym(u, v):
    return Dv * Laplace2d(v) + u * v**2 - (F + k) * v 

values = []

for i in range(T + 1):
    u += dt * u_prym(u, v)
    v += dt * v_prym(u, v)
    
    if i % 100 == 0:
        im_collection.append(np.copy(v))
    if i in t_snapshots:  
        values.append([np.copy(u), np.copy(v)])

fig, ax = plt.subplots()
img = ax.imshow(im_collection[0], cmap='plasma', vmin=0, vmax=0.5)
ax.axis('off')
ax.set_title("Grey-Scott Reaction-Diffusion (v concentration)")
cbar = fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Concentration of v")

def update(i):
    img.set_data(im_collection[i])
    return img,

anim = FuncAnimation(fig, update, frames=len(im_collection), interval=50, blit=True)
anim.save(os.path.join(RESULTS_DIR, 'GC1.gif'), writer='pillow', fps=5)
plt.close()