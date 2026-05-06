# A one-dimensional simulation of Grey-Scott system that describes many physical and biological systems
import numpy as np 
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
F = 0.025
k = 0.055
u = np.ones(N)
v = np.zeros(N)
t_snapshots = [0, 500, 1500, 2500, 10000]
N1 = N // 4; N2 = 3 * N // 4

u[N1:N2+1] = 0.4 + np.random.rand(N2-N1+1) * 0.2
v[N1:N2+1] = 0.2 + np.random.rand(N2-N1+1) * 0.2

def Laplace1d(f):
    return (np.roll(f, 1) + np.roll(f, -1) - 2 * f) / dx**2

def u_prym(u, v):
    return Du * Laplace1d(u) - u * v**2 + F - F * u 

def v_prym(u, v):
    return Dv * Laplace1d(v) + u * v**2 - (F + k) * v 

values = []

for i in range(T + 1):
    u += dt * u_prym(u, v)
    v += dt * v_prym(u, v)

    if i in t_snapshots:  
        values.append([np.copy(u), np.copy(v)])

fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
fig.suptitle("Grey-Scott Reaction-Diffusion Evolution")

indices = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]

for idx, (r, c) in enumerate(indices):
    ax[r, c].plot(x, values[idx][0], label='u (Inhibitor)')
    ax[r, c].plot(x, values[idx][1], label='v (Activator)')
    ax[r, c].set_title(f"Time Step: {t_snapshots[idx]}")
    ax[r, c].set_xlabel("Position x")
    ax[r, c].set_ylabel("Concentration")
    ax[r, c].legend(loc='upper right', fontsize='small')

ax[1, 2].axis('off')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(RESULTS_DIR, "grey_scott_1d.png"))
plt.close()