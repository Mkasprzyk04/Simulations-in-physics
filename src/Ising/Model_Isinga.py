# Simulation of Ising model, showing how does size of cluster size and correlation function depend on iteration 
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import os

# --- FOLDER SAVE CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "Results")

# Creates the Results folder if it doesn't exist yet
os.makedirs(RESULTS_DIR, exist_ok=True)

J = 1
L = 100
MSC = 2001 
T = 2
Spin = np.random.choice([-1,1], (L,L))
kb = 1
beta = 1/(T*kb)
vec = np.array([[1,0],[0,1], [0,-1],[-1,0]])
point = np.array([])
Delta = 0
chi_temp = [0.5]

@jit(nopython=True)
def PBC(x):
    return x % L

@jit(nopython=True)
def delta(point, Spin):
    Delta = 0
    for a in vec:
        neighbor = PBC(point + a)        
        Delta += Spin[neighbor[0]][neighbor[1]]
    return 2 * J * Delta * Spin[point[0]][point[1]]

@jit(nopython=True)
def change_point(Delta, point, Spin):
    if Delta < 0:
        Spin[point[0]][point[1]] = Spin[point[0]][point[1]] * (-1)
    else:
        r = np.random.rand()
        prob = np.exp(-(beta * Delta))
        if r < prob:
            Spin[point[0]][point[1]] = Spin[point[0]][point[1]] * (-1)
    return Spin

@jit(nopython=True)
def correlation(Spin):
    L = len(Spin) // 2 
    rmax = L
    chi = []
    for r0 in range(L):
        chi_r = 0 
        for i in range(rmax):
            for j in range(L):
                chi_r += Spin[i][j] * Spin[i][PBC(j + r0)]
        chi.append(chi_r / L**2)
    return chi

@jit(nopython=True)
def cluster_size(chi):
    rmax = 0
    L = len(chi)
    denominator = 0
    for i in range(L):
        if chi[i] > 0.3:
            rmax += 1 
        else:
            break

    for j in range(rmax):
        denominator += np.log(chi[j])

    denominator *= -2
    R = rmax * (rmax + 1) / denominator
    return R

@jit(nopython=True)
def mainloop(MSC, point, Delta, Spin, chi_temp):  
    saved_spins = []  
    chi = []
    size = []

    for i in range(MSC):
        for j in range(L**2):
            point = np.random.randint(0, L, 2)
            Delta = delta(point, Spin)
            Spin = change_point(Delta, point, Spin)
        
        if i in [1, 10, 100, 1000, 5000]:
            saved_spins.append(np.copy(Spin))
            chi.append(correlation(Spin))

        if i in [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]:
            chi_temp = correlation(Spin)
            size.append(np.log(cluster_size(chi_temp)))
            chi_temp = [1]
            
    return saved_spins, chi, size 

def draw(Spin, idx):
    plt.figure()
    plt.imshow(Spin, cmap='viridis')
    plt.colorbar(label='Spin (+1 / -1)')
    plt.xlabel("Position X")
    plt.ylabel("Position Y")
    plt.title(f"Spin matrix (save step {idx+1})")
    plt.savefig(os.path.join(RESULTS_DIR, f"Spin_matrix_{idx+1}.png"))
    plt.close() # Closing the window prevents plots from overlapping in memory

def draw_chi(chi, idx):
    plt.figure()
    L_half = len(Spin) // 2
    X = np.linspace(1, L_half, L_half)
    Y = chi
    plt.scatter(X, Y)
    plt.xlabel("Distance (r)")
    plt.ylabel("Correlation function C(r)")
    plt.title(f"Correlation function (save step {idx+1})")
    plt.savefig(os.path.join(RESULTS_DIR, f"Correlation_{idx+1}.png"))
    plt.close()


# Run the main loop
Spin_matrix, chi, size = mainloop(MSC, point, Delta, Spin, chi_temp)

# Draw and save matrices and correlations
for idx, spin in enumerate(Spin_matrix):
    draw(spin, idx)

for idx, h in enumerate(chi):
    draw_chi(h, idx)

# --- CLUSTER SIZE PLOT ---
sizex = np.log([10, 20, 50, 100, 200, 500, 1000, 2000]) # Removed 5000 because MSC = 2001, so the script doesn't reach it

plt.figure()
a, b = np.polyfit(sizex, size, 1)
x_line = np.linspace(min(sizex), max(sizex), 100)
y_line = a * x_line + b
plt.scatter(sizex, size, color='blue')
plt.plot(x_line, y_line, label=f"Fit: y = {a:.2f}x + {b:.2f}", linewidth=2, color='red')

# X and Y axis labels
plt.xlabel("ln(Iterations)")
plt.ylabel("ln(Cluster size R)")
plt.title("Cluster size vs. iterations")
plt.legend()

# Save the final plot to the Results folder
plt.savefig(os.path.join(RESULTS_DIR, "Cluster_size_vs_iterations.png"))
plt.close()