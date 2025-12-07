import numpy as np
import matplotlib.pyplot as plt
from numba import jit

J = 1
L = 500
MSC = 1001 
T = 2
Spin = np.random.choice([-1,1], (L,L))
kb = 1
beta = 1/(T*kb)
vec = np.array([[1,0],[0,1], [0,-1],[-1,0]])
point = np.array([])
Delta = 0
chi = []

@jit(nopython = True)
def PBC(x):
    return x%L

@jit(nopython = True)
def delta(point,Spin):
    Delta = 0
    for a in vec:
        somsiad = PBC(point + a)        
        Delta += Spin[somsiad[0]][somsiad[1]]
    return 2 * J * Delta * Spin[point[0]][point[1]]


@jit(nopython = True)
def change_point(Delta, point, Spin):
    if Delta < 0:
        Spin[point[0]][point[1]] = Spin[point[0]][point[1]]*(-1)
    else:
        r = np.random.rand()
        prob = np.exp(-(beta * Delta))
        if r < prob:
            Spin[point[0]][point[1]] = Spin[point[0]][point[1]]*(-1)
    return Spin

@jit(nopython=True)
def correlation(Spin):
    L = len(Spin)
    rmax = L
    chi = []
    for i in range(L):
        chi_rows = 0 
        for r0 in range(rmax):
            chi_r = 0.0              # inicjalizacja!
            for j in range(L):
                chi_r += Spin[i][j] * Spin[i][PBC(j + r0)]
            chi_rows += chi_r / L
        chi.append(chi_rows / L)
    return chi

def draw(Spin):
    plt.imshow(Spin)
    plt.colorbar()
    plt.show()

def draw_chi(chi):
    X = np.linspace(1,L, L )
    Y = chi
    plt.scatter(X,Y)
    plt.show()

@jit(nopython = True)
def mainloop(MSC, point, Delta, Spin):  
    saved_spins = []  
    chi = []
    for i in range(MSC):
        for j in range(L**2):
            point = np.random.randint(0, L, 2)
            Delta = delta(point,Spin)
            Spin = change_point(Delta, point, Spin)
        if i in [1, 10, 100, 1000, 5000]:
            saved_spins.append(np.copy(Spin))
        if i in [10, 100, 1000]:
            chi.append(correlation(Spin))
    return saved_spins, chi


Spin_matrix, chi = mainloop(MSC, point, Delta, Spin)


for spin in Spin_matrix:
    draw(spin)

for h in chi:
    draw_chi(h)