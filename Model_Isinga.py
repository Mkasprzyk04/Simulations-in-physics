import numpy as np
import matplotlib.pyplot as plt
from numba import jit

J = 1
L = 100
MSC = 5001 
T = 2
Spin = np.random.choice([-1,1], (L,L))
kb = 1
beta = 1/(T*kb)
vec = np.array([[1,0],[0,1], [0,-1],[-1,0]])
point = np.array([])
Delta = 0
chi_temp = [0.5]

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
    L = len(Spin) / 2 
    rmax = L
    chi = []
    for r0 in range(L):
        chi_r = 0 
        for i in range(rmax):
            for j in range(L):
                chi_r += Spin[i][j] * Spin[i][PBC(j + r0)]
        chi.append(chi_r/ L**2)
    return chi

@jit(nopython=True)
def cluster_size(chi):
    rmax = 0
    L = len(chi)
    mianownik = 0
    for i in range(L):
        if chi[i] > 0.3:
            rmax +=1 
        else:
            break

    for j in range(rmax ):
        mianownik += np.log(chi[j])

    mianownik *= -2
    R = rmax*(rmax + 1) / mianownik
    return R

@jit(nopython=True)
def mainloop(MSC, point, Delta, Spin, chi_temp):  
    saved_spins = []  
    chi = []
    size = []

    for i in range(MSC):
        for j in range(L**2):
            point = np.random.randint(0, L, 2)
            Delta = delta(point,Spin)
            Spin = change_point(Delta, point, Spin)
        
        if i in [1, 10, 100, 1000, 5000]:
            saved_spins.append(np.copy(Spin))
            chi.append(correlation(Spin))

        if i in [10,20,50, 100,200,500,1000, 2000, 5000]:
            chi_temp = correlation(Spin)
            size.append(np.log(cluster_size(chi_temp)))
            chi_temp = [1]
            
    return saved_spins, chi, size 

def draw(Spin):
    plt.imshow(Spin)
    plt.colorbar()
    plt.show()

def draw_chi(chi):
    L = len(Spin)
    L = int(L/2)
    X = np.linspace(1,L, L)
    Y = chi
    plt.scatter(X,Y)
    plt.show()


Spin_matrix, chi, size = mainloop(MSC, point, Delta, Spin, chi_temp)


for spin in Spin_matrix:
    draw(spin)
for h in chi:
    draw_chi(h)

sizex =  np.log([10,20,50, 100,200,500,1000, 2000, 5000])

a, b = np.polyfit(sizex, size, 1)
x_line = np.linspace(min(sizex), max(sizex), 100)
y_line = a * x_line + b
plt.scatter(sizex, size)
plt.plot(x_line, y_line, label=f"y = {a:.2f}x + {b:.2f}", linewidth=2)
plt.legend()
plt.show()
