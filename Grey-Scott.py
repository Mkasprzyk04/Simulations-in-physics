#A one dimensional simulation of Grey-Scott system that describes many physical an biological systems
import numpy as np 
import matplotlib.pyplot as plt 
N = 100
L = 2 
dx = 0.02
x = np.linspace(dx,L, N)
dt = 1 
T = 10000
Du = 2 * 10**(-5)
Dv = 10**(-5)
F = 0.025
k = 0.055
u = np.ones(N)
v = np.zeros(N)
t = [0, 500, 1500, 2500, 10000]
N1=N//4; N2=3*N//4

u[N1:N2+1] = 0.4 + np.random.rand(N2-N1+1) * 0.2
v[N1:N2+1] = 0.2 + np.random.rand(N2-N1+1) * 0.2

def PBC(x):
    return x % N

def Laplace1d(f):
    return (np.roll(f,1) + np.roll(f,-1) - 2 * f) / dx**2

def u_prym(u,v):
    return Du * Laplace1d(u) - u * v**2  + F - F * u 

def v_prym(u,v):
    return Dv * Laplace1d(v) + u * v**2   - (F+k) * v 

values = []

for i in range(T + 1):
    u +=  dt * u_prym(u,v)
    v +=  dt * v_prym(u,v)

    if i in t:  
        values.append([np.copy(u), np.copy(v)])

fig, ax = plt.subplots(nrows=2, ncols=3)

ax[0, 0].plot(x , values[0][0], x,values[0][1] ) 
ax[0, 1].plot(x , values[1][0], x, values[1][1] ) 
ax[0, 2].plot(x , values[2][0], x, values[2][1] ) 
ax[1, 0].plot(x , values[3][0], x, values[3][1] ) 
ax[1, 1].plot(x , values[4][0], x, values[4][1] ) 
plt.show()