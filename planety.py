#Simulation of trajectory and potential energy in two body system using Euler, Verlet and Leapfrog method of solving differential equations
import numpy as np
import matplotlib.pyplot as plt

def F(r,M,m):
    K = -0.01 * M *m / np.linalg.norm(r)**2
    return K * r / np.linalg.norm(r)

def V(r,M,m):
    K = -0.01 *M *m / np.linalg.norm(r)
    return K

def Euler(r0,dt, p0,T):
    X = np.array([])
    Y = np.array([])
    P = np.array([])
    time = np.array([])
    U = np.array([])

    for i  in range(T):
        X = np.append(X,r0[0])
        Y = np.append(Y,r0[1])
        P = np.append(P,np.linalg.norm(p0)**2/m)
        U = np.append(U, V(r0,M,m))

        time = np.append(time, (i + 1)*dt)
        r0 = r0 + (p0 * dt) / m  + (F(r0,M,m) * (dt)**2) / (2 * m)
        p0 = p0 + F(r0,M,m) * dt

    return X, Y, P, U,  time

def Verlet(r0,dt, p0,T):
    X = np.array([])
    Y = np.array([])
    P = np.array([])
    time = np.array([])
    rmin = np.array([])
    U = np.array([])
    rsr = np.array([])

    for i in range(T):
        X = np.append(X,r0[0])
        Y = np.append(Y,r0[1])
        P = np.append(P,np.linalg.norm(p0)**2/m)
        time = np.append(time, (i+1)*dt)
        U = np.append(U, V(r0,M,m))

        if i == 0:
            rmin = r0 - (p0 * dt) / m  + (F(r0,M,m) * (dt)**2) / (2 * m)
            r0 = 2 * r0 - rmin + (F(r0,M,m) * (dt)**2) / m
            rsr = r0
            p0 = (rsr - rmin)/ dt
        else:        
            r0 = 2 * rsr - rmin + (F(r0,M,m) * (dt)**2) / m
            rmin = rsr
            rsr = r0
            p0 = (rsr - rmin) / dt 
    return X, Y, P, U, time

def Żabcia(r0,dt, p0,T):
    X = np.array([])
    Y = np.array([])
    P = np.array([])
    time = np.array([])
    vmin = p0 /m
    U = np.array([])

    for i in range(T):
        X = np.append(X,r0[0])
        Y = np.append(Y,r0[1])
        P = np.append(P,np.linalg.norm(vmin)**2 * m/2)
        time = np.append(time, i*dt)
        U = np.append(U, V(r0,M,m))
        r0 = r0 + vmin * dt
        vmin = vmin + F(r0,M,m) * dt / 2*m
        
    
    return X, Y, P, U, time

M = 500
m = 0.1
dt = 0.001
T = 50000
r0 = np.array([2,0])
p0 = np.array([0,0.1])

X,Y ,P, U,  time = Euler(r0,dt,p0, T)

plt.scatter(X,Y)
plt.show()
plt.scatter(time, U)
plt.show()
plt.scatter(time, P)
plt.show()
plt.scatter(time,U + P)
plt.show()

X,Y ,P, U,  time = Verlet(r0,dt,p0, T)

plt.scatter(X,Y)
plt.show()
plt.scatter(time, U)
plt.show()
plt.scatter(time, P)
plt.show()
plt.scatter(time,U + P)
plt.show()