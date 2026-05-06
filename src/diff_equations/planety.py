# Simulation of trajectory and potential energy in two body system using Euler, Verlet and Leapfrog method of solving differential equations
import numpy as np
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "Results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def F(r, M, m):
    K = -0.01 * M * m / np.linalg.norm(r)**2
    return K * r / np.linalg.norm(r)

def V(r, M, m):
    K = -0.01 * M * m / np.linalg.norm(r)
    return K

def Euler(r0, dt, p0, T):
    X = np.array([])
    Y = np.array([])
    P = np.array([])
    time = np.array([])
    U = np.array([])

    for i in range(T):
        X = np.append(X, r0[0])
        Y = np.append(Y, r0[1])
        P = np.append(P, np.linalg.norm(p0)**2 / m)
        U = np.append(U, V(r0, M, m))

        time = np.append(time, (i + 1) * dt)
        r0 = r0 + (p0 * dt) / m  + (F(r0, M, m) * (dt)**2) / (2 * m)
        p0 = p0 + F(r0, M, m) * dt

    return X, Y, P, U, time

def Verlet(r0, dt, p0, T):
    X = np.array([])
    Y = np.array([])
    P = np.array([])
    time = np.array([])
    r_prev = np.array([])
    U = np.array([])
    r_mid = np.array([])

    for i in range(T):
        X = np.append(X, r0[0])
        Y = np.append(Y, r0[1])
        P = np.append(P, np.linalg.norm(p0)**2 / m)
        time = np.append(time, (i + 1) * dt)
        U = np.append(U, V(r0, M, m))

        if i == 0:
            r_prev = r0 - (p0 * dt) / m  + (F(r0, M, m) * (dt)**2) / (2 * m)
            r0 = 2 * r0 - r_prev + (F(r0, M, m) * (dt)**2) / m
            r_mid = r0
            p0 = (r_mid - r_prev) / dt
        else:        
            r0 = 2 * r_mid - r_prev + (F(r0, M, m) * (dt)**2) / m
            r_prev = r_mid
            r_mid = r0
            p0 = (r_mid - r_prev) / dt 
    return X, Y, P, U, time

def Leapfrog(r0, dt, p0, T):
    X = np.array([])
    Y = np.array([])
    P = np.array([])
    time = np.array([])
    vmin = p0 / m
    U = np.array([])

    for i in range(T):
        X = np.append(X, r0[0])
        Y = np.append(Y, r0[1])
        P = np.append(P, np.linalg.norm(vmin)**2 * m / 2)
        time = np.append(time, i * dt)
        U = np.append(U, V(r0, M, m))
        r0 = r0 + vmin * dt
        vmin = vmin + F(r0, M, m) * dt / 2*m
        
    return X, Y, P, U, time

M = 500
m = 0.1
dt = 0.001
T = 50000
r0 = np.array([2, 0])
p0 = np.array([0, 0.1])


X, Y, P, U, time = Euler(r0, dt, p0, T)

plt.figure()
plt.scatter(X, Y)
plt.title("Euler Method - Trajectory")
plt.xlabel("Position X")
plt.ylabel("Position Y")
plt.savefig(os.path.join(RESULTS_DIR, "Euler_trajectory.png"))
plt.close()

plt.figure()
plt.scatter(time, U)
plt.title("Euler Method - Potential Energy")
plt.xlabel("Time")
plt.ylabel("Potential Energy (U)")
plt.savefig(os.path.join(RESULTS_DIR, "Euler_potential_energy.png"))
plt.close()

plt.figure()
plt.scatter(time, P)
plt.title("Euler Method - Kinetic Energy")
plt.xlabel("Time")
plt.ylabel("Kinetic Energy (K)")
plt.savefig(os.path.join(RESULTS_DIR, "Euler_kinetic_energy.png"))
plt.close()

plt.figure()
plt.scatter(time, U + P)
plt.title("Euler Method - Total Energy")
plt.xlabel("Time")
plt.ylabel("Total Energy (E)")
plt.savefig(os.path.join(RESULTS_DIR, "Euler_total_energy.png"))
plt.close()


X, Y, P, U, time = Verlet(r0, dt, p0, T)

plt.figure()
plt.scatter(X, Y)
plt.title("Verlet Method - Trajectory")
plt.xlabel("Position X")
plt.ylabel("Position Y")
plt.savefig(os.path.join(RESULTS_DIR, "Verlet_trajectory.png"))
plt.close()

plt.figure()
plt.scatter(time, U)
plt.title("Verlet Method - Potential Energy")
plt.xlabel("Time")
plt.ylabel("Potential Energy (U)")
plt.savefig(os.path.join(RESULTS_DIR, "Verlet_potential_energy.png"))
plt.close()

plt.figure()
plt.scatter(time, P)
plt.title("Verlet Method - Kinetic Energy")
plt.xlabel("Time")
plt.ylabel("Kinetic Energy (K)")
plt.savefig(os.path.join(RESULTS_DIR, "Verlet_kinetic_energy.png"))
plt.close()

plt.figure()
plt.scatter(time, U + P)
plt.title("Verlet Method - Total Energy")
plt.xlabel("Time")
plt.ylabel("Total Energy (E)")
plt.savefig(os.path.join(RESULTS_DIR, "Verlet_total_energy.png"))
plt.close()