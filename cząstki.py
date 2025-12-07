import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import imageio.v2 as imageio
import os, glob
import numba 

particle_number = 16
box_size = 8.0
eps = 1.0
sigma = 1.0
promien = sigma / 2
dt = 0.025
m = 1.0
kb = 1.0
temp = 0.1
rc = 2.5 * sigma
T = 5000  # liczba kroków
temp_ext = 0.1

class Czastka:
    def __init__(self, promien, pos, vel):
        self.promien = promien
        self.r = pos.copy()
        self.v = vel.copy()
        self.v_half = self.v.copy()
        self.F = np.zeros(2)

def lj_force(r12):
    r = np.linalg.norm(r12)
    if r == 0:
        return np.zeros_like(r12)
    F_scalar = -(48 * eps / sigma**2) * ((sigma / r)**14 - 0.5 * (sigma / r)**8)
    return F_scalar * (r12 / r)

def apply_pbc(r):
    return r % box_size

def minimum_image(r12):
    return r12 - box_size * np.round(r12 / box_size)

def vels(particle_number):
    v = np.random.rand(particle_number, 2) - 0.5
    V_mean = np.mean(v, axis=0)
    Ek = np.sum(np.linalg.norm(v, axis=1)**2) / (2 * m)
    fs = np.sqrt(kb * temp * particle_number / (2 * Ek))
    v = (v - V_mean) * fs
    return v

def init_particles(particle_number, v):
    particles = []
    side = int(np.sqrt(particle_number))
    spacing = box_size / side
    for i in range(side):
        for j in range(side):
            pos = np.array([(i + 0.5) * spacing, (j + 0.5) * spacing])
            particles.append(Czastka(promien, pos, v[len(particles)]))
    return particles

def compute_forces(particles):
    N = len(particles)
    for p in particles:
        p.F[:] = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            rij = minimum_image(particles[j].r - particles[i].r)
            r = np.linalg.norm(rij)
            if r < rc and r > 1e-12:
                F = lj_force(rij)
                particles[i].F += F
                particles[j].F -= F

def kinetic_energy(particles):
    return sum(0.5 * m * np.linalg.norm(p.v)**2 for p in particles)

def pressure(particles):
    V = box_size ** 2
    N = len(particles)

    # część kinetyczna
    p_kin = 0.0
    for p in particles:
        p_kin += m * np.dot(p.v, p.v)
    p_kin *= 1.0 / (2 * V)


    w = 0.0
    for i in range(N):
        for j in range(i+1, N):
            rij = minimum_image(particles[j].r - particles[i].r)
            Fij = lj_force(rij)
            w += np.dot(rij, Fij)

    p_conf = w / (2 * V)

    return p_kin + p_conf


def integrate_leapfrog_isokinetic_zabka(particles, T_ext):
    N = len(particles)

    v_half_old = [p.v_half.copy() for p in particles]

    v_u_list = []
    for p, v_old in zip(particles, v_half_old):
        v_u = v_old + (p.F / m) * (dt / 2.0)
        v_u_list.append(v_u)

    sum_m_v2 = sum(m * np.dot(vu, vu) for vu in v_u_list)
    T_inst = sum_m_v2 / (2.0 * N * kb)

    if T_inst <= 0:
        eta = 1.0
    else:
        eta = np.sqrt(T_ext / T_inst)

    new_v_half_list = []
    for p, v_old in zip(particles, v_half_old):
        v_half_new = (2.0 * eta - 1.0) * v_old + eta * (p.F / m) * dt
        new_v_half_list.append(v_half_new)

    for p, v_half_new in zip(particles, new_v_half_list):
        p.v_half = v_half_new.copy()
        p.r += p.v_half * dt
        p.r = apply_pbc(p.r)

    compute_forces(particles)

    for p in particles:
        p.v = p.v_half + (p.F / m) * (dt / 2.0)

    return T_inst, eta



def rysuj(particles, step, Ek):
    plt.clf()
    ax = plt.gca()
    for p in particles:
        cir = Circle((p.r[0], p.r[1]), radius=p.promien)
        ax.add_patch(cir)
    plt.xlim((0, box_size))
    plt.ylim((0, box_size))
    plt.title(f"Step {step}")
    plt.gcf().set_size_inches((6,6))
    plt.savefig(f"img{step:06d}.png")

def animacja(filenames):
    with imageio.get_writer("movie.gif", mode="I") as writer:
        for fn in filenames:
            writer.append_data(imageio.imread(fn))

def usun_png():
    for fname in glob.glob("img*.png"):
        os.remove(fname)

def potential_energy(particles):
    N = len(particles)
    U = 0.0
    for i in range(N):
        for j in range(i+1, N):
            rij = minimum_image(particles[j].r - particles[i].r)
            r = np.linalg.norm(rij)
            if r < rc and r > 1e-12:
                U += 4 * eps * ((sigma / r)**12 - (sigma / r)**6)
    return U

v = vels(particle_number)
particles = init_particles(particle_number, v)
compute_forces(particles)


temps = []
energies = []
pressures_list = []
potential_energies = []
filenames = []

for i in range(T):
    T_inst, eta_val = integrate_leapfrog_isokinetic_zabka(particles, temp_ext)

    Ek = kinetic_energy(particles)
    Ep = potential_energy(particles)
    E_tot = Ek + Ep

    temps.append(T_inst)
    energies.append(E_tot)
    pressures_list.append(pressure(particles))
    potential_energies.append(Ep)

    if i % 50 == 0:
        rysuj(particles, i, Ek)
        filenames.append(f"img{i:06d}.png")

energies_per_particle = [E / particle_number for E in energies]
animacja(filenames)
usun_png()

plt.figure(figsize=(12,5))
plt.plot(temps, label="Temperatura (termostat)")
plt.plot(pressures_list, label="Ciśnienie")
plt.plot(energies_per_particle, label="Energia całkowita / N")
plt.title("Temperatura, ciśnienie i energia całkowita na cząstkę w czasie")
plt.xlabel("Krok")
plt.ylabel("Wartość")
plt.legend()
plt.show()

plt.figure(figsize=(12,5))
plt.plot(energies, label="Energia całkowita")
plt.plot([kinetic_energy(particles) for _ in range(T)], label="Energia kinetyczna")
plt.plot(potential_energies, label="Energia potencjalna")
plt.title("Energia całkowita, kinetyczna i potencjalna w czasie")
plt.xlabel("Krok")
plt.ylabel("Energia")
plt.legend()
plt.show()