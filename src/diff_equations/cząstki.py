# Script that simulates interactions between particles within the thermostat, with periodic boundary conditions and showing animation of forming solid state 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import imageio.v2 as imageio
import os
import glob 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "Results")
os.makedirs(RESULTS_DIR, exist_ok=True)

particle_number = 16
box_size = 8.0
eps = 1.0
sigma = 1.0
radius = sigma / 2
dt = 0.025
m = 1.0
kb = 1.0
temp = 0.1
rc = 2.5 * sigma
T = 5000 
temp_ext = 0.1

class Particle:
    def __init__(self, radius, pos, vel):
        self.radius = radius
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
            particles.append(Particle(radius, pos, v[len(particles)]))
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

def integrate_leapfrog_isokinetic(particles, T_ext):
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

def draw(particles, step, Ek):
    plt.clf()
    ax = plt.gca()
    for p in particles:
        cir = Circle((p.r[0], p.r[1]), radius=p.radius)
        ax.add_patch(cir)
    plt.xlim((0, box_size))
    plt.ylim((0, box_size))
    plt.title(f"Step {step}")
    plt.gcf().set_size_inches((6,6))
    plt.savefig(os.path.join(RESULTS_DIR, f"img{step:06d}.png"))
    plt.close()

def animate(filenames):
    with imageio.get_writer(os.path.join(RESULTS_DIR, "movie.gif"), mode="I") as writer:
        for fn in filenames:
            writer.append_data(imageio.imread(fn))

def remove_pngs():
    for fname in glob.glob(os.path.join(RESULTS_DIR, "img*.png")):
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
    T_inst, eta_val = integrate_leapfrog_isokinetic(particles, temp_ext)

    Ek = kinetic_energy(particles)
    Ep = potential_energy(particles)
    E_tot = Ek + Ep

    temps.append(T_inst)
    energies.append(E_tot)
    pressures_list.append(pressure(particles))
    potential_energies.append(Ep)

    if i % 50 == 0:
        draw(particles, i, Ek)
        filenames.append(os.path.join(RESULTS_DIR, f"img{i:06d}.png"))

energies_per_particle = [E / particle_number for E in energies]
animate(filenames)
remove_pngs()

plt.figure(figsize=(12,5))
plt.plot(temps, label="Temperature (thermostat)")
plt.plot(pressures_list, label="Pressure")
plt.plot(energies_per_particle, label="Total energy / N")
plt.title("Temperature, pressure and total energy per particle over time")
plt.xlabel("Step")
plt.ylabel("Value")
plt.legend()
plt.savefig(os.path.join(RESULTS_DIR, "thermodynamics.png"))
plt.close()

plt.figure(figsize=(12,5))
plt.plot(energies, label="Total energy")
plt.plot([kinetic_energy(particles) for _ in range(T)], label="Kinetic energy")
plt.plot(potential_energies, label="Potential energy")
plt.title("Total, kinetic and potential energy over time")
plt.xlabel("Step")
plt.ylabel("Energy")
plt.legend()
plt.savefig(os.path.join(RESULTS_DIR, "energies.png"))
plt.close()