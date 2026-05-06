#Simulation of Wa-tor planet, sped up with numba library and unfinished fit of Lotka-Voltera Model 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numba import njit
from scipy.integrate import odeint
from scipy.optimize import least_squares


# parametry
L = 200
FISH = 1
SHARK = 2
EMPTY = 0
start = 300
A = 3   # czas rozmnażania ryb
B = 20  # czas rozmnażania rekinów
E = 3   # energia startowa rekina

INITIAL_FISH = 1200
INITIAL_SHARKS = 300
T = 1200

# kierunki sąsiedztwa
vecs = np.array([[0,1],[1,0],[-1,0],[0,-1]])

# tablice
Space = np.zeros((L,L), dtype=np.int32)
breed = np.zeros((L,L), dtype=np.int32)
energy = np.zeros((L,L), dtype=np.int32)

counts_fish = []
counts_sharks = []
time_steps = []

def lotka_volterra(X, t, alpha, beta, delta, gamma):
    x, y = X
    dxdt = alpha*x - beta*x*y
    dydt = delta*x*y - gamma*y
    return [dxdt, dydt]

def simulate(params, t, x0, y0):
    alpha, beta, delta, gamma = params
    sol = odeint(lotka_volterra, [x0, y0], t, args=(alpha, beta, delta, gamma))
    return sol

def weighted_residuals(params, t, data):
    sim = simulate(params, t, data[0,0], data[0,1])
    weights = 1.0 / (np.abs(data) + 10)
    return ((sim - data) * weights).ravel()

# periodic boundary
@njit
def PBC(x):
    return x % L

@njit
def init_world(Space, breed, energy):
    count = 0
    while count < INITIAL_FISH:
        i = np.random.randint(L)
        j = np.random.randint(L)
        if Space[i,j] == EMPTY:
            Space[i,j] = FISH
            breed[i,j] = A
            count += 1

    count = 0
    while count < INITIAL_SHARKS:
        i = np.random.randint(L)
        j = np.random.randint(L)
        if Space[i,j] == EMPTY:
            Space[i,j] = SHARK
            breed[i,j] = B
            energy[i,j] = E
            count += 1

@njit
def step(Space, breed, energy):
    moved = np.zeros_like(Space)
    for i in range(L):
        for j in range(L):
            organism = Space[i,j]
            if organism == EMPTY or moved[i,j]:
                continue

            order = np.random.permutation(4)
            ni, nj = i, j

            if organism == FISH:
                for k in order:
                    di, dj = vecs[k]
                    ti = PBC(i+di)
                    tj = PBC(j+dj)
                    if Space[ti,tj] == EMPTY:
                        ni, nj = ti, tj
                        break

                if (ni, nj) != (i, j):
                    if breed[i,j] <= 0:
                        Space[ni,nj] = FISH
                        breed[ni,nj] = A
                        breed[i,j] = A
                    else:
                        Space[ni,nj] = FISH
                        breed[ni,nj] = breed[i,j] - 1
                        Space[i,j] = EMPTY
                        breed[i,j] = 0
                    moved[ni,nj] = 1

            elif organism == SHARK:
                ate_fish = False
                for k in order:
                    di, dj = vecs[k]
                    ti = PBC(i+di)
                    tj = PBC(j+dj)
                    if Space[ti,tj] == FISH:
                        ni, nj = ti, tj
                        ate_fish = True
                        break

                if not ate_fish:
                    for k in order:
                        di, dj = vecs[k]
                        ti = PBC(i+di)
                        tj = PBC(j+dj)
                        if Space[ti,tj] == EMPTY:
                            ni, nj = ti, tj
                            break

                if (ni, nj) != (i, j):
                    if ate_fish:
                        energy[i,j] += E
                    if breed[i,j] <= 0:
                        Space[ni,nj] = SHARK
                        breed[ni,nj] = B
                        energy[ni,nj] = energy[i,j]
                        breed[i,j] = B
                        energy[i,j] = E
                    else:
                        Space[ni,nj] = SHARK
                        breed[ni,nj] = breed[i,j] - 1
                        energy[ni,nj] = energy[i,j] - 1
                        Space[i,j] = EMPTY
                        breed[i,j] = 0
                        energy[i,j] = 0
                    moved[ni,nj] = 1
                else:
                    energy[i,j] -= 1

                if energy[ni,nj] <= 0:
                    Space[ni,nj] = EMPTY
                    breed[ni,nj] = 0
                    energy[ni,nj] = 0

# inicjalizacja świata
init_world(Space, breed, energy)

# przygotowanie grafiki
fig, ax = plt.subplots()
img = ax.imshow(Space, cmap='viridis', vmin=0, vmax=2)
ax.axis('off')
cbar = fig.colorbar(img, ax=ax)
cbar.set_ticks([0,1,2])
cbar.set_ticklabels(['Pusty','Ryba','Rekin'])

def update(frame):
    step(Space, breed, energy)
    fish_count = np.count_nonzero(Space == FISH)
    shark_count = np.count_nonzero(Space == SHARK)

    if frame >start:    
        counts_fish.append(fish_count)
        counts_sharks.append(shark_count)
        time_steps.append(frame)

    img.set_data(Space)
    ax.set_title(f"Krok: {frame}, Ryby: {fish_count}, Rekiny: {shark_count}")
    return img,

anim = FuncAnimation(fig, update, frames=T, interval=1)
anim.save('wator_improved.gif', writer='pillow', fps=10)


data = np.column_stack((counts_fish, counts_sharks))
t = np.array(time_steps)

# Początkowe parametry (dostosowane do skali populacji)
initial_guess = [0.5, 0.001, 0.0001, 0.3]

# Użycie ważonych residuals dla lepszego dopasowania
result = least_squares(weighted_residuals, initial_guess, args=(t, data), 
                      bounds=([0, 0, 0, 0], [10, 1, 1, 10]))

alpha, beta, delta, gamma = result.x

print(f"Dopasowane parametry Lotki-Volterry:")
print(f"α (wzrost ryb) = {alpha:.6f}")
print(f"β (drapieżnictwo) = {beta:.6f}")
print(f"δ (konwersja) = {delta:.6f}")
print(f"γ (śmiertelność rekinów) = {gamma:.6f}")

t_dense = np.linspace(min(time_steps), max(time_steps), 500)

x0 = counts_fish[0]
y0 = counts_sharks[0]

lv_solution = odeint(lotka_volterra, [x0, y0], t_dense, args=(alpha, beta, delta, gamma))
lv_fish = lv_solution[:,0]
lv_sharks = lv_solution[:,1]

plt.figure()
plt.scatter(time_steps, counts_fish, color = 'blue')
plt.plot(t_dense, lv_fish, color='red', linestyle='-', label='Ryby (Lotka-Volterra)')
plt.xlabel("Krok czasu")
plt.ylabel("Liczba organizmów")
plt.legend()
plt.title("Dynamika populacji ryb")
plt.savefig("Populacja ryb")


plt.figure()
plt.scatter(time_steps, counts_sharks, color = 'blue')
plt.plot(t_dense, lv_sharks, color='red', linestyle='-', label='Rekiny (Lotka-Volterra)')
plt.xlabel("Krok czasu")
plt.ylabel("Liczba organizmów")
plt.legend()
plt.title("Dynamika populacji rekinów")
plt.savefig("Populacja rekinów")

plt.figure()
plt.scatter(counts_sharks[::5], counts_fish[::5])
plt.xlabel("Liczba rekinów")
plt.ylabel("Liczba ryb")
plt.title("Dynamika populacji ryb i rekinów")
plt.savefig("Wykres fazowy")

