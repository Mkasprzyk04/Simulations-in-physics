#Lista zaimportowanych bibliotek    
import qutip as qt
from scipy.sparse import diags
import numpy as np

def run_simulation(gamma,omega, T_periods):
    # Parametry użyte do stworzenia siatki i rozwiązywania równań różniczkowych
    L = 5.0           # granica x
    N = 400            # liczba punktów
    x = np.linspace(-L, L, N)
    dx = x[1] - x[0]

    # Tworzymy operator drugiej pochodnej(Laplasjan) z warunkami Dirichleta, zapewniającymi brak ewolucji czasowej na granicach funkcji falowej
    lap = diags(
    [1, -2, 1], offsets=[-1, 0, 1], shape=(N, N)
    ).toarray() / dx**2
#Wyżej wspomniane warunki Dirichleta
    lap[0, :] = 0
    lap[0, 0] = 1
    lap[-1, :] = 0
    lap[-1, -1] = 1
# Konwersja do obiektu przetwarzanego przez qutip, czyli używaną przezemnie bibliotekę numeryczną
    lap = qt.Qobj(lap)
    T = - 1 / (2.0 * 1.0) * lap  # przy m=1, ħ=1

# Operator odpowiadający x^3
    X3 = qt.Qobj(diags(x**3, 0))

# Stała część potencjału
    V0 = -0.5 * x**2 + 0.0625 * x**4
    V0_op = qt.Qobj(diags(V0, 0))

# Część czasowo zależna w formacie zalecanym przez qutip
    H_td = [ X3, "gamma * cos(omega * t)" ]
    H = [ T + V0_op, H_td ]

# Definicja unormowanego stanu początkowego z dowolnością w wyborze parametrów
    sigma = 1
    x0 = 0
    psi0 =  np.exp(-(x - x0 )**2/(2*sigma**2 ))
    psi0[-int(L)] = 0
    psi0[int(L)] = 0
    psi0 = qt.Qobj(psi0)
    psi0 = psi0 / psi0.norm()

#Przygotowanie listy odpowiadającej czasowej ewolucji i użytej w animacji
    t = T_periods * 2 * np.pi / omega
    tlist = np.linspace(0, t, 100)

# utworzymy tablicę V o wymiarach odpowiadających wymiarowi ewolucji czasowej, jest to swego rodzaju "zdjęcie" potencjału dla danej przestrzeni w zadanym czasie.
    x3 = x**3
    V = np.zeros((len(tlist), N))
    for i, tval in enumerate(tlist):
        V[i] = V0 + gamma * x3 * np.cos(omega * tval)

#Definiujemy sztucznie maksymalną liczbę iteracji dla każdego skoku czasowego oraz rozwiązujemy równanie różniczkowe przy pomocy wbudowanej w qutip funkcji
    opts = qt.Options(nsteps = 1e7) 
    res = qt.mesolve(H, psi0, tlist, c_ops=[], e_ops=[], args={'gamma': gamma, 'omega': omega}, options=opts)
    
    psi2 = np.array([np.abs(state.full().flatten())**2 for state in res.states])
    return x, psi2, V

