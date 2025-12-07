from scipy import linspace
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np 
#f = pojedyńczy: 0.21 podwójny: 0.31, 0.33, chaos:
omega = 2* 0.213 * np.pi
f = 0.4

def constant(x):
    return np.full(x.shape, 1)

def equation1(t, xvz):
    x, v, z = xvz
    a, b, c,  = 1, 1, 2/10, #parametry równania 
    return [v, b*x - a*x**3 - c*v  + f * np.cos(omega * t ), omega]

def equation2(t, xvz):
    x, v, z = xvz
    a, b, c,  = 1, 1, 2/10, #parametry równania 
    return [v, b*x - a*x**3 - c*v,  omega]

a, b = 0, 500
t = np.array(linspace(a, b, 4000))

sol1 = solve_ivp(equation1, [a, b], [1,0.15,omega], t_eval=t)
sol2 = solve_ivp(equation2, [a, b], [0,0.15,omega], t_eval=t)
sol3 = solve_ivp(equation2, [a, b], [-0.1363,0.15,omega], t_eval=t)

plt.plot(sol1.y[0], sol1.y[1], color = "b",label = "rozwiązanie dla f")
plt.plot(sol2.y[0], sol2.y[1], color = "r", label = "położenie równowagi nr.1")
plt.plot(sol3.y[0], sol3.y[1], color = "g", label = "położenie równowagi nr.2")
plt.xlabel("$x$")
plt.ylabel("$\dot{x}$")
plt.title("f=" + str(f))
plt.legend()
plt.savefig("lorenz_xz.png")
plt.close()

plt.subplot(211)
plt.plot(sol1.t, sol1.y[0], color ="b")
plt.plot(sol1.t, constant(sol1.t), color = "r")
plt.xlabel("$t$")
plt.ylabel("$x(t)$")
plt.subplot(212)
plt.plot(sol1.t, sol1.y[1])
plt.ylabel("$v(t)$")
plt.xlabel("$t$")
plt.savefig("lorenz_x.png")
plt.close()