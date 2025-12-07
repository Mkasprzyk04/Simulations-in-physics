import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys 

N = int(sys.argv[1])
Matrix = np.random.rand(N,N)

fig, ax = plt.subplots()
im = ax.matshow(Matrix)


def animate(i):
    for i in range(N+1):
        Matrix[[i-1, i]] = Matrix[[i, i-1]]
        im.set_array(Matrix)

    return im

animation = FuncAnimation(fig, animate, frames = N, interval = 100)

plt.show()
