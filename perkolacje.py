import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from collections import deque

L = 20
L_pol = int(L/2)
lattice = np.ones((L,L))*(-1)
p = 0.5
startowy =0
tossing = np.random.random((L,L)) < p
cmap = colors.ListedColormap(['Blue','red', 'green'])
cluster = deque() # tworzymy kolejkę
cluster.append([1,0])
neis = [(-1,0),(1,0),(0,-1),(0,1)]
for i in range(L):
    lattice[0][i] = -2
    lattice [L-1][i] = -2

def color (i,j):
    if lattice[i][j] == -2:
        return 'border'
    
    if tossing[i][j] == True:
        lattice[i][j] = 1 
        return True
    
    if tossing[i][j] == False:
        lattice[i][j] = 0
        return False

def PBC(i,j):
    if i == L:
        i = 0
    if i ==0:
        i = 0
    if i == -1:
        i = L -1
    if i == L-1:
        i = 0
    return i,j 

def cluster_expansion(i,j):

    if color(i,j) == True:
        for nei in neis:
            i = i + nei[0]
            j = j + nei[1]
            i,j = PBC(i,j)
            cluster.append([i,j])
            print(cluster)
        
        


while len(cluster)>0:
    i,j = cluster[0]
    print('siusiak')
    cluster_expansion(i,j)
    print(cluster)
    cluster.popleft()

plt.figure(figsize=(6,6))
plt.pcolor(lattice[::-1],cmap=cmap,edgecolors='k', linewidths=3)
plt.show()