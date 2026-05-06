#Diffusion-Limited Aggregation simulation that shows how do crystals and other structures form
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

L = 200
midpoint = int(L/2)
map  = np.zeros([L,L])
map[midpoint][midpoint] = 1 
R = 5
vec = np.array([[1,0],[0,1], [0,-1],[-1,0]])
Rmax =  R + 30 
STICK_PROB = 1
M = 10 

def r_calculate(point):
   x = point - midpoint
   return np.linalg.norm(x)

def new_R(point, R):
    r_count = r_calculate(point)
    if r_count < R:
        return R
    else:
        return r_count + 10
      

def end_point(point, R):
    r = r_calculate(point)
    local_Rmax = R + 30

    if r <= R:
        for a in vec:
            somsiad = (point + a)
            if map[somsiad[0]][somsiad[1]] == 1:
                if np.random.rand() < STICK_PROB:
                    map[point[0]][point[1]] = 1
                    R = new_R(point, R)
                    return 1, R

    if r > local_Rmax:
        return 2, R

    return 0, R
def is_surrounded(point):
    b = 0
    temp_point = point 
    for a in vec:
        point = point + a
        if map[point[0]][point[1]] == 1:
            b +=1
        point = temp_point
    if b ==4:
        return True
    else:
        return False

def random_movement(point):
    temp_point = point 
    a = 0
    while(True): 
        x = np.random.randint(0, 4)
        point = point + vec[x]
        if map[point[0]][point[1]] !=1:
            break
        point = temp_point
        if is_surrounded(point) == True:
            break
        

    return point

def point_init(R):
    theta  = np.random.rand()* 2 * np.pi        
    x = int((R) * np.cos(theta)) + midpoint
    y = int((R) * np.sin(theta)) + midpoint

    return np.array([x,y])

def point_life(point, R):

    while True:
        status, R = end_point(point, R)
        if status == 1 or status == 2:
            break
        point = random_movement(point)

    return R, status

a = 0
im_collection = [map.copy()]
x = len(im_collection)

while(True):  
    start_point = point_init(R)    
    R, status = point_life(start_point, R)

    if status == 1:
        a = a + 1 
        if a%50 == 0 and a != 0:
            im_collection.append(map.copy())
        if a == 2000:
            break
        
fig, ax = plt.subplots()
img = ax.imshow(im_collection[0], cmap='gray', vmin=0, vmax=1)
ax.axis('off') 

def update(frame_index):
    img.set_data(im_collection[frame_index])
    return (img,)

anim = FuncAnimation(fig, update, frames=len(im_collection),interval=100, blit=True )

anim.save('anim1.gif', writer='pillow', fps=10)
print("Zapisano animację jako anim.gif")