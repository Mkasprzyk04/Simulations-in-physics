import numpy as np
import matplotlib.pyplot as plt
import imageio 

L = 500
midpoint = int(L/2)
map  = np.zeros([L,L])
map[midpoint][midpoint] = 1 
R = 5
vec = np.array([[1,0],[0,1], [0,-1],[-1,0]])
Rmax =  R + 30 

def r_calculate(point):
   x = point - midpoint
   return np.linalg.norm(x)

def new_R(point, R):
    r_count = r_calculate(point)
    if r_count < R:
        return R
    else:
        # increase R when a particle attaches farther out
        return r_count + 10
      

def end_point(point, R):
    r = r_calculate(point)
    # use a local Rmax based on current R
    local_Rmax = R + 30

    if r <= R:
        for a in vec:
            somsiad = (point + a)
            if map[somsiad[0]][somsiad[1]] == 1:
                map[point[0]][point[1]] = 1
                R = new_R(point, R)
                return 1, R

    if r > local_Rmax:
        map[point[0]][point[1]] = 0
        return 2, R

    return 0, R

def random_movement(point):
    c  = np.random.rand()
    neighbour = np.array([])
    if c <= 0.25:
       neighbour = point + vec[0]
    elif  0.25 < c <= 0.5:
       neighbour = point + vec[1]   
    elif  0.5 < c <= 0.75:
       neighbour = point + vec[2]   
    elif  0.75 < c <= 1:
       neighbour = point + vec[3]  

    return neighbour

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
im_collection = []
while(True):  
    start_point = point_init(R)    
    R, status = point_life(start_point, R)

    if status == 1:
        a = a + 1 
    if a%50 == 0 and a != 0 :
        im_collection.append(map)


    if a == 1000:
        break

imgs = [(frame * 255).astype(np.uint8) for frame in im_collection]
imageio.mimsave("anim.gif", imgs, duration=0.7)  # 0.2s na klatkę
print("GIF zapisany jako anim.gif")