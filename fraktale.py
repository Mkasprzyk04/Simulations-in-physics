import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.optimize import curve_fit 
from scipy.stats import chisquare

def linear(x,a,b):
    return a*x + b


wekt0 = np.array([0,0])
m = np.array([0.5, 0, 0, 0.5, 0.25, np.sqrt(3.)/4])
n = np.array([0.5, 0, 0, 0.5, 0.0, 0])
p = np.array([0.5, 0, 0, 0.5, 0.5, 0])
states_list = np.array([m,n,p])
probability = np.array([1/3,1/3,1/3])

r = random.choices(states_list, weights = probability, k = 7000)

X = np.array([0])
Y = np.array([0])
for weights in r:    
    wekt0 = np.array([(wekt0[0] * weights[0] + wekt0[1] * weights[1] + weights[4]), (weights[2] * wekt0[0] + weights[3] * wekt0[1] + weights[5])])
    X = np.append(X,wekt0[0])
    Y = np.append(Y, wekt0[1])

a = 0
A = np.array([])
R = np.array([])
err = np.array([])
for r in range(13):
    histogram = np.histogram2d(X,Y, 2**r)
    R = np.append(R, r)
    err = np.append(err, 0.3)
    for i in range(len(histogram[0][0])):
        for j in range(len(histogram[0][0])):
            if histogram[0][i][j] !=0 :
                a = a+1        
    A = np.append(A,np.log(a))
    a = 0


for i in range(len(R)):
    A = A[:-1]
    err = err[:-1]
    R = R[:-1]
    plot_info = curve_fit(linear,R,A, sigma = err)
    a,b = plot_info[0][0], plot_info[0][1]
    fitted_data = linear(R,a,b)
    chi2 = chisquare(A,fitted_data)[1]    
    if chi2 < 0.5:
        break

plt.plot(R,fitted_data)

plt.errorbar(R,A, yerr = err, linestyle = "none")
plt.show()

