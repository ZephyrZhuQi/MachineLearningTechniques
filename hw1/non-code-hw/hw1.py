#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 16:13:01 2019

@author: z
"""
import numpy as np
import matplotlib.pyplot as plt
import qpsolvers 

x = np.array([
        [1, 0],
        [0, 1],
        [0, -1],
        [-1, 0],
        [0, 2],
        [0, -2],
        [-2, 0]
        ])

z = np.zeros((7,2))

y = np.array([[
        -1,
        -1,
        -1,
        1,
        1,
        1
        ]])

for i in range(7):
    x1=x[i,0]
    x2=x[i,1]
    z[i,0]=2*x2**2 - 4*x1 + 2
    z[i,1]=x1**2 - 2*x2 - 3
    
#print(z)

z=np.delete(z,6,0)
#print(z)
c=['green' if label ==1 else 'red' for label in y[0]]
plt.scatter(x=z[:,0],y=z[:,1],color=c)
plt.show

ones=np.array([[1,1,1,1,1,1]])
M = np.array([[0., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
P = np.dot(M.T, M)  # quick way to build a symmetric matrix
q = np.array([[0.,0.,0.]]).reshape((3,))
G = -1*y.T*np.insert(z,0,values=ones,axis=1)
h = -1*np.ones((6,1)).reshape((6,))
print("M",M)
print("P",P)
print("q",q)
print("G",G)
print("h",h)


print (qpsolvers.solve_qp(P, q, G, h))
#[-5.00000000e+00  1.00000000e+00 -5.55111512e-17]



