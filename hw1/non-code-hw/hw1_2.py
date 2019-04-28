#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 23:23:20 2019

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

y = np.array([[
        -1,
        -1,
        -1,
        1,
        1,
        1,
        1
        ]])

def kernel_function(x1,x2):
    return (1+np.dot(x1.T , x2))**2

'''
c=['green' if label ==1 else 'red' for label in y[0]]
plt.scatter(x=z[:,0],y=z[:,1],color=c)
plt.show
'''

#ones=np.array([[1,1,1,1,1,1]])
#M = np.array([[0., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
M= np.zeros((7,7))
for n in range(7):
    for m in range(7):
        #M[n,m]=y[0][n]*y[0][m]*np.dot(z[n].T , z[m])
        M[n,m]=y[0][n]*y[0][m]*kernel_function(x[n],x[m])
#P = np.dot(M.T, M)  # quick way to build a symmetric matrix
P=M
print("P",P)
q = -np.ones((7,))
print("q",q)
G = -np.eye(7)#some question
print("G",G)
h = np.zeros((7,))
print("h",h)
A = y
b = 0
print("A",A)
print("b",b)


alpha=qpsolvers.solve_qp(P, q, G, h)
print (alpha)
#[-5.00000000e+00  1.00000000e+00 -5.55111512e-17]

#sv=[np.where(alpha==alphan) if alphan > 0 else -1 for alphan in alpha]
sv=[1,2,3,4,5]
print(sv)
b=0
for s in sv:
    #print(s)
    sum=0
    for n in range(7):
        #print("an",alpha[n])
        sum+=alpha[n]*y[0][n]*kernel_function(x[n],x[s])
    print(y[0][s]-sum)#[-1,-1,1,1,1]
    b+=(y[0][s]-sum)
    #print("b",b)
b/=len(sv)#求出每个support vector的b，然后平均
print(b)

'''
x=[x1,x2]
sum=0
for n in sv:
    sum+=alpha[n]*y[0][n]*kernel_function(x[n],x)
print(sum)
'''
c=[0 for _ in range(7)]
for n in sv:
    c[n]=alpha[n]*y[0][n]
print(c)
print("constant",c[1]+c[2]+c[3]+c[4]+c[5])#parameters for hyperplane
print("x22",c[1]+c[2]+4*c[4]+4*c[5])#parameters for hyperplane
print("x2",2*c[1]-2*c[2]+4*c[4]-4*c[5])#parameters for hyperplane
print("x12",c[3])#parameters for hyperplane
print("x1",-2*c[3])#parameters for hyperplane
'''
constant -1.664071665933299
x22 0.6658838463380059
x2 -1.1368683772161603e-13
x12 0.8877279793178862
x1 -1.7754559586357723

'''
