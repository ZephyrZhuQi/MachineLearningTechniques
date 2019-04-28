from sklearn import svm
from sys import argv
import matplotlib.pyplot as plt
import numpy as np
#argv[1]:train data | argv[2]:problem number

def rbf(x1,x2):
    return np.exp(-80*np.sum((x1-x2)**2))

def read_data(file):
    X = []
    y = []
    with open(file,"r") as f:
        for line in f:
            data = line.split()
            y.append(float(data[0]))#digit precision?
            X.append([float(x) for x in data[1:]])
    X = np.array(X)
    y = np.array(y)
    return X, y

def multi2binary(y, label):
    y_new = []
    for y_elem in y:
        y_new.append(1 if y_elem == label else -1)
    y_new = np.array(y_new)
    return y_new

def plot13(X, y):
    C = [x for x in range(-5,5,2)]
    abs_w = []
    for i in range(5):
        clf = svm.SVC(kernel='linear',C=10**C[i])
        clf.fit(X,y)
        abs_w.append(np.sqrt(np.sum(clf.coef_**2)))
    fig = plt.figure()
    plt.plot(C,abs_w,'ro')
    plt.plot(C,abs_w,'b')
    plt.title('Problem13')
    plt.xlabel('log10C')
    plt.ylabel('||w||')
    for x,y in zip(C,abs_w):
        plt.text(x,y+0.003,str(round(y,4)))
    plt.show()
    fig.savefig('Problem13.png')

def plot14(X, y):
    C = [x for x in range(-5,5,2)]
    Ein = []
    for i in range(5):
        clf = svm.SVC(kernel='poly',C=10**C[i],degree=2,gamma=1,coef0=1)
        clf.fit(X,y)
        Ein.append(1.0 - clf.score(X,y))
    fig = plt.figure()
    plt.plot(C,Ein,'ro')
    plt.plot(C,Ein,'b')
    plt.title('Problem14')
    plt.xlabel('log10C')
    plt.ylabel('Ein')
    for x,y in zip(C,Ein):
        plt.text(x,y+0.001,str(round(y,4)))
    plt.show()
    fig.savefig('Problem14.png')

def plot15(X, y):
    C = [x for x in range(-2,3,1)]
    distance = []
    for i in range(5):
        clf = svm.SVC(kernel='rbf',C=10**C[i],gamma=80)
        clf.fit(X,y)
        w_square = 0.0
        alpha = np.abs(clf.dual_coef_.reshape(-1))
        X_SV = clf.support_vectors_
        y_SV = y[clf.support_]
        for i in range(len(alpha)):
            for j in range(len(alpha)):
                w_square = w_square + alpha[i]*alpha[j]*y_SV[i]*y_SV[j]*rbf(X_SV[i],X_SV[j])
        dis = 1.0/np.sqrt(w_square)
        distance.append(dis)
    fig = plt.figure()
    plt.plot(C,distance,'ro')
    plt.plot(C,distance,'b')
    plt.title('Problem15')
    plt.xlabel('log10C')
    plt.ylabel('distance')
    for x,y in zip(C,distance):
        plt.text(x,y+0.05,str(round(y,3)))
    plt.show()
    fig.savefig('Problem15.png')

def plot16(X, y):
    gamma = [x for x in range(-2,3,1)]
    gamma_times = [0 for x in range(5)]
    for i in range(100):
        index = np.arange(X.shape[0])
        np.random.seed(i)
        np.random.shuffle(index)
        X_val = X[index[:1000]]
        y_val = y[index[:1000]]
        X_train = X[index[1000:]]
        y_train = y[index[1000:]]
        best = 0.0
        for j in range(4,-1,-1):
            clf = svm.SVC(kernel='rbf',C=0.1,gamma=10**gamma[j])
            clf.fit(X_train, y_train)
            perform = clf.score(X_val, y_val)
            if perform > best:
                best = perform
                choice = j
        gamma_times[choice] = gamma_times[choice]+1
    fig = plt.figure()
    plt.bar(gamma,gamma_times)
    plt.title('Problem16')
    plt.xlabel('log10gamma')
    plt.ylabel('times')
    for x,y in zip(gamma,gamma_times):
        plt.text(x,y+1,str(y))
    plt.show()
    fig.savefig('Problem16.png')

def main():
    X_train, y_train = read_data(argv[1])
    #X_test, y_test = read_data(argv[2])
    problem = int(argv[2])
    if problem == 13:
        y = multi2binary(y_train, 2)
        plot13(X_train, y)
    elif problem == 14:
        y = multi2binary(y_train, 4)
        plot14(X_train, y)
    elif problem == 15:
        y = multi2binary(y_train, 0)
        plot15(X_train, y)
    elif problem == 16:
        y = multi2binary(y_train, 0)
        plot16(X_train, y)

if __name__ == '__main__':
    main()

