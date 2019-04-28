from sys import argv
import numpy as np
import matplotlib.pyplot as plt

def read_file(file):
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    i = 0
    with open(file, "r") as f:
        for line in f:
            data = line.split()
            data = ['1'] + data
            if i < 400:
                X_train.append([float(x) for x in data[:-1]])
                y_train.append(int(data[-1]))
            else:
                X_test.append([float(x) for x in data[:-1]])
                y_test.append(int(data[-1]))
            i = i + 1
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    return X_train, y_train, X_test, y_test



def main():
    X_train, y_train, X_test, y_test = read_file(argv[1])
    num_feature = X_train.shape[1]
    lambdas = [0.05, 0.5, 5, 50, 500]
    Eins = []
    Eouts = []
    for lambd in lambdas:
        Ein = 0
        Eout = 0
        left = np.linalg.inv(np.dot(X_train.transpose(), X_train) + lambd * np.eye(num_feature))
        right = np.dot(X_train.transpose(), y_train)
        WREG = np.dot(left, right) #ridge regression
        y_train_real = np.dot(X_train, WREG)
        y_train_sign = np.where(y_train_real>0, +1, -1)
        num_train = X_train.shape[0]
        num_test = X_test.shape[0]
        for i in range(num_train):
            if y_train_sign[i] != y_train[i]:
                Ein = Ein + 1
        Ein = float(Ein)/float(num_train)
        Eins.append(Ein)
        y_predict_real = np.dot(X_test, WREG) 
        y_predict_sign = np.where(y_predict_real>0, +1, -1)
        for i in range(num_test):
            if y_predict_sign[i] != y_test[i]:
                Eout = Eout + 1
        Eout = float(Eout)/float(num_test)
        Eouts.append(Eout)
    fig = plt.figure()
    plt.plot(lambdas, Eins, 'b')
    plt.plot(lambdas, Eins, 'ro')
    plt.title('Ein with respect to lambda')
    plt.xlabel('lambda')
    plt.ylabel('Ein')
    for x,y in zip(lambdas, Eins):
        plt.text(x, y, str(y))
    plt.show()
    fig.savefig('Ein.png')
    fig = plt.figure()
    plt.plot(lambdas, Eouts, 'b')
    plt.plot(lambdas, Eouts, 'ro')
    plt.title('Eout with respect to lambda')
    plt.xlabel('lambda')
    plt.ylabel('Eout')
    for x,y in zip(lambdas, Eouts):
        plt.text(x, y, str(y))
    plt.show()
    fig.savefig('Eout.png')
    print(Eins)
    print(Eouts)

if __name__ == '__main__':
    main()
