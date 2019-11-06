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
            if i < 400: # use the first 400 examples for training
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
    num_train = X_train.shape[0]
    num_test = X_test.shape[0]
    lambdas = [0.05, 0.5, 5, 50, 500]
    Eins = []
    Eouts = []
    for lambd in lambdas:
        Ein = 0
        Eout = 0
        y_train_real = np.zeros(num_train)
        y_predict_real = np.zeros(num_test)
        for iteration in range(250): # 250 iterations of bagging (250 gt's)
            #bootstrapping using uniform distribution
            idxs = np.random.randint(0, num_train, size = num_train)
            X_train_t = []# 400 bootstrapped examples
            y_train_t = []
            for i in idxs:
                X_train_t.append(X_train[i])
                y_train_t.append(y_train[i])
            X_train_t = np.array(X_train_t)
            y_train_t = np.array(y_train_t)
            left = np.linalg.inv(np.dot(X_train_t.transpose(), X_train_t) + lambd * np.eye(num_feature))
            right = np.dot(X_train_t.transpose(), y_train_t)
            #ridge regression for one gt
            WREG = np.dot(left, right) 
            #regression of gt
            y_train_real_t = np.dot(X_train, WREG)
            #classification of gt
            y_train_sign_t = np.where(y_train_real_t>0, +1, -1)
            #uniform blending for train set
            y_train_real = y_train_real + y_train_sign_t
            #regression of gt for test set 
            y_predict_real_t = np.dot(X_test, WREG) 
            #classification of gt for test set
            y_predict_sign_t = np.where(y_predict_real_t>0, +1, -1)
            #uniform blending for test set
            y_predict_real = y_predict_real + y_predict_sign_t
        # take the sign operation before uniform aggregation
        y_train_sign = np.where(y_train_real>0, +1, -1)
        y_predict_sign = np.where(y_predict_real>0, +1, -1)
        for i in range(num_train):
            if y_train_sign[i] != y_train[i]:
                Ein = Ein + 1
        Ein = float(Ein)/float(num_train)
        Eins.append(Ein)
        for i in range(num_test):
            if y_predict_sign[i] != y_test[i]:
                Eout = Eout + 1
        Eout = float(Eout)/float(num_test)
        Eouts.append(Eout)
    fig = plt.figure()
    plt.plot(lambdas, Eins, 'b')
    plt.plot(lambdas, Eins, 'ro')
    plt.title('Bagging Ein with respect to lambda')
    plt.xlabel('lambda')
    plt.ylabel('Bag Ein')
    for x,y in zip(lambdas, Eins):
        plt.text(x, y, str(y))
    plt.show()
    fig.savefig('Bag_Ein.png')
    fig = plt.figure()
    plt.plot(lambdas, Eouts, 'b')
    plt.plot(lambdas, Eouts, 'ro')
    plt.title('Bagging Eout with respect to lambda')
    plt.xlabel('lambda')
    plt.ylabel('Bag Eout')
    for x,y in zip(lambdas, Eouts):
        plt.text(x, y, str(y))
    plt.show()
    fig.savefig('Bag_Eout.png')
    print(Eins)
    print(Eouts)

if __name__ == '__main__':
    main()
