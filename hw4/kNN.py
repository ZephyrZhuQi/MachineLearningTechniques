import numpy as np
import matplotlib.pyplot as plt

def read_data(file):
    X, Y = [], []
    with open(file, 'r') as f:
        for line in f:
            data = line.split()
            X.append([float(x) for x in data[:-1]])
            Y.append(int(data[-1]))
    X = np.array(X)
    Y = np.array(Y)
    return X, Y


def Euclidean_D(x1, x2):
    return np.sqrt(np.sum(np.power((x1-x2), 2)))


def main():
    #k = 1
    ks = [1, 3, 5, 7, 9]  # the number of nearest neighbors
    #ks = [1, 3, 5]
    Eins, Eouts = [], []
    X_train, Y_train = read_data('./data/hw4_train.dat')
    X_test, Y_test = read_data('./data/hw4_test.dat')
    n_train = Y_train.shape[0]
    print('n_train', n_train)
    n_test = Y_test.shape[0]
    print('n_test', n_test)
    for k in ks:
        Eout = 0
        for i in range(n_test):
            distance = []
            for j in range(n_train):
                distance.append(Euclidean_D(X_test[i], X_train[j]))
            knn_index = np.argsort(distance)[:k]
            #print(distance)
            #print('i',knn_index)
            predict = np.sign(np.mean(Y_train[knn_index]))
            #print('predict', predict)
            if predict != Y_test[i]:
                Eout += 1
        Eout /= n_test
        Eouts.append(Eout)
        print('Eout', Eout)
        Ein = 0
        for i in range(n_train):
            distance = []
            for j in range(n_train):
                distance.append(Euclidean_D(X_train[i], X_train[j]))
            knn_index = np.argsort(distance)[:k]
            #print(distance)
            #print('i',knn_index)
            predict = np.sign(np.mean(Y_train[knn_index]))
            #print('predict', predict)
            if predict != Y_train[i]:
                Ein += 1
        Ein /= n_train
        Eins.append(Ein)
        print('Ein', Ein)
    # plotting
    fig = plt.figure()
    plt.plot(ks, Eins, 'b')
    plt.plot(ks, Eins, 'ro')
    plt.title('Ein with respect to k')
    plt.xlabel('k')
    plt.ylabel('Ein')
    for x,y in zip(ks, Eins):
        plt.text(x, y, str(y))
    plt.show()
    fig.savefig('Ein.png')
    fig = plt.figure()
    plt.plot(ks, Eouts, 'b')
    plt.plot(ks, Eouts, 'ro')
    plt.title('Eout with respect to k')
    plt.xlabel('k')
    plt.ylabel('Eout')
    for x,y in zip(ks, Eouts):
        plt.text(x, y, str(y))
    plt.show()
    fig.savefig('Eout.png')
    print(Eins)
    print(Eouts)

if __name__ == '__main__':
    main()
