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

def exp_d(x1,x2,gamma):
    return np.exp(-gamma*np.power(np.linalg.norm(x1-x2),2))

def main():
    gammas = [0.001, 0.1, 1, 10, 100]
    #ks = [1, 3, 5]
    Eins, Eouts = [], []
    X_train, Y_train = read_data('./data/hw4_train.dat')
    X_test, Y_test = read_data('./data/hw4_test.dat')
    n_train = Y_train.shape[0]
    print('n_train', n_train)
    n_test = Y_test.shape[0]
    print('n_test', n_test)
    for gamma in gammas:
        Eout = 0
        for i in range(n_test):
            score = 0
            for m in range(n_train):
                score += Y_train[m]*exp_d(X_test[i],X_train[m],gamma)
            predict = np.sign(score)
            print('predict', predict)
            if predict != Y_test[i]:
                Eout += 1
        Eout /= n_test
        Eouts.append(Eout)
        print('Eout', Eout)
        Ein = 0
        for i in range(n_train):
            score = 0
            for m in range(n_train):
                score += Y_train[m]*exp_d(X_train[i],X_train[m],gamma)
            predict = np.sign(score)
            print('predict', predict)
            if predict != Y_train[i]:
                Ein += 1
        Ein /= n_train
        Eins.append(Ein)
        print('Ein', Ein)
    # plotting
    fig = plt.figure()
    plt.plot(gammas, Eins, 'b')
    plt.plot(gammas, Eins, 'ro')
    plt.title('Ein with respect to gamma')
    plt.xlabel('gamma')
    plt.ylabel('Ein')
    for x,y in zip(gammas, Eins):
        plt.text(x, y, str(y))
    plt.show()
    fig.savefig('Ein_uniform.png')
    fig = plt.figure()
    plt.plot(gammas, Eouts, 'b')
    plt.plot(gammas, Eouts, 'ro')
    plt.title('Eout with respect to gamma')
    plt.xlabel('gamma')
    plt.ylabel('Eout')
    for x,y in zip(gammas, Eouts):
        plt.text(x, y, str(y))
    plt.show()
    fig.savefig('Eout_uniform.png')
    print(Eins)
    print(Eouts)

if __name__ == '__main__':
    main()
