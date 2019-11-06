import numpy as np 
import matplotlib.pyplot as plt 

def read_data(file):
    X = []
    with open(file, 'r') as f:
        for line in f:
            data = line.split()
            X.append([float(x) for x in data[:]])
    X = np.array(X)
    return X

def Euclidean_D(x1, x2):
    return np.sqrt(np.sum(np.power((x1-x2), 2)))

def converge_d(u1,u2):
    return np.array_equal(u1, u2)

def main():
    ks = [2, 4, 6, 8, 10]
    #ks = [2,3]
    X = read_data('./data/hw4_nolabel_train.dat')
    n_train = X.shape[0]
    Ein_AVGs = []
    Ein_VARs = []
    for k in ks:
        Ein_ts = []
        for _ in range(500):
            centroids = X[np.random.choice(n_train,k,replace=False)]
            label = np.zeros(n_train)
            converge = False
            error = np.zeros(n_train)
            while not converge:
                old_centroids = np.copy(centroids)
                for i in range(n_train):
                    min_distance = np.inf
                    for j in range(k):
                        distance = Euclidean_D(X[i],centroids[j])
                        if distance < min_distance:
                            min_distance = distance
                            label[i] = j 
                for i in range(k):
                    centroids[i] = np.mean(X[label==i],axis = 0)
                converge = converge_d(old_centroids,centroids)

            for i in range(n_train):
                error[i] = Euclidean_D(X[i],centroids[int(label[i])])**2
            Ein_t = np.sum(error)
            Ein_t/=n_train
            Ein_ts.append(Ein_t)
        Ein_AVG = np.mean(Ein_ts)
        Ein_AVGs.append(Ein_AVG)
        Ein_VAR = np.var(Ein_ts)
        Ein_VARs.append(Ein_VAR)
    # plotting
    fig = plt.figure()
    plt.plot(ks, Ein_AVGs, 'b')
    plt.plot(ks, Ein_AVGs, 'ro')
    plt.title('Average Ein with respect to k')
    plt.xlabel('k')
    plt.ylabel('Average Ein')
    for x,y in zip(ks, Ein_AVGs):
        plt.text(x, y, str(y))
    plt.show()
    fig.savefig('Ein_avg_kmeans.png')

    fig = plt.figure()
    plt.plot(ks, Ein_VARs, 'b')
    plt.plot(ks, Ein_VARs, 'ro')
    plt.title('Variance Ein with respect to k')
    plt.xlabel('k')
    plt.ylabel('Variance Ein')
    for x,y in zip(ks, Ein_VARs):
        plt.text(x, y, str(y))
    plt.show()
    fig.savefig('Ein_var_kmeans.png')

if __name__ == '__main__':
    main()