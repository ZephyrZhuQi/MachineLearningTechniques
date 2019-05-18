from sys import argv
import numpy as np
import matplotlib.pyplot as plt

def read_file(file):
    X = []
    y = []
    with open(file, "r") as f:
        for line in f:
            data = line.split()
            X.append([float(x) for x in data[:-1]])
            y.append(int(data[-1]))
    X = np.array(X)
    y = np.array(y)
    return X, y

def plot_t(x, y, xlabel, ylabel, title, interval):
    fig = plt.figure()
    plt.plot(x, y, 'ro')
    plt.plot(x, y, 'b')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    i = 0
    for x1,y1 in zip(x, y):
        i = i + 1 
        if i == interval:
            plt.text(x1, 1.1*y1, str(y1))
            i = 0
        
        
    plt.show()
    fig.savefig(title+'.png')


def decision_stump(X, y, u):#(data, label, weight)
    num_feature = X.shape[1]
    N = X.shape[0]#data numbers
    #for any feature i, sort the x values
    X_sort = np.sort(X, axis = 0)
    minus_inf = float('-inf')
    min_Ein = float('inf')
    for i in range(num_feature):#feature loop
        thresholds = []
        thresholds.append(minus_inf)
        for n in range(N-1):
            midpoint = (X_sort[n][i] + X_sort[n+1][i])/2
            thresholds.append(midpoint)
        for theta in thresholds:#threshold loop
            for s in [-1, +1]:#direction loop
                score = X[:,i]-theta
                h = (s*np.where(score>0, +1, -1)).reshape(N,1)#decision stump
                #u-weighted 0/1 error
                Ein = 0
                for k in range(N):
                    if h[k]!=y[k]:
                        Ein = Ein + u[k]
                Ein = Ein / N
                if Ein < min_Ein:
                    min_Ein = Ein
                    best_s = s
                    best_i = i
                    best_theta = theta
                    predict = h

    return best_s, best_i, best_theta, min_Ein, predict

def adaboost(X, y, X_test, y_test):
    N = X.shape[0]#number of training examples
    N_test = X_test.shape[0]#number of testing examples
    u = (np.ones(N)/N).reshape(N,1)#initial weight
    T = 300#iterations
    G_score = np.zeros((N, 1))
    G_score_test = np.zeros((N_test, 1))
    Ein_gt = []#for plot
    ts = []#for plot
    Ein_Gt = []#for plot
    Ut = []#for plot
    Eout_Gt = []#for plot
    for t in range(T):
        ts.append(t)
        s, i, theta, min_Ein, predict = decision_stump(X, y, u)
        #Ein(gt)
        Ein_gtt = 0
        for k in range(N):
            if predict[k]!=y[k]:
                Ein_gtt = Ein_gtt +1
        Ein_gtt = Ein_gtt/N
        Ein_gt.append(Ein_gtt)
        #for update weights
        epsilon = min_Ein*N/sum(u)
        diamond = np.sqrt((1-epsilon)/epsilon)

        alpha = np.log(diamond)
        G_score = G_score + alpha * predict
        G = np.where(G_score>0, +1, -1)
        #Ein(Gt)
        Ein_Gtt = 0
        for k in range(N):
            if G[k]!=y[k]:
                Ein_Gtt = Ein_Gtt + 1
        Ein_Gtt = Ein_Gtt/N
        Ein_Gt.append(Ein_Gtt)
        #Eout(Gt)
        Eout_Gtt = 0
        score = X_test[:,i]-theta
        h = (s*np.where(score>0, +1, -1)).reshape(N_test,1)#decision stump
        G_score_test = G_score_test + alpha * h
        G_test = np.where(G_score_test>0, +1, -1)
        for k in range(N_test):
            if G_test[k]!=y_test[k]:
                Eout_Gtt = Eout_Gtt + 1
        Eout_Gtt = Eout_Gtt / N_test
        Eout_Gt.append(Eout_Gtt)
        #Ut
        Ut.append(float(sum(u)))
        #update weights
        for k in range(N):
            if predict[k]!= y[k]:
                u[k] = u[k]*diamond
            else:
                u[k] = u[k]/diamond

    plot_t(ts, Ein_gt, 't', 'Ein(gt)', 'Ein(gt)', 10)
    plot_t(ts, Ein_Gt, 't', 'Ein(Gt)', 'Ein(Gt)', 20)
    plot_t(ts, Ut, 't', 'Ut', 'Ut', 40)
    plot_t(ts, Eout_Gt, 't', 'Eout(Gt)', 'Eout(Gt)', 30)
    print(Ein_gt[-1])
    print(Ein_Gt[-1])
    print(Ut[-1])
    print(Eout_Gt[-1])
    return G



def main():
    X_train, y_train = read_file(argv[1])
    X_test, y_test = read_file(argv[2])
    G = adaboost(X_train, y_train, X_test, y_test)
    

if __name__ == '__main__':
    main()
