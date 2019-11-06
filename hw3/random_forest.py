# random forest (RF) = bagging + fully-grown C&RT decision tree
# using unpruned_part.py

import matplotlib.pyplot as plt
import numpy as np
import unpruned_cart

num_tree = 30000


def main():
    X, y = unpruned_cart.read_data('data/hw3_train.dat')
    X_test,y_test = unpruned_cart.read_data('data/hw3_test.dat')
    #unpruned_cart.plot_data(X, y)
    num_train = X.shape[0]
    num_test = X_test.shape[0]
    #print(num_train)
    ts = [i for i in range(num_tree)]# x axis
    Ein_gts = []# Ein of 30000 trees
    G = np.zeros(num_train)#forest for training
    G_test = np.zeros(num_test)# forest for testing
    Ein_Gts = []
    Eout_Gts = []
    for iteration in range(num_tree):
        idxs = np.random.randint(0, num_train, size = int(num_train*0.8))
        X_train_t = []# bootstrapped examples
        y_train_t = []
        for i in idxs:
            X_train_t.append(X[i])
            y_train_t.append(y[i])
        X_train_t = np.array(X_train_t)
        y_train_t = np.array(y_train_t)
        #unpruned_cart.plot_data(X_train_t, y_train_t)
        #Gtree = unpruned_cart.DecisionTree(X_train_t, y_train_t)
        #plt.savefig('data_split.png')
        #plt.pause(0.1)
        #unpruned_cart.treePlotter.createPlot(Gtree)
        Gtree_predict = unpruned_cart.DecisionTree_predict(X_train_t, y_train_t)
        # calculate Ein
        Ein = 0
        predict = []
        for x_vec in X:
            predict.append(unpruned_cart.predict_function(x_vec, Gtree_predict))
        for i in range(len(y)):
            if predict[i] != y[i]:
                Ein += 1
        Ein /= len(y)
        print("Ein: ", Ein)
        Ein_gts.append(Ein)
        # forest 
        G = G + predict
        G_sign = np.where(G>0,+1,-1)
        Ein_Gt = 0 # the random forest with the first t trees
        for i in range(len(y)):
            if G_sign[i]!=y[i]:
                Ein_Gt +=1
        Ein_Gt/=len(y)
        print("Ein_Gt: ",Ein_Gt)
        Ein_Gts.append(Ein_Gt)
        # calculate Eout
        #Eout = 0
        predict_test = []
        for x_vec in X_test:
            predict_test.append(unpruned_cart.predict_function(x_vec, Gtree_predict))
        G_test = G_test + predict_test
        G_test_sign = np.where(G_test>0, +1, -1)
        Eout_Gt = 0
        for i in range(len(y_test)):
            if G_test_sign[i]!=y_test[i]:
                Eout_Gt +=1
        Eout_Gt/=len(y_test)
        print("Eout_Gt: ",Eout_Gt)
        Eout_Gts.append(Eout_Gt)
        '''for i in range(len(y_test)):
            if predict_test[i] != y_test[i]:
                Eout += 1
        Eout /= len(y_test)
        print("Eout: ", Eout)'''
        # print(Gtree)
    fig = plt.figure()
    #plt.bar(ts,Ein_gts)
    plt.hist(Ein_gts,bins=np.linspace(0,0.16,17))
    plt.xlabel('Ein(gt)')
    plt.ylabel('count')
    plt.show()
    fig.savefig('Ein(gt)_30000.png')

    fig = plt.figure()
    plt.plot(ts,Ein_Gts,'ro')
    plt.plot(ts,Ein_Gts,'b')
    plt.xlabel('the random forest with the first t trees')
    plt.ylabel('the Ein of random forest')
    plt.show()
    fig.savefig('Ein(Gt)_rf.png')

    fig = plt.figure()
    plt.plot(ts,Eout_Gts,'ro')
    plt.plot(ts,Eout_Gts,'b')
    plt.xlabel('the random forest with the first t trees')
    plt.ylabel('the Eout of random forest')
    plt.show()
    fig.savefig('Eout(Gt)_rf.png')

if __name__ == '__main__':
    main()
