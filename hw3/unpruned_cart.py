import numpy as np 
import matplotlib.pyplot as plt
import treePlotter

#iteration = 0
       
def read_data(file):
    X,y = [],[]
    with open(file,'r') as f:
        for line in f:
            data = line.split()
            X.append([float(x) for x in data[0:-1]])
            y.append(int(data[-1]))
    X = np.array(X)
    y = np.array(y)
    return X,y

def plot_data(X,y):# plot the x and o markers
    plt.figure()
    idx_positive = np.where(y == 1)
    idx_negative = np.where(y == -1)
    plt.scatter(X[idx_positive,0],X[idx_positive,1],facecolors='none',edgecolors='b',linewidths=3,label='+1')
    plt.scatter(X[idx_negative,0],X[idx_negative,1],marker='x',color='r',label='-1')
    plt.legend(loc = 'best')
    
def plot_line(i,theta):# plot the small trees separating the plane horizontally/vertically
    if i == 0:
        plt.axvline(x=theta,linewidth=5,color='gray') # vertical line
    elif i == 1:
        plt.axhline(y=theta,linewidth=5,color='gray') # horizontal line

def gini_index(X,y):# (subset_data, subset_label)
    N = X.shape[0]
    if N == 0:# this got be changed
        return 0
    positive_ratio = np.where(y > 0)[0].size/N
    negative_ratio = 1 - positive_ratio
    gini = 1 - np.power(positive_ratio,2) - np.power(negative_ratio,2)
    return gini

def decision_stump(X, y):#(data, label)
    num_feature = X.shape[1]
    N = X.shape[0]#data numbers
    #for any feature i, sort the x values
    X_sort = np.sort(X, axis = 0)
    minus_inf = float('-inf')
    min_weighted_impurity = float('inf')
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
                # calculate weighted impurity
                left_ratio = (np.where(h == +1)[0].size)/N#weights
                right_ratio = 1-left_ratio#weights
                left_index = np.where(h == +1)[0]
                right_index = np.where(h == -1)[0]
                left_subset = X[left_index,:]
                right_subset = X[right_index,:]
                left_subset_y = y[left_index]
                right_subset_y = y[right_index]
                left_impurity = gini_index(left_subset,left_subset_y)
                right_impurity = gini_index(right_subset,right_subset_y)
                weighted_impurity = left_ratio*left_impurity + right_ratio*right_impurity
                if weighted_impurity < min_weighted_impurity:
                    min_weighted_impurity = weighted_impurity
                    best_s = s
                    best_i = i
                    best_theta = theta
                    best_left_subset = left_subset
                    best_left_subset_y = left_subset_y
                    best_right_subset = right_subset
                    best_right_subset_y = right_subset_y
                                  
    return best_s, best_i, best_theta, \
    best_left_subset, best_left_subset_y, best_right_subset, best_right_subset_y


def DecisionTree(X,y):
    #global iteration
    if np.all(y==y[0]):#如果所有的label都一样
    #if terminate(all yn the same or all xn the same):
        return y[0]
        #return gc(x)
    else:
        s,i,theta,left_X,left_y,right_X,right_y = decision_stump(X,y)
        print("feature %d > %f?"%(i,theta))
        plot_line(i,theta)
        plt.pause(1)
        #iteration = iteration + 1
        #if iteration == 5:
        #    return 
        left = DecisionTree(left_X,left_y)
        right = DecisionTree(right_X,right_y)
        Gtree = {}
        if s == +1:
            Gtree['feature'+str(i)+'>'+str(theta)] = {'yes':left,'no':right}
        elif s == -1:
            Gtree['feature'+str(i)+'<='+str(theta)] = {'yes':left,'no':right}
        return Gtree
        #G(x)= sum of Gc(x)

def DecisionTree_predict(X,y):
    #global iteration
    if np.all(y==y[0]):#如果所有的label都一样
    #if terminate(all yn the same or all xn the same):
        return y[0]
        #return gc(x)
    else:
        s,i,theta,left_X,left_y,right_X,right_y = decision_stump(X,y)
        #plot_line(i,theta)
        #plt.pause(1)
        #iteration = iteration + 1
        #if iteration == 5:
        #    return 
        left = DecisionTree_predict(left_X,left_y)
        right = DecisionTree_predict(right_X,right_y)
        Gtree = {}
        if s == +1:
            Gtree['feature'] = i
            Gtree['value'] = theta
            Gtree['left'] = left
            Gtree['right'] = right
        elif s == -1:
            Gtree['feature'] = i
            Gtree['value'] = theta
            Gtree['left'] = right
            Gtree['right'] = left
        return Gtree
        #G(x)= sum of Gc(x)

def predict_function(x,G_tree):
    if type(G_tree).__name__ != 'dict':
        return G_tree
    feat = G_tree['feature']
    theta = G_tree['value']
    left = G_tree['left']
    right = G_tree['right']
    if x[feat]>theta:
        if type(left).__name__ == 'dict':
            return predict_function(x,left)
        else:
            return left
    elif x[feat]<=theta:
        if type(right).__name__ == 'dict':
            return predict_function(x,right)
        else:
            return right

def main():
    X,y=read_data('data/hw3_train.dat')
    plot_data(X,y)
    Gtree = DecisionTree(X,y)
    plt.savefig('data_split.png')
    plt.show()
    treePlotter.createPlot(Gtree)
    Gtree_predict = DecisionTree_predict(X,y)
    # calculate Ein
    Ein = 0
    predict=[]
    for x_vec in X:
        predict.append(predict_function(x_vec, Gtree_predict))
    for i in range(len(y)):
        if predict[i] != y[i]:
            Ein += 1
    Ein/=len(y)
    print("Ein: ",Ein)
    # calculate Eout
    X_test,y_test = read_data('data/hw3_test.dat')
    Eout = 0
    predict_test = []
    for x_vec in X_test:
        predict_test.append(predict_function(x_vec,Gtree_predict))
    for i in range(len(y_test)):
        if predict_test[i] != y_test[i]:
            Eout += 1
    Eout/=len(y_test)
    print("Eout: ",Eout)
    #print(Gtree)
    

if __name__ == '__main__':
    main()
