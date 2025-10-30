from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np
X,Y = make_classification(n_samples = 1000,n_features = 20,n_informative = 15,n_redundant = 5,random_state = 42)
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.3,random_state = 42)
tree = DecisionTreeClassifier(random_state = 42)
tree.fit(X_train,Y_train)
prediction1 = tree.predict(X_test)
# print(prediction1)
accuracy1 = len(prediction1[prediction1 == Y_test]) / len(Y_test)
# print(accuracy1)
from sklearn.ensemble import RandomForestClassifier
trees = RandomForestClassifier(n_estimators = 100,random_state = 42)
trees.fit(X_train,Y_train)
prediction2 = trees.predict(X_test)
accuracy2 = len(prediction2[prediction2 == Y_test])/len(Y_test)
# print(accuracy2)
def entropy(Y):
    if len(Y) == 0:
        return 0
    values,counts = np.unique(Y,return_counts = True)
    probabilities = counts/len(Y)
    return -np.sum(probabilities * np.log2(probabilities + 1e-9))
import matplotlib.pyplot as plt
def feature_importance(X,Y):
    entropy_list = []
    total_entropy = entropy(Y)
    feature_list = []
    for feature in range(X.shape[1]):
        values,counts = np.unique(X[:,feature],return_counts = True)
        entropy_sum = 0.0
        for i in range(len(values)):
            subset_X = X[X[:,feature] == values[i]]
            subset_Y = Y[X[:,feature] == values[i]]
            if len(subset_Y) > 0:
                entropy_ = entropy(subset_Y)
                entropy_sum += entropy_* (counts[i] / len(X))
        entropy_list.append(total_entropy - entropy_sum)
        feature_list.append(feature)
    plt.plot(feature_list,entropy_list)
    plt.show()

# X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.3,random_state = 42)
# values,count = np.unique(X_train[:,0],return_counts = True)
# print(Y_train.shape)
# print(count)
# print(values)
# for i in range(len(values)):
#     subset_Y = Y_test[X_test[:,0] == values[i]]
#     print(subset_Y)
#     print('*'*30)
# data = pd.DataFrame({
#     '天气': ['晴','晴','阴','雨','雨','雨','阴','晴','晴','雨','晴','阴','阴','雨'],
#     '温度': ['热','热','热','温','凉','凉','凉','温','凉','温','温','温','热','温'],
#     '湿度': ['高','高','高','高','正常','正常','正常','高','正常','正常','正常','高','正常','高'],
#     '风速': ['弱','强','弱','弱','弱','强','强','弱','弱','弱','强','强','弱','强'],
#     '打网球': ['否','否','是','是','是','否','是','否','是','是','是','是','是','否']
# })
# list = np.random.choice(10,200,replace= True).reshape(20,10)
# list2 = np.random.choice([1,2],20)
# print(list)
# feature_importance(X,Y)

















