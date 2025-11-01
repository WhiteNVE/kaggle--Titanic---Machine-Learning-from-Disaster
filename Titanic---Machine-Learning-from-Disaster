import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
from math import log2
###解决是否生还问题我们有三个经典模型，分别是逻辑回归，决策树，随机森林等模型
###首先，我们先构造出逻辑回归模型
class logic_regression:
    #标准化X，同时对X样本数据进行清洗
    def __init__(self,penalty_method = 'l2',lambda_ = 0.01):
        #为了防止过拟合我们可以加入惩罚机制，可以将其默认为正则化来进行防过拟合
        self.penalty = penalty_method
        self.lambda_ = lambda_
    def X_standard(self,X):
        a = 0
        ###因为当进行预测结果的时候的数据是没有survived这个特征的，所以我们要对其进行区分
        if 'Survived' in X.columns:
            a = 1
            Y = X['Survived'].copy()
            X = X.drop(columns = ['Survived']).copy()
        ###对客观无关数据删去
        X = X.drop(columns = ['PassengerId','Name','Ticket','Cabin']).copy()
        ###对样本数据进行清洗NA值
        X['Age'] = X['Age'].fillna(X['Age'].median())
        X['Embarked'] = X['Embarked'].ffill()
        ###因为逻辑回归所用到的数据必须得是数值类型，所以我们将字符类型转化为数值类型
        X['Sex'] = X['Sex'].map({'female' : 0,'male' : 1})
        ###使用独热编码来对分类型字符进行数值转化
        X = pd.get_dummies(X,columns = ['Embarked'],dtype = int)
        X = X.fillna(X.median())
        X = X.to_numpy()
        ###使用scikitlearn库中的标准化函数来对数据进行缩放，以减小梯度计算误差
        X_stand = StandardScaler()
        X_scaler = X_stand.fit_transform(X)
        X_scaler = np.column_stack([np.ones(X_scaler.shape[0]), X_scaler])
        ###我们需要保存标准化函数，目的是为了在预测时将接收的预测数据进行同比例标准化
        self.X_stand = X_stand
        if a == 0: return X_scaler
        return X_scaler,Y
    #使用S函数计算概率
    def sigmoid(self,Z):
        return 1 / (1 + np.exp(-np.clip(Z,-500,500)))
    #梯度，即交叉熵对于theta的偏导
    def theta_grad(self,X,Y,theta):
        Z = np.dot(X,theta)
        hx = self.sigmoid(Z)
        grad = (np.dot(X.T,hx - Y)) / len(X)
        #如果要加入惩罚机制来防止过拟合的话，我们使用正则化
        if self.penalty == 'l2' :
            grad[1:] += self.lambda_ * theta[1:] / len(theta)
        return grad
    #训练并计算出theta
    def fit(self,X,time = 1000,a = 0.001):
        #对X进行缩放减小误差
        X_scaler,Y = self.X_standard(X)
        theta = np.zeros(X_scaler.shape[1])
        for i in range(time):
            grad = self.theta_grad(X_scaler,Y,theta)
            theta = theta - a * grad
        self.theta = theta
        self.X = X_scaler
        self.Y = Y
    #以上是使用梯度下降法计算theta，而以下是torch优化器计算theta
    def fit_torch(self, X, time=1000):
        X_scaler,Y = self.X_standard(X)
        theta = torch.zeros(X_scaler.shape[1], requires_grad=True)
        X_scaler = torch.tensor(X_scaler, dtype=torch.float)
        Y = torch.tensor(Y, dtype=torch.float)
        optimizer = torch.optim.Adam(params = [theta], lr=0.001)
        for i in range(1000):
            optimizer.zero_grad()
            Z = torch.matmul(X_scaler, theta)
            hx = torch.sigmoid(Z)
            ###如果不进行正则化时可以直接使用torch内置的交叉熵函数
            loss = nn.BCELoss()(hx, Y)
            ###但是如果进行了正则化，则要用自己所算出的交叉熵函数再用优化器进行梯度计算
            loss.backward()
            optimizer.step()
        self.theta = theta.detach().numpy()
        self.X = X_scaler.numpy()
        self.Y = Y.numpy()
    ########如下的fit_nomalizetion函数是我编写的正规方程计算最佳theta
    ########但是逻辑回归是没有正规方程的，所以如下是我写的错误算法
    def fit_nomalization(self, X):
        # 这里的X我们不写成X_scaler了，因为比较繁琐
        X,Y = self.X_standard(X)
        theta = np.zeros(X.shape[1])
        theta = (np.linalg.inv(X.T @ X)) @ X.T @ Y
        self.theta = theta
        self.X = X
        self.Y = Y
    ###预测函数，用于对测试样本进行预测最终是否生还
    def predict(self,X):
        X_scaler = self.X_standard(X)
        Z = np.dot(X_scaler,self.theta)
        probability = self.sigmoid(Z)
        survived = np.where(probability >= 0.5,1,0)
        return survived
    ###计算交叉熵，交叉熵越小说明越稳定，误差越小，但可能出现过拟合的情况使得交叉熵很小，但是最终预测结果偏差较大
    def cross_entropy(self):
        X = self.X
        Y = self.Y
        Z = np.dot(X,self.theta)
        hx = self.sigmoid(Z)
        entropy = -(Y*np.log(hx + 1e-15) + (1 - Y) * np.log(1 - hx + 1e-15)).mean()
        ###如果使用了正则化，那么交叉熵的公式将会改变
        if self.penalty == 'l2':
            entropy += 0.5 * self.lambda_ * ((self.theta[1:]**2).sum())/len(self.theta)
        return entropy
###以上我们便实现了逻辑回归模型，接下来继续编写决策树模型构建
class DecisionTreeClassifier():
    def __init__(self,depth_max = 6):
        ###这里的最大深度是指一棵树的最大分支数量，目的是为了防止出现过拟合情况
        self.depth_max = depth_max
        self.target = 'Survived'
    def data_standard(self,X):
        ###依旧先进行数据清洗
        X = X.drop(columns = ['PassengerId','Name','Ticket','Cabin'])
        X['Embarked'] = X['Embarked'].fillna(X['Embarked'].mode()[0])
        numeric_features = X.select_dtypes(include = ['int64','float64']).columns
        str_features = X.select_dtypes(include = ['object','category']).columns
        for feature in numeric_features:
            X[feature] = X[feature].fillna(X[feature].median())
        for feature in str_features:
            X[feature] = X[feature].fillna(X[feature].mode()[0])
        ###决策树与逻辑回归存在着对特征类型的要求不同，对决策树而言无论是数值类型还是字符类型都可以进行预测
        ###但是通常是将数值类型映射为分类类型，因为很难存在相同大小的数值，决策树就是要依赖于对相同数值的分支进行再分支来一步步预测
        X['Age'] = pd.cut(X['Age'],bins = [0,12,18,35,60,100],labels = [1,2,3,4,5])
        X['Fare'] = pd.cut(X['Fare'],bins = [0,10,20,30,40,50,60,1000],labels = [1,2,3,4,5,6,7])
        return X
    ###计算熵值
    def entropy(self,X,target):
        if len(X) == 1: return 0
        values,counts = np.unique(X[target],return_counts = True)
        partition = counts / len(X)
        return -(partition * np.log2(partition)).sum()
    ###获得最大信息增量的特征以及最大信息增量
    def imformation_gain_best(self,X,target):
        max_gain = -1
        max_feature = None
        total_entropy = self.entropy(X, target)
        X_tmp = X.drop(columns = target)
        for feature in X_tmp.columns:
            values,counts = np.unique(X[feature],return_counts = True)
            sum_entropy = 0.0
            for value,count in zip(values,counts):
                subset = X[X[feature] == value]
                entropy = self.entropy(subset,target)
                partition = count / len(X)
                if feature == target:
                    print(partition,entropy)
                sum_entropy += partition * entropy
            gain = total_entropy - sum_entropy
            if gain > max_gain :
                max_gain = gain
                max_feature = feature
        return max_feature,max_gain
    ###fit_one与fit是连起来用的，这样方便理顺逻辑
    def fit_one(self,X,target,depth = 0):
        if len(np.unique(X[target])) == 1: return X[target].iloc[0]
        if len(X.columns) == 1 : return X[target].mode()[0]
        if depth >= self.depth_max : return X[target].mode()[0]
        feature,gain = self.imformation_gain_best(X,target)
        # print(feature,gain)
        if gain <= 0 :
            if len(X) == 1:
                return X[target][0]
            return X[target].mode()[0]
        tree = {feature : {}}
        values = np.unique(X[feature])
        for value in values:
            subset = X[X[feature] == value].drop(columns = feature)
            if len(subset) == 0:
                return X[target].mode()[0]
            tree[feature][value] = self.fit_one(subset,target,depth + 1)
        return tree
    def fit(self,X,is_random_trees = False):
        if not is_random_trees:
            X = self.data_standard(X)
        self.tree = self.fit_one(X,self.target)
    ###predict_one与predict一样是连起来使用的，这样子逻辑跟简单理解
    def predict_one(self,X,tree):
        if not isinstance(tree,dict) : return tree
        feature = list(tree.keys())[0]
        value = X[feature]
        if value not in tree[feature]:
            value = np.random.choice(list(tree[feature].keys()))
        return self.predict_one(X,tree[feature][value])
    def predict(self,sample,is_random_trees = False):
        if not is_random_trees:
            sample = self.data_standard(sample)
        prediction = []
        sample = sample.reset_index()
        for i in range(len(sample)):
            result = self.predict_one(sample.iloc[i],self.tree)
            prediction.append(result)
        return prediction
###以上便完成了决策树的构建，而决策树主要用于量化以构建出随机森林模型
###随机森林是训练出多棵树并让多棵树对数据进行预测并取得最大预测结果，大大减小了过拟合情况的发生
class random_trees:
    def __init__(self,max_time = 500):
        ###这里的最大次数是指构建出树的最大数量
        self.max_time = max_time
    ###同样要对数据进行标准化，与决策树标准化相似
    def sample_standard(self,X):
        self.target = 'Survived'
        X = X.drop(columns = ['PassengerId','Name','Ticket','Cabin'])
        X['Embarked'] = X['Embarked'].fillna(X['Embarked'].mode()[0])
        numeric_features = X.select_dtypes(include = ['int64','float64']).columns
        str_features = X.select_dtypes(include = ['object','category']).columns
        for feature in numeric_features:
            X[feature] = X[feature].fillna(X[feature].median())
        for feature in str_features:
            X[feature] = X[feature].fillna(X[feature].mode()[0])
        X['Age'] = pd.cut(X['Age'],bins = [-1,12,18,35,60,100],labels = [1,2,3,4,5])
        X['Fare'] = pd.cut(X['Fare'],bins = [-1,10,20,30,40,50,60,1000],labels = [1,2,3,4,5,6,7])
        # count = X.isna().sum(axis = 0)
        # print(count)
        return X
    ###有放回的抽取样本，使得样本多样性高并且提供自然验证机制
    def boot_strap_samples(self,samples):
        indices = np.random.choice(len(samples),len(samples),replace = True)
        sample = samples.iloc[indices]
        return sample
    ###无放回的抽取特征用于样本构建当中，此处不进行有放回抽样是因为特征数量往往比较小而样本数量往往比较大，更符合自然机制
    def get_random_feature(self,samples,target):
        samples = samples.drop(columns = target)
        column = np.random.choice(samples.columns,int(len(samples.columns) / 2) + 1,replace = False).tolist()
        column.append(target)
        return column
    ###根据样本数据训练出多棵决策树
    def fit(self,samples):
        trees = []
        features = []
        samples = self.sample_standard(samples)
        for i in range(self.max_time):
            sample = self.boot_strap_samples(samples)
            indices = self.get_random_feature(sample,self.target)
            tree = DecisionTreeClassifier()
            sample = sample.reset_index()
            tree.fit(sample[indices],is_random_trees = True)
            trees.append(tree)
            indices.remove(self.target)
            features.append(indices)
        self.trees = trees
        self.features = features
    ###根据所训练的多棵决策树对最终结果进行预测
    def predict(self,sample):
        sample = self.sample_standard(sample)
        result = []
        for tree,feature in zip(self.trees,self.features):
            result.append(tree.predict(sample[feature],is_random_trees = True))
        result = np.array(result).T
        values,counts = np.unique(result,return_counts = True)
        max_sort = np.argmax(counts)
        result = result[:,max_sort]
        return result
###以上我们便完成了三种模型的构建，那么我可以开始测试每个模型的精确率了
def get_accuracy(sample_key,result):
    count1 = np.where(result == sample_key['Survived'], 1, 0).sum()
    accuracy = count1 / len(sample_key)
    return accuracy
###用于训练模型的样本数据
sample_train = pd.read_csv(r"C:\study\project\train.csv",encoding = 'utf-8')
###用于测试所训练出的模型的精确率的测试样本
sample_test = pd.read_csv(r"C:\study\project\test.csv",encoding = 'utf-8')
###这是sample_test的是否生还表
sample_key = pd.read_csv(r"C:\study\project\gender_submission.csv",encoding = 'utf-8')
###使用逻辑回归中的torch优化器进行预测
logic_torch = logic_regression()
logic_torch.fit_torch(sample_train)
result1 = logic_torch.predict(sample_test)
accuracy1 = get_accuracy(sample_key,result1)
###构建决策树进行预测
tree = DecisionTreeClassifier()
tree.fit(sample_train)
result2 = tree.predict(sample_test)
accuracy2 = get_accuracy(sample_key,result2)
trees = random_trees()
###构建随机森林进行预测
trees.fit(sample_train)
result3 = trees.predict(sample_test)
accuracy3 = get_accuracy(sample_key,result3)
print(accuracy1)#0.8851
print(accuracy2)#0.9090
print(accuracy3)#0.9186
###通过比较这三个模型的准确率我们可以得出结论：随机森林在该测试样例中的准确率是最高的
###封装为csv文件并上传到kaggle中
submission = pd.DataFrame({
    'PassengerId':sample_key['PassengerId'],
    'Survived': result3
})
submission.to_csv(r"C:\study\submission.csv",index = False)
###最终得分0.76315



























