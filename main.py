#!/usr/bin/python
#coding:utf-8
import pandas as pd
import fe
from pandas import Series,DataFrame
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_colwidth', 200)
pd.set_option('display.width', None)

print("=======1. 读入数据并观察数据状态=======")
data_train = pd.read_csv("/Users/apple/Desktop/train2.csv")
data_test = pd.read_csv("/Users/apple/Desktop/test2.csv")
data_test.loc[data_test['Fare'].isnull(), 'Fare'] = 0 # 特殊化处理，因为只有一个空值，直接置为0
full_data = [data_train, data_test]

print("=======1.1 随机森林拟合年龄需要特殊处理=======")
isFirstTime = False
if isFirstTime:
    for dataset in full_data:
        dataset, rfr = fe.set_missing_ages(dataset)  # 4. Age -> 补充missing age
    data_train.to_csv("/Users/apple/Desktop/train2.csv", index=False)
    data_test.to_csv("/Users/apple/Desktop/test2.csv", index=False)

print("=======3. Feature Engineering =======")
import sklearn.preprocessing as preprocessing
scaler = preprocessing.StandardScaler()

for dataset in full_data:
    dataset = fe.set_Cabin_type(dataset) # 1. Cabin -> 补充missing cabin
    dataset = fe.set_onehot_features(dataset) # 2. Cabin, Embarked, Sex, Pclass -> OneHot
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1 # 3. Family_size -> 新增
    dataset_Age = pd.DataFrame(dataset['Age']) # 5. Age -> Scaling
    age_scale_param = scaler.fit(dataset_Age)
    dataset['Age_scaled'] = scaler.fit_transform(dataset_Age, age_scale_param)
    dataset_Fare = pd.DataFrame(dataset['Fare']) # 6. Fare -> Scaling
    fare_scale_param = scaler.fit(dataset_Fare)
    dataset['Fare_scaled'] = scaler.fit_transform(dataset_Fare, age_scale_param)

print("+++++++3.1 Train Data Preview+++++++")
#print(data_train.head(5))
print("+++++++3.2 Test Data Preview+++++++")
#print(data_test.head(5))

print("=======2. 分析数据分布=======")
pclass_stat = data_train[['Pclass', 'Survived']].groupby('Pclass', as_index=False).mean()
sibsp_stat = data_train[['SibSp', 'Survived']].groupby('SibSp', as_index=False).mean()
parch_stat = data_train[['Parch', 'Survived']].groupby('Parch', as_index=False).mean()
familysize_stat = data_train[['FamilySize', 'Survived']].groupby('FamilySize', as_index=False).mean()
print(pd.concat([pclass_stat, sibsp_stat, parch_stat, familysize_stat], axis=1))

print("=======4. Cross Validation Scores =======")
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

 #简单看看打分情况
clf = LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
all_data = data_train.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*|FamilySize')
X = all_data.values[:,1:]
y = all_data.values[:,0]
scores = cross_val_score(clf, X, y, cv=5)
print(scores)

print("=======5. 训练模型 =======")
# 用正则取出我们要的属性值
train_df = data_train.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*|FamilySize')
train_np = train_df.values

# y即Survival结果
y = train_np[:, 0]

# X即特征属性值
X = train_np[:, 1:]

# fit到RandomForestRegressor之中
clf = LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
clf.fit(X, y)

print(pd.DataFrame({"columns":list(train_df.columns)[1:], "coef":list(clf.coef_.T)}))