#!/usr/bin/python
#coding:utf-8
import pandas as pd
import numpy as np
import fe
from pandas import Series,DataFrame
pd.set_option('display.max_rows', 1000)
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

print("=======2. Feature Engineering =======")
import sklearn.preprocessing as preprocessing
scaler = preprocessing.StandardScaler()

for dataset in full_data:
    # dataset = fe.set_Cabin_type(dataset) # 1. Cabin -> 补充missing cabin
    dataset.loc[(dataset.Cabin.isnull()), 'Cabin'] = "N0" # 1. Round 2 - CabinClass
    dataset['CabinClass'] = dataset['Cabin'].apply(fe.get_cabin_number)

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1  # 2. Family_size -> 新增
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

    dataset['CategoricalAge'] = pd.cut(dataset['Age'], [0,16,32,48,64,80]) # 4. Age -> Categorize
    dataset['CategoricalFare'] = pd.qcut(dataset['Fare'], 4) # 5. Fare -> Categorize

    # # Mapping Age
    # dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
    # dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    # dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    # dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    # dataset.loc[dataset['Age'] > 64, 'Age'] = 4
    #
    # # Mapping Fare
    # dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
    # dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    # dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
    # dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
    # dataset['Fare'] = dataset['Fare'].astype(int)

    dataset['Title'] = dataset['Name'].apply(fe.get_title) # 6. Find out the title in the name
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col',
                                                 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    dataset = fe.set_onehot_features(
        dataset)  # 3. Cabin, Embarked, Sex, Pclass, SibSp, Parch, FamilySize, IsAlone, Age, Fare -> OneHot

    dataset_Age = pd.DataFrame(dataset['Age']) # 4. Age -> Scaling
    age_scale_param = scaler.fit(dataset_Age)
    dataset['Age_scaled'] = scaler.fit_transform(dataset_Age, age_scale_param)
    dataset_Fare = pd.DataFrame(dataset['Fare']) # 5. Fare -> Scaling
    fare_scale_param = scaler.fit(dataset_Fare)
    dataset['Fare_scaled'] = scaler.fit_transform(dataset_Fare, age_scale_param)

print("=======只有一个T舱的乘客，特殊处理=======")
data_train.drop(['CabinClass_T'], axis=1, inplace=True)
print(data_train.head(5))
print(data_test.head(5))

print("=======3. 分析数据分布=======")
pclass_stat = data_train[['Pclass', 'Survived']].groupby('Pclass', as_index=False).mean()
sibsp_stat = data_train[['SibSp', 'Survived']].groupby('SibSp', as_index=False).mean()
parch_stat = data_train[['Parch', 'Survived']].groupby('Parch', as_index=False).mean()
familysize_stat = data_train[['FamilySize', 'Survived']].groupby('FamilySize', as_index=False).mean()
embark_stat = data_train[['Embarked', 'Survived']].groupby('Embarked', as_index=False).mean()
isalone_stat = data_train[['IsAlone', 'Survived']].groupby('IsAlone', as_index=False).mean()
age_stat = data_train[['CategoricalAge', 'Survived']].groupby('CategoricalAge', as_index=False).mean()
fare_stat = data_train[['CategoricalFare', 'Survived']].groupby('CategoricalFare', as_index=False).mean()
print(pd.concat([pclass_stat, embark_stat, sibsp_stat, parch_stat, familysize_stat, isalone_stat, age_stat, fare_stat], axis=1))

print("=======3.1 Round 2 - Cabin Class Statistics =======")
print(data_train[['CabinClass', 'Survived']].groupby('CabinClass', as_index=False).mean())

print("=======4. Cross Validation Scores =======")
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

 #简单看看打分情况
clf = LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
all_data = data_train.filter(regex='Survived|Age_.*|SibSp_.*|Parch_2|Parch_3|Fare_.*|CabinClass_.*|Embarked_.*|Sex_.*|Pclass_.*|FamilySize_.*|IsAlone_.*|Title_.*')
X = all_data.values[:,1:]
y = all_data.values[:,0]
scores = cross_val_score(clf, X, y, cv=5)
print(scores)

print("=======5. 训练模型 =======")
# 用正则取出我们要的属性值
train_df = data_train.filter(regex='Survived|Age_.*|SibSp_.*|Parch_2|Parch_3|Fare_.*|CabinClass_.*|Embarked_.*|Sex_.*|Pclass_.*|FamilySize_.*|IsAlone_.*|Title_.*')
train_np = train_df.values

# y即Survival结果
y = train_np[:, 0]

# X即特征属性值
X = train_np[:, 1:]

# fit到RandomForestRegressor之中
clf2 = LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
clf2.fit(X, y)

coef_pd = pd.DataFrame({"columns":list(train_df.columns)[1:], "coef":list(clf2.coef_.T)})
print(coef_pd)

print("=======6. 预测结果 =======")
test = data_test.filter(regex='Age_.*|SibSp_.*|Parch_.*|Fare_.*|CabinClass_.*|Embarked_.*|Sex_.*|Pclass_.*|FamilySize_.*|IsAlone_.*|Title_.*')
predictions = clf2.predict(test)
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].values, 'Survived':predictions.astype(np.int32)})
result.to_csv("/Users/apple/Desktop/result.csv", index=False)