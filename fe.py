#!/usr/bin/python
#coding:utf-8
from sklearn.ensemble import RandomForestRegressor
from pandas import get_dummies
import re

### 使用 RandomForestClassifier 填补缺失的年龄属性
def set_missing_ages(df):
    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
    # 乘客分成已知年龄和未知年龄两部分
    unknown_age = age_df[age_df['Age'].isnull()].values
    known_age = age_df[age_df.Age.notnull()].values

    # y即目标年龄
    y = known_age[:, 0]

    # X即特征属性值
    X = known_age[:, 1:]

    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)

    # 用得到的模型进行未知年龄结果预测
    predictedAges = rfr.predict(unknown_age[:, 1::])

    # 用得到的预测结果填补原缺失数据
    df.loc[(df.Age.isnull()), 'Age'] = predictedAges

    return df, rfr


def set_Cabin_type(df):
    df.loc[(df.Cabin.notnull()), 'Cabin'] = "Yes"
    df.loc[(df.Cabin.isnull()), 'Cabin'] = "No"
    return df

def set_onehot_features(df):
    # Round 1
    dummies_CabinClass = get_dummies(df['CabinClass'], prefix='CabinClass')
    dummies_Embarked = get_dummies(df['Embarked'], prefix='Embarked')
    dummies_Sex = get_dummies(df['Sex'], prefix='Sex')
    dummies_Pclass = get_dummies(df['Pclass'], prefix='Pclass')
    df[list(dummies_CabinClass.dtypes.index)] = dummies_CabinClass
    df[list(dummies_Embarked.dtypes.index)] = dummies_Embarked
    df[list(dummies_Sex.dtypes.index)] = dummies_Sex
    df[list(dummies_Pclass.dtypes.index)] = dummies_Pclass
    # Round 2 - SibSp, Parch, FamilySize, IsAlone
    dummies_SibSp = get_dummies(df['SibSp'], prefix='SibSp')
    df[list(dummies_SibSp.dtypes.index)] = dummies_SibSp
    dummies_Parch = get_dummies(df['Parch'], prefix='Parch')
    df[list(dummies_Parch.dtypes.index)] = dummies_Parch
    dummies_FamilySize = get_dummies(df['FamilySize'], prefix='FamilySize')
    df[list(dummies_FamilySize.dtypes.index)] = dummies_FamilySize
    dummies_IsAlone = get_dummies(df['IsAlone'], prefix='IsAlone')
    df[list(dummies_IsAlone.dtypes.index)] = dummies_IsAlone
    # Round 2 - Age, Fare
    # dummies_Age = get_dummies(df['Age'], prefix='Age')
    # df[list(dummies_Age.dtypes.index)] = dummies_Age
    # dummies_Fare = get_dummies(df['Fare'], prefix='Fare')
    # df[list(dummies_Fare.dtypes.index)] = dummies_Fare
    # Round 2 - Title
    dummies_title = get_dummies(df['Title'], prefix='Title')
    df[list(dummies_title.dtypes.index)] = dummies_title
    return df

def get_title(name):
	title_search = re.search(' ([A-Za-z]+)\.', name)
	# If the title exists, extract and return it.
	if title_search:
		return title_search.group(1)
	return ""

def get_cabin_number(cabin):
    cabin_class_search = re.search('([A-Z])', cabin)
    # cabin_number_search = re.search('[A-Z](\d+)', cabin)
    if cabin_class_search:
        return cabin_class_search.group(1)
    print(cabin)
    return ""