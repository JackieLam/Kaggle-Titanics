#!/usr/bin/python
#coding:utf-8
from sklearn.ensemble import RandomForestRegressor
from pandas import get_dummies

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
    dummies_Cabin = get_dummies(df['Cabin'], prefix='Cabin')
    dummies_Embarked = get_dummies(df['Embarked'], prefix='Embarked')
    dummies_Sex = get_dummies(df['Sex'], prefix='Sex')
    dummies_Pclass = get_dummies(df['Pclass'], prefix='Pclass')
    df[list(dummies_Cabin.dtypes.index)] = dummies_Cabin
    df[list(dummies_Embarked.dtypes.index)] = dummies_Embarked
    df[list(dummies_Sex.dtypes.index)] = dummies_Sex
    df[list(dummies_Pclass.dtypes.index)] = dummies_Pclass
    return df