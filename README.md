**Round 1 - Baseline Model**

Basic Feature Engineering
1. Random Forest ['Age', 'Fare', 'Parch', 'SibSp', 'Pclass'] to fit the missing age value
2. Missing value -> "Cabin"
3. One Hot Encoding for ['Cabin', 'Embarked', 'Sex', 'Pclass']
4. New feature "FamilySize" = "Parch" + "SibSp"
5. Standard Scaling -> "Age", "Fare"