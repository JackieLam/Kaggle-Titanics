###Round 1 - Baseline Model

Basic Feature Engineering
1. Random Forest ['Age', 'Fare', 'Parch', 'SibSp', 'Pclass'] to fit the missing age value
2. Missing value -> "Cabin"
3. One Hot Encoding for ['Cabin', 'Embarked', 'Sex', 'Pclass']
4. New feature "FamilySize" = "Parch" + "SibSp"
5. Standard Scaling -> "Age", "Fare"

###Round 2 - Additional Feature Engineering
1. One Hot Encoding for "Parch", "Sibsp", "FamilySize"
-> cross validation slightly **increase from 81% to 82%**
-> Reason: (survived, num of Parch / Sibsp / FamilySize) not linear related
2. Add one feature IsAlone and do one hot encoding
-> slightly increase
3. Turn the "Age" and "Fare" to bucket
-> slightly decrease
4. Find out the title in the name and create a new feature "Title"
-> cross validation **increase from 82% to 84%**
5. Find out the number in the Cabin and see how it impact the scores
-> cross validation improved, test prediction not improved

***After round 2 test prediction increased from 76% to 78%***

