import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn import tree as tree2
import matplotlib.pyplot as plt

col_names = ['workclass', 'education', 'martial-status', 'occupation', 'relationship', 'race', 'sex', 'native-country',
             'income']
feature_names = ['workclass', 'education', 'martial-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
labels = ['income']

dfTrain = pd.read_csv('trained.csv', skipinitialspace=True)
dfTest = pd.read_csv('tested.csv', skipinitialspace=True)


def oneHotBind(original, feature_to_encode):
    dummies = pd.get_dummies(original[feature_to_encode])
    res = pd.concat([original, dummies], axis=1)
    res = res.drop(feature_to_encode, axis=1)
    return (res)


dfTrain = oneHotBind(dfTrain, ['workclass', 'education', 'martial-status', 'occupation', 'relationship', 'race', 'sex', 'native-country',
                         'income'])
dfTest = oneHotBind(dfTest, ['workclass', 'education', 'martial-status', 'occupation', 'relationship', 'race', 'sex', 'native-country',
                        'income'])

for attributes in dfTrain.keys():
    if attributes not in dfTest.keys():
        print("Adding missing feature {}".format(attributes))
        dfTest[attributes] = 0


country = dfTest.pop('native-country_Holand-Netherlands')
dfTest.drop(index= 100)
dfTest.insert(72, 'native-country_Holand-Netherlands', country)

dfTrain.to_csv('trained2.csv')
dfTest.to_csv('tested2.csv')

#split training data
X_train = dfTrain.iloc[:, 1:99]
Y_train = dfTrain.iloc[:, 99:101]

#split testing data
X_test = dfTest.iloc[:, 1:99]
Y_test = dfTest.iloc[:, 99:101]

tree = DecisionTreeClassifier(max_depth= 3, random_state=42)
tree.fit(X_train, Y_train)
predictions = tree.predict(X_test)

print("Accuracy: " + str(accuracy_score(Y_test, predictions)))
print(classification_report(Y_test, predictions))



