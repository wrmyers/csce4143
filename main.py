import pandas as pd
import numpy as np

dfTrain = pd.read_csv('adultdata.csv', skipinitialspace=True)
dfTest = pd.read_csv('adulttest.csv', skipinitialspace=True)

print(len(dfTrain.index))
print(len(dfTest.index))

dfTrain = dfTrain[(dfTrain.values != '?').all(axis=1)]
print(len(dfTrain.index))

dfTest = dfTest[(dfTest.values != '?').all(axis=1)]
print(len(dfTest.index))

dfTrain = dfTrain.drop(['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week'], axis=1)
dfTest = dfTest.drop(['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week'], axis=1)

dfTrain.to_csv('trained.csv')
dfTest.to_csv('tested.csv')

