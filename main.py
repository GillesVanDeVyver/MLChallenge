from statistics import mean
import pandas_schema
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import  cross_validate
import pandas as pd
from pandas_schema import Column
from pandas_schema.validation import CustomElementValidation
import numpy as np

def check_decimal(dec):
    if dec == 'nan':
        return False
    try:
        dec = float(dec)
        if np.isnan(dec):
            return False
        if dec >10 or dec <-10:
            return False
    except :
        return False
    return True


def check_int(num):
    try:
        int(num)
    except ValueError:
        return False
    return True


decimal_validation = [CustomElementValidation(lambda d: check_decimal(d), 'is not decimal')]
int_validation = [CustomElementValidation(lambda i: check_int(i), 'is not integer')]
null_validation = [CustomElementValidation(lambda d: d is not np.nan, 'this field cannot be null')]
bool_validation = [CustomElementValidation(lambda d: d in ['False','True'], 'is not bool')]

schema = pandas_schema.Schema([
            Column('id', int_validation ),
            Column('y',  ),
            Column('x1',decimal_validation ),
            Column('x2',decimal_validation ),
            Column('x3',decimal_validation ),
            Column('x4', decimal_validation),
            Column('x5', bool_validation ),
            Column('x6', ),
            Column('x7', decimal_validation ),
            Column('x8',decimal_validation ),
            Column('x9',decimal_validation),
            Column('x10', decimal_validation)])

trainData = pd.read_csv('TrainOnMe.csv')
errors = schema.validate(trainData)
errors_index_rows = [e.row for e in errors]
data_clean = trainData.drop(index=errors_index_rows)
pd.DataFrame({'col':errors}).to_csv('errors.csv')
data_clean.to_csv('clean_data.csv')

def preProcess(data) :
    boolToInt = {True: 1, False: 0}
    data['x5'] = data['x5'].replace(boolToInt)
    gradeToInt = {"F": 1, "Fx": 1.5, "E": 2, "D": 3, "C": 4, "B": 5, "A": 6}
    data['x6'] = data['x6'].replace(gradeToInt)
    feature_cols = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10']
    X = data[feature_cols]
    X_scaled = preprocessing.scale(X)
    return X_scaled


cleanTest = pd.read_csv('clean_data.csv')
trainX = preProcess(cleanTest)
trainY = cleanTest['y']

clf = GradientBoostingClassifier(max_depth=3, n_estimators=64, learning_rate=0.129, min_samples_split=20,
                                 min_samples_leaf=5)
clf.fit(trainX,trainY)

cleanEval = pd.read_csv('EvaluateOnMe.csv')
testX = preProcess(cleanEval)
predictions = clf.predict(testX)

f = open("104695.txt", "w")
for str in predictions:
    f.write(str)
    f.write('\n')
f.close()


#The following code is the core part used for hyperparameter tuning
"""
scores = cross_validate(clf, trainX, trainY, cv=10, return_train_score=True)
score =mean(scores['test_score'])
print("avg validation score :",mean(scores['test_score']))
print("avg train score:",mean(scores['train_score']))
"""