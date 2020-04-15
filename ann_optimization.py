from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.svm import SVC
import pandas as pd


import pylab as pl
import csv
import numpy as np
from sklearn import svm
from sklearn.decomposition import PCA
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

###################    DATA         ######################
X = pd.read_csv('IEEE14Data10k.csv')
y = pd.read_csv('IEEE14Labels10k.csv')
Xt = pd.read_csv('IEEE14Data10k.csv')
Yt = pd.read_csv('IEEE14Labels10k.csv')

X = np.asarray(X)
y = np.asarray(y).ravel()
Xt = np.asarray(Xt)
Yt = np.asarray(Yt).ravel()

#################   TRAINING     #########################
alphaValues = [1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,10,100,1e3,1e4,1e5,1e6,1e7,1e8,1e9,1e10]
f1Results = []
accuracyResults = []

for a in alphaValues:
    print("Alpha: ", a)
    clf = MLPClassifier(solver='lbfgs', alpha=a, hidden_layer_sizes=(18,))
    clf.fit(X, y)

    #Testing
    y_true, y_pred = Yt, clf.predict(Xt)
    #print(classification_report(y_true, y_pred))

    currentF1Score = f1_score(y_true, y_pred, average='weighted')
    currentAccuracy = clf.score(Xt,Yt)

    f1Results.append(currentF1Score)
    accuracyResults.append(currentAccuracy)

###############     SAVING DATA   ########################
# Create a Pandas dataframe from the data.
df = pd.DataFrame({'Alpha Values': alphaValues, 'Accuracy': accuracyResults, 'F1 Score': f1Results})

# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('output/ANNoptimization_IEEE14.xlsx', engine='xlsxwriter')

# Convert the dataframe to an XlsxWriter Excel object.
df.to_excel(writer, sheet_name='Sheet1')

# Close the Pandas Excel writer and output the Excel file.
writer.save()
