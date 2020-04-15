import pandas as pd
import numpy as np
import math
from sklearn import svm
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
import binary_optimization as opt#import library
from geneticfs import GeneticFS
import time
import warnings
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once

### Choose IEEE power system to run test here
sysName = 'IEEE14'
# sysName = 'IEEE30'
# sysName = 'IEEE57'
###################    DATA         ######################
X = pd.read_csv('{}Data40k.csv'.format(sysName))
Y = pd.read_csv('{}Labels40k.csv'.format(sysName))
X_opt = pd.read_csv('{}Data1k.csv'.format(sysName))
Y_opt = pd.read_csv('{}Labels1k.csv'.format(sysName))
Xt = pd.read_csv('{}Data10k.csv'.format(sysName))
Yt = pd.read_csv('{}Labels10k.csv'.format(sysName))

X = np.asarray(X)
Y = np.asarray(Y).ravel()
X_opt = np.asarray(X_opt)
Y_opt = np.asarray(Y_opt).ravel()
Xt = np.asarray(Xt)
Yt = np.asarray(Yt).ravel()

##Important Variables
nFeatures = len(X[0])
indx_BCS = [ 0 , 3 , 6 , 7, 13, 14, 20, 26, 27, 29, 31]
indx_BPSO = [ 0  ,3  ,5 ,15 ,20 ,21 ,22, 29]
indx_GA = [0, 2, 5, 20, 21, 22, 24, 33]
M_BCS = math.ceil((len(indx_BCS)+2)/2)
M_BPSO = math.ceil((len(indx_BPSO)+2)/2)
M_GA = math.ceil((len(indx_GA)+2)/2)
M_NA = math.ceil((len(X[0])+2)/2)

print(M_BCS)
print(M_BPSO)
print(M_GA)
print(M_NA)

#################     FEATURE EXTRACTION   ##################
X_BCS = X[:  , indx_BCS]
Xt_BCS = Xt[:  , indx_BCS]
X_BPSO = X[:  , indx_BPSO]
Xt_BPSO = Xt[:  , indx_BPSO]
X_GA = X[:  , indx_GA]
Xt_GA = Xt[:  , indx_GA]

X_BCS = np.asarray(X_BCS)
X_BPSO = np.asarray(X_BPSO)
X_GA = np.asarray(X_GA)
Xt_BCS = np.asarray(Xt_BCS)
Xt_BPSO = np.asarray(Xt_BPSO)
Xt_GA = np.asarray(Xt_GA)

print("Feature Selection Completed. Size of Data:")
print(X_BCS.shape)
print(X_BPSO.shape)
print(X_GA.shape)
print(Xt_BCS.shape)
print(Xt_BPSO.shape)
print(Xt_GA.shape)

#############   TRAINING     ##########################
knn_NA = KNeighborsClassifier(n_neighbors=12)
knn_BCS = KNeighborsClassifier(n_neighbors=12)
knn_BPSO = KNeighborsClassifier(n_neighbors=12)
knn_GA = KNeighborsClassifier(n_neighbors=12)
ann_BCS = MLPClassifier(solver='lbfgs', alpha=1e-6, hidden_layer_sizes=(M_BCS,))
ann_BPSO = MLPClassifier(solver='lbfgs', alpha=1e-6, hidden_layer_sizes=(M_BPSO,))
ann_GA = MLPClassifier(solver='lbfgs', alpha=1e-6, hidden_layer_sizes=(M_GA,))
ann_NA = MLPClassifier(solver='lbfgs', alpha=1e-6, hidden_layer_sizes=(M_NA,))

knn_NA.fit(X,Y)
knn_BCS.fit(X_BCS,Y)
knn_BPSO.fit(X_BPSO,Y)
knn_GA.fit(X_GA,Y)
ann_NA.fit(X,Y)
ann_BCS.fit(X_BCS,Y)
ann_BPSO.fit(X_BPSO,Y)
ann_GA.fit(X_GA,Y)

##########     CALCULATING RESULTS    ###################
##Algorithms in order
algorithms = ['knn_NA','knn_BCS','knn_BPSO','knn_GA','ann_NA','ann_BCS','ann_BPSO','ann_GA']

##Accuracies
A1 = knn_NA.score(Xt,Yt)
A2 = knn_BCS.score(Xt_BCS,Yt)
A3 = knn_BPSO.score(Xt_BPSO,Yt)
A4 = knn_GA.score(Xt_GA,Yt)
A5 = ann_NA.score(Xt,Yt)
A6 = ann_BCS.score(Xt_BCS,Yt)
A7 = ann_BPSO.score(Xt_BPSO,Yt)
A8 = ann_GA.score(Xt_GA,Yt)
accuracies = [A1,A2,A3,A4,A5,A6,A7,A8]

##F1 Scores
y_true, y_pred = Yt, knn_NA.predict(Xt)
F1 = f1_score(y_true, y_pred, average='weighted')
y_true, y_pred = Yt, knn_BCS.predict(Xt_BCS)
F2 = f1_score(y_true, y_pred, average='weighted')
y_true, y_pred = Yt, knn_BPSO.predict(Xt_BPSO)
F3 = f1_score(y_true, y_pred, average='weighted')
y_true, y_pred = Yt, knn_GA.predict(Xt_GA)
F4 = f1_score(y_true, y_pred, average='weighted')
y_true, y_pred = Yt, ann_NA.predict(Xt)
F5 = f1_score(y_true, y_pred, average='weighted')
y_true, y_pred = Yt, ann_BCS.predict(Xt_BCS)
F6 = f1_score(y_true, y_pred, average='weighted')
y_true, y_pred = Yt, ann_BPSO.predict(Xt_BPSO)
F7 = f1_score(y_true, y_pred, average='weighted')
y_true, y_pred = Yt, ann_GA.predict(Xt_GA)
F8 = f1_score(y_true, y_pred, average='weighted')
f1Scores = [F1,F2,F3,F4,F5,F6,F7,F8]

numOfFeatures = [len(X[0]),len(X_BCS[0]),len(X_BPSO[0]),len(X_GA[0]), len(X[0]),len(X_BCS[0]),len(X_BPSO[0]),len(X_GA[0])]

###############     SAVING DATA   ########################
# Create a Pandas dataframe from the data.
df = pd.DataFrame({'Algorithms': algorithms, 'Accuracy': accuracies, 'F1 Score': f1Scores, 'Number of Features': numOfFeatures})

# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('results_{}.xlsx'.format(sysName), engine='xlsxwriter')

# Convert the dataframe to an XlsxWriter Excel object.
df.to_excel(writer, sheet_name='Sheet1')

# Close the Pandas Excel writer and output the Excel file.
writer.save()
