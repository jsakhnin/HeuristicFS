import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

###################    DATA         ######################
X = pd.read_csv('IEEE14Data10k.csv')
y = pd.read_csv('IEEE14Labels10k.csv')
Xt = pd.read_csv('IEEE14Data1k.csv')
Yt = pd.read_csv('IEEE14Labels1k.csv')

X = np.asarray(X)
y = np.asarray(y).ravel()
Xt = np.asarray(Xt)
Yt = np.asarray(Yt).ravel()

#################   VARIABLES     #########################
accuracyResults =[]
nResults = []
f1Results = []
nValues = range(1,200)

#################   TRAINING     #########################
for n in nValues:
    print("Neighbours: ", n)
    # Declare an of the KNN classifier class with the value with neighbors.
    knn = KNeighborsClassifier(n_neighbors=n)

    # Fit the model with training data and target values
    knn.fit(X,y)

    #Testing the KNN
    y_true, y_pred = Yt, knn.predict(Xt)
    #print(classification_report(y_true, y_pred))

    currentF1Score = f1_score(y_true, y_pred, average='weighted')
    currentAccuracy = knn.score(Xt,Yt)

    f1Results.append(currentF1Score)
    accuracyResults.append(currentAccuracy)
    nResults.append(n)

###############     SAVING DATA   ########################
# Create a Pandas dataframe from the data.
df = pd.DataFrame({'N Values': nResults, 'Accuracy': accuracyResults, 'F1 Score': f1Results})

# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('output/KNNoptimization_IEEE57.xlsx', engine='xlsxwriter')

# Convert the dataframe to an XlsxWriter Excel object.
df.to_excel(writer, sheet_name='Sheet1')

# Close the Pandas Excel writer and output the Excel file.
writer.save()
