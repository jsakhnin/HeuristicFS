import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

###################    DATA         ######################
X = pd.read_csv('IEEE14Data10k.csv')
y = pd.read_csv('IEEE14Labels10k.csv')
Xt = pd.read_csv('IEEE14Data1K.csv')
Yt = pd.read_csv('IEEE14Labels1k.csv')

X = np.asarray(X)
y = np.asarray(y).ravel()
Xt = np.asarray(Xt)
Yt = np.asarray(Yt).ravel()

##################   TRAINING      #######################
# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-4,1e-3,1e-2,1e-1,1,10,1e2,1e3],
                     'C': [0.001,0.01,0.1,1,5,10,20,30,50,70,100,300,500,1000]}]

score = 'accuracy'

print("# Tuning hyper-parameters for %s" % score)
print()

clf = GridSearchCV(SVC(), tuned_parameters, cv=5,scoring=score,verbose=2)
clf.fit(X, y)

print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()
print("Grid scores on development set:")
print()
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
print()

print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
y_true, y_pred = Yt, clf.predict(Xt)
print(classification_report(y_true, y_pred))
print()


################   DATA OUTPUT     ###############
scores_df = pd.DataFrame(clf.cv_results_).sort_values(by='rank_test_score')
# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('output/SVMoptimization_IEEE14.xlsx', engine='xlsxwriter')

# Convert the dataframe to an XlsxWriter Excel object.
scores_df.to_excel(writer, sheet_name='Sheet1')

# Close the Pandas Excel writer and output the Excel file.
writer.save()
