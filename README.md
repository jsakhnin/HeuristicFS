# HeuristicFS
Heuristic optimization algorithms used for feature selection to reduce the dimensions of training data on classifiers. The application is attack detection in smart power systems.

## Data
The data is generated using MATLAB and matpower library following the mathematical rules of power flow calculations and False Data Injection (FDI) attacks.
The corresponding labels are binary, with 0 being normal and 1 representing attacked samples.

## Experiment
The experiment begins with running the svm, knn, and ann optimization scripts which find the optimal parameters for these classifiers. The test is performed on the smallest power system, the IEEE 14-bus.

Once ideal parameters for classifiers are found, the main script can be ran to test the feature selection algorithms on all three classifiers.

The results of this experiment can be found in output, and the publication can be found the the paper folder.
