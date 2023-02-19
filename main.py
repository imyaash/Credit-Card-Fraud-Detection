# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 17:37:40 2023

@author: imyaash-admin
"""

# Import required libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as tts, cross_val_score as cvs, GridSearchCV as gscv
from sklearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeClassifier as DTC
from imblearn.pipeline import make_pipeline as mp
from imblearn.over_sampling import RandomOverSampler as ROS
from sklearn.linear_model import LogisticRegression as LRM
from sklearn.ensemble import RandomForestClassifier as RFC, AdaBoostClassifier as ABC, GradientBoostingClassifier as GBC
from sklearn.metrics import make_scorer as ms, f1_score as f1, accuracy_score as acs, balanced_accuracy_score as bas, precision_score as ps, recall_score as rs, roc_auc_score as ras, confusion_matrix as cm, ConfusionMatrixDisplay as CMD, roc_curve as rc

# Read the creditcard.csv file into a pandas dataframe
df = pd.read_csv("Dataset/creditcard.csv")

# Print the shape of the dataframe and its summary statistics
df.shape
des = df.describe()
print(des)
print(df.info())

# Check for duplicate rows in the dataframe
print(df.duplicated().sum())

# Check for missing values in the dataframe and print the percentage of missing values in each column
if sum(df.isna().sum()) != 0:
    for i in df.columns:
        if df[i].isna().sum() > 0:
            nan = df[i].isna().sum()
            length = len(df[i])
            print("Column: ", i, "NaN Values: ", nan, "Column length: ", length, "Percentage NaN: ", round((nan / length) * 100, 2), "%")
else:
    print("No NaN in the DataFrame.")

# Plot the kernel density estimate for each column of the dataframe
for i in df.columns:
    sns.kdeplot(df[i])
    plt.show()

# Plot a heatmap of the correlation matrix of the dataframe
plt.figure(figsize = (16, 16))
sns.heatmap(df.corr(), annot = True, cmap = "coolwarm")
plt.show()

# Plot a boxplot for each column of the dataframe, after dropping the "Time" and "Amount" columns
plt.figure(figsize = (16, 16))
df.drop(["Time", "Amount"], axis = 1).boxplot()
plt.show()

# Plot a boxplot for each column of the dataframe, grouped by the "Class" column
for i in df.iloc[:, :-1].columns:
    sns.boxplot(df["Class"], df[i])
    plt.show()

# Plot a pie chart showing the proportion of non-fraudulent and fraudulent transactions in the dataframe
plt.pie(df["Class"].value_counts(), labels = ["Non-Fraudulent", "Fraudulent"], autopct = "%1.1f%%", startangle = 90)
plt.show()

# Splitting the dataset into independent and dependent variables, and further splitting the independent and dependent variables into training and testing datasets with a 70:30 split ratio
independent = df.drop("Class", axis = 1)
dependent = df["Class"]

trainX, testX, trainY, testY = tts(independent, dependent, train_size = 0.7, random_state = 1997)

# Calculating the ratio of non-fraudulent transactions to fraudulent transactions in the training and testing datasets
trainClassDist = trainY.value_counts()
trainClassRatio = trainClassDist[0] / trainClassDist[1]
print("Class Ratio in Train Set")
print(trainClassRatio)
testClassDist = testY.value_counts()
testClassRatio = testClassDist[0] / testClassDist[1]
print("Class Ratio in Test Set")
print(testClassRatio)

# Selecting features from the training and testing datasets using a decision tree classifier
featSel = SelectFromModel(DTC(random_state = 1997), prefit = False)
trainXSel = featSel.fit_transform(trainX, trainY)
testXSel = featSel.transform(testX)

# Defining a list of models to train and test on the dataset
models = [LRM(class_weight = "balanced", n_jobs = -1, random_state = 1997),
          # SVC(class_weight = "balanced", random_state = 9999),
          RFC(class_weight = "balanced", n_jobs = -1, random_state = 1997),
          ABC(random_state = 1997),
          GBC(random_state = 1997)]

# Function to train and test the models using cross-validation, and output the mean F1 score and standard deviation for each model
def crossValModels(models, X, y):
    modelName = []
    f1ScoreMean = []
    f1ScoreStd = []
    for model in models:
        pipe = mp(ROS(random_state = 1997), model)
        scores = cvs(pipe, X, y, cv = 5, scoring = ms(f1))
        modelName.append(type(model).__name__)
        f1ScoreMean.append(scores.mean())
        f1ScoreStd.append(scores.std())
        print(f"{type(model).__name__} CV F1-score: {scores.mean():.4f} +/- {scores.std():.4f}")
    crossValModelScores = pd.DataFrame({"ModelName": modelName, "MeanF1Score": f1ScoreMean, "F1ScoreStdDev": f1ScoreStd})
    print("Model Cross Validation F1-scores")
    print(crossValModelScores)
    return crossValModelScores

# Calling the function to train and test the models and store the results in a pandas dataframe
crossValScores = crossValModels(models, trainXSel, trainY)

"""
The cross validation results are shown in a pandas dataframe with three columns.
Each row of the dataframe corresponds to a classification model that was cross-validated using 5-fold cross-validation.
The columns provide the following information:

    ModelName: The name of the classification model that was tested.
    MeanF1Score: The mean F1-score of the model across the 5 folds of the cross-validation.
                The F1-score is a performance metric that combines precision and recall and is particularly useful for imbalanced datasets.
                The value of MeanF1Score indicates the overall performance of the model in terms of correctly predicting both the positive and negative classes.
    F1ScoreStdDev: The standard deviation of the F1-score across the 5 folds of the cross-validation.
                The value of F1ScoreStdDev indicates the variability of the performance of the model across different folds.

The cross-validation results show the mean F1-score and standard deviation of the F1-score for each of the four classification models
(Logistic Regression, Random Forest, AdaBoost, and Gradient Boosting) after performing oversampling to address the class imbalance issue.

In general, an F1-score of 1 indicates perfect precision and recall, while a score of 0 indicates poor performance.
Therefore, a higher F1-score is generally considered to be better.

Based on the results, the Random Forest classifier achieved the highest F1-score of 0.844082, which is significantly higher than the other models.
This indicates that the Random Forest model has a better balance between precision and recall in predicting the positive class (i.e., credit card fraud)
in the imbalanced dataset after oversampling.

The AdaBoost classifier achieved a relatively lower F1-score of 0.217858, which is lower than both the Random Forest and Gradient Boosting classifiers.
This indicates that the AdaBoost model may not be as effective in addressing the class imbalance issue and in correctly identifying the positive class in the dataset.

Overall, the results suggest that oversampling can be an effective technique for addressing class imbalance in imbalanced datasets,
and that the Random Forest classifier can be a particularly effective model for predicting credit card fraud in such datasets.
However, it is important to note that the performance of the models may depend on various factors, such as the specific dataset, the features, and the hyperparameters used.
"""

# Optimising hyperparameters for RandomForestClassifier Model using GridSearchCrossValidation
# Grid Searching for class_weight
rfc = RFC(n_jobs = -1, verbose = 1, random_state = 1997)
paramGrid = {
    "class_weight": [{0: 1, 1:550}, {0: 1, 1:560}, {0: 1, 1:570}, {0: 1, 1:580}, {0: 1, 1:590}]
    }
gridSearch = gscv(rfc, param_grid = paramGrid, cv = 5, scoring = "f1")
gridSearch.fit(trainXSel, trainY)
print(type(rfc).__name__)
print("Best Parameters: ", gridSearch.best_params_)
print("Best score: ", gridSearch.best_score_)

# Grid Searching for criterion
rfc2 = RFC(class_weight = {0:1, 1:550}, n_jobs = -1, verbose = 1, random_state = 1997)
paramGrid2 = {
    "criterion": ["gini", "entropy", "log_loss"]
    }
gridSearch2 = gscv(rfc2, param_grid = paramGrid2, cv = 5, scoring = "f1")
gridSearch2.fit(trainXSel, trainY)
print(type(rfc2).__name__)
print("Best Parameters: ", gridSearch2.best_params_)
print("Best score: ", gridSearch2.best_score_)

# Grid Searching for oob_score
rfc3 = RFC(class_weight = {0:1, 1:550}, criterion = "entropy", n_jobs = -1, verbose = 1, random_state = 1997)
paramGrid3 = {
    "oob_score": [True, False]
    }
gridSearch3 = gscv(rfc3, param_grid = paramGrid3, cv = 5, scoring = "f1")
gridSearch3.fit(trainXSel, trainY)
print(type(rfc3).__name__)
print("Best Parameters: ", gridSearch3.best_params_)
print("Best score: ", gridSearch3.best_score_)

# Grid Searching for ccp_alpha
rfc4 = RFC(class_weight = {0:1, 1:550}, criterion = "entropy", oob_score = True, n_jobs = -1, verbose = 1, random_state = 1997)
paramGrid4 = {
    "ccp_alpha": np.linspace(0.0, 1.0, 10)
    }
gridSearch4 = gscv(rfc4, param_grid = paramGrid4, cv = 5, scoring = "f1")
gridSearch4.fit(trainXSel, trainY)
print(type(rfc4).__name__)
print("Best Parameters: ", gridSearch4.best_params_)
print("Best score: ", gridSearch4.best_score_)

# Grid Searching for max_sample
rfc5 = RFC(class_weight = {0:1, 1:550}, criterion = "entropy", oob_score = True, ccp_alpha = 0.0, n_jobs = -1, verbose = 1, random_state = 1997)
paramGrid5 = {
    "max_samples": np.linspace(0.0, 1.0, 10)
    }
gridSearch5 = gscv(rfc5, param_grid = paramGrid5, cv = 5, scoring = "f1")
gridSearch5.fit(trainXSel, trainY)
print(type(rfc5).__name__)
print("Best Parameters: ", gridSearch5.best_params_)
print("Best score: ", gridSearch5.best_score_)

# Grid Searching for n_estimators
rfc6 = RFC(class_weight = {0:1, 1:550}, criterion = "entropy", oob_score = True, ccp_alpha = 0.0, n_jobs = -1, verbose = 1, random_state = 1997)
paramGrid6 = {
    "n_estimators": [50, 100, 200, 500, 800, 1000, 1250, 1500, 1750, 2000]
    }
gridSearch6 = gscv(rfc6, param_grid = paramGrid6, cv = 5, scoring = "f1")
gridSearch6.fit(trainXSel, trainY)
print(type(rfc6).__name__)
print("Best Parameters: ", gridSearch6.best_params_)
print("Best score: ", gridSearch6.best_score_)

# Initialise & fit RandomForestClassifier model using optmial hypermaters as identified by GridSearchCrossValidation
clf = RFC(n_estimators = 100, criterion = "entropy", oob_score = True, n_jobs = -1, random_state = 1997, verbose = 10, class_weight = {0:1, 1:550}, ccp_alpha = 0.0)
clf.fit(trainXSel, trainY)
# Making Perdiction
pred = clf.predict(testXSel)
# Computing classification metrics
accuracy = acs(testY, pred)
balancedAccuracy = bas(testY, pred)
precision = ps(testY, pred)
recall = rs(testY, pred)
f1Score = f1(testY, pred)
rocAUC = ras(testY, pred)
confusion = cm(testY, pred)
# Printing classification metrics
print("Accuracy Score: ", accuracy)
print("Balanced Accuracy Score: ", balancedAccuracy)
print("Precision Score: ", precision)
print("Recall Score: ", recall)
print("F1 Score: ", f1Score)
print("ROC AUC Score: ", rocAUC)
print("Confusion Matrix:")
print(confusion)

"""
The output shows the calculated values of these metrics for a specific model, which has a high accuracy score of 0.9995 but a low recall score of 0.7532,
indicating that the model may have trouble identifying true positive cases. The balanced accuracy score and ROC AUC score are both relatively high,
which suggests that the model is doing a good job of balancing the true positive rate with the true negative rate.
The confusion matrix provides a visual representation of the number of true positives, true negatives, false positives, and false negatives for the model's predictions.
"""

# Plotting confusion matrix
disp = CMD(confusion_matrix = confusion, display_labels = clf.classes_)
disp.plot()

# Computing and plotting roc_auc
fpr, tpr, thresholds = rc(testY, clf.predict_proba(testXSel)[:, 1])
rocauc = ras(testY, clf.predict_proba(testXSel)[:, 1])
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % rocauc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()
