# Credit-Card-Fraud-Detection
This is a Python script that analyses the credit card fraud dataset by performing pre-processing, feature selection, and classification using several models. The script first imports the necessary libraries and reads in the dataset using pandas. It then prints the shape and summary statistics of the dataset and checks for any missing values. If any missing values are found, the script prints the name of the column, the number of missing values, the length of the column, and the percentage of missing values.

![alt text](https://github.com/imyaash/Credit-Card-Fraud-Detection/blob/main/PlotsAndImages/1.%20dfShape.png)
![alt text](https://github.com/imyaash/Credit-Card-Fraud-Detection/blob/main/PlotsAndImages/2.%20dfDescribe.png)
![alt text](https://github.com/imyaash/Credit-Card-Fraud-Detection/blob/main/PlotsAndImages/3.%20dfInfo.png)

Next, the script generates kernel density estimation (KDE) plots, a heatmap of the correlation matrix, box plots, individual box plots for each feature against the class variable, and a pie chart to show the distribution of classes in the dataset.

![alt text](https://github.com/imyaash/Credit-Card-Fraud-Detection/blob/main/PlotsAndImages/4.%20KDEPlots.png)
![alt text](https://github.com/imyaash/Credit-Card-Fraud-Detection/blob/main/PlotsAndImages/5.%20CorrPlot.png)
![alt text](https://github.com/imyaash/Credit-Card-Fraud-Detection/blob/main/PlotsAndImages/6.%20Boxplots.png)
![alt text](https://github.com/imyaash/Credit-Card-Fraud-Detection/blob/main/PlotsAndImages/7.%20BoxplotsVsClass.png)
![alt text](https://github.com/imyaash/Credit-Card-Fraud-Detection/blob/main/PlotsAndImages/8.%20ClassPie.png)

The dataset is then split into training and testing sets, and the ratio of non-fraudulent to fraudulent transactions is printed for both sets. The script uses a decision tree classifier for feature selection, and the resulting transformed training and testing sets are stored in trainXSel and testXSel.

![alt text](https://github.com/imyaash/Credit-Card-Fraud-Detection/blob/main/PlotsAndImages/9.%20ClassRatio.png)

Finally, the script instantiates four classification models and passes them through a cross-validation pipeline. The cross-validation F1-scores are printed for each model, and a data frame of model names, mean F1-scores, and standard deviations of F1-scores is returned. The results show the mean F1-score and standard deviation of the F1-score for each of the four classification models (Logistic Regression, Random Forest, AdaBoost, and Gradient Boosting) after performing oversampling to address the class imbalance issue.

Based on the results, the Random Forest classifier achieved the highest F1-score of 0.844082, which is significantly higher than the other models. This indicates that the Random Forest model has a better balance between precision and recall in predicting the positive class (i.e., credit card fraud) in the imbalanced dataset after oversampling. The AdaBoost classifier achieved a relatively lower F1-score of 0.217858, which is lower than both the Random Forest and Gradient Boosting classifiers. This indicates that the AdaBoost model may not be as effective in addressing the class imbalance issue and in correctly identifying the positive class in the dataset.

The script uses the Random Forest Classifier (RFC) algorithm to build a predictive model for a binary classification problem. It uses GridSearchCV from Scikit-learn to perform hyperparameter tuning for the RFC algorithm, including the class_weight, criterion, oob_score, ccp_alpha, max_samples, and n_estimators. After hyperparameter tuning, the script fits the best model on the training set and evaluates its performance on the test set using several metrics such as accuracy, balanced accuracy, precision, recall, F1 score, ROC AUC, and confusion matrix. The output shows the values of these metrics for the best model.

![alt text](https://github.com/imyaash/Credit-Card-Fraud-Detection/blob/main/PlotsAndImages/10.%20ClassificationScores.png)
![alt text](https://github.com/imyaash/Credit-Card-Fraud-Detection/blob/main/PlotsAndImages/11.%20ConfusionMatrix.png)
![alt text](https://github.com/imyaash/Credit-Card-Fraud-Detection/blob/main/PlotsAndImages/12.%20ROCAUC.png)
