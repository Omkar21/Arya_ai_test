As mentioned in the problem. We have to classify the binery selection depending upon "training_set" csv file.

This file contains around 3910 records with 57 features. For testing the trained model, we got the "test_set" file. This file containe around 691 lines of testing data with 57 features.

Approch:
1. Checked all lines from "training_set" to avoid any null value.
2. I have imported the training data with pandas libraries and stored the data in dataframe.
3. Then divided this data into the features set: X_train and trining data results in Y_train.
4. Scale down all the features from X_train.
5. There are 57 features given. To reduce the dimentions PCA is used.
6. By using principle component analysis, dimentions reduces to 35 components and I got the varience upto 0.793883.
7. Then to select the classification model, I have applied the K fold cross validation on following classifiers:
	Logistic Regression
	Naive Bayes
	Support Vector Classifier
	K-nearest Neighbors
	Random forest tree
	Decision tree
8. Main idea behind using the kFoldCrossValidation is to evaluate the skill of our machine learning models. I have divided the training set into 10 folds and 	then tested the models performence.
9. I got following accuracy results for K fold cross validation:
	Logistic Regression: 90.057%
	Naive Bayes: 83.601%
	Support Vector Classifier: 90.249%
	K-nearest Neighbors: 89.865%
	Random forest tree: 92.319%
	Decision tree: 87.053%
	
10. From K-for cross validatio, I have selected the "Random forest tree" classifier to train on training set.
11. With Random forest classifier I got 92.455% accuracy. We got around 723 correct results from 782 rows from validation test set.
12. Finally results are predicted and printed for the test data from "test_set" file.
	
	