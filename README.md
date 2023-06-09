# TSF_GRIP-Data-Science-Tasks

#*****Task 1: Prediction using Supervised Learning*****

**Description:-**

To Predict the percentage of a student based on the number of study hours.
The goal of this task is to predict the percentage of a student based on the number of study hours using supervised learning. We will be using a simple linear regression model for this prediction task, as it involves only two variables: the study hours and the corresponding percentage scores.

**To accomplish this task, we will follow the following steps:**
**	Data Loading:** We will start by loading the data required for training and testing the model. The dataset contains two columns: 'Hours' (representing the number of study hours) and 'Scores' (representing the corresponding percentage scores).
**	Data Visualization:** Next, we will visualize the data using a scatter plot to identify any correlation between the study hours and the percentage scores. This will help us determine whether a linear regression model is suitable for this prediction task.
**	Data Preparation:** Before training the model, we need to split the data into training and testing sets. The training set will be used to train the model, while the testing set will be used to evaluate its performance. We will use scikit-learn's train_test_split function for this purpose.
**	Model Training:** Once the data is prepared, we will train a linear regression model using the training set. The model will learn the relationship between the study hours and the percentage scores from the training data.
**	Prediction:** After training the model, we will use it to make predictions on the testing set. This will give us the predicted percentage scores based on the given study hours.
**	Model Evaluation:** To assess the performance of the model, we will calculate the mean absolute error (MAE) between the actual percentage scores and the predicted percentage scores. A lower MAE indicates a better-performing model.
**	Prediction for New Data:** Finally, we will use the trained model to predict the percentage of a student who studies for 9.25 hours a day. This will provide an estimate of the expected score for a student studying for that duration.
