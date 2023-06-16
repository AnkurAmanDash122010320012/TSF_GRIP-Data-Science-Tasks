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

#*****Task 2: Prediction using Unsupervised Learning*****

**Description:-**

In this unsupervised machine learning task, the objective is to apply K-means clustering to the given 'Iris' dataset and determine the optimum number of clusters.

To accomplish this task of predicting the optimum number of clusters and visually representing them using the Iris dataset, we will follow the following steps:
•**Setup-Importing necessary libraries:** In this step, we import the necessary libraries such as pandas, numpy, matplotlib, and sklearn. These libraries provide the required functions and tools for data manipulation, visualization, and unsupervised machine learning.
•**Identifying the most suitable number of clusters through the implementation of the elbow method:** Here, we employ the Elbow Method to find the optimal number of clusters for our dataset. This method involves fitting the K-means clustering algorithm to the data for a range of cluster numbers and calculating the sum of squared distances between each data point and its centroid. We plot these values and identify the "elbow" point, which indicates the number of clusters where the rate of decrease in the sum of squared distances significantly slows down. This will help us determine the optimal number of clusters to use for the K-means algorithm.
•**Constructing the KMeans classifier:** Once we have determined the optimal number of clusters, we proceed to create the KMeans classifier using the sklearn library. We fit the classifier to the dataset and obtain the predicted labels for each data point.
•**Visualizing the Clusters:** To visualize the clustering results, we plot a scatter plot of the data points, where each point is colored based on its predicted cluster. Additionally, we plot the cluster centers as black points. By using different colors for each iris species, we can observe the separation and distribution of the clusters in the feature space.
•**Assigning labels to the predicted clusters:** In this step, we assign labels to the predicted clusters based on the majority iris species in each cluster. By comparing the predicted labels with the actual species labels, we determine the corresponding iris species for each predicted cluster.
•**Incorporating the forecasted results into the dataset:** We append the predicted class labels to the original Iris dataset as a new column. This allows us to analyze the distribution of the iris species within the clusters and gain insights into the clustering results.
•**Histogram Plot-Cluster Distribution:** We generate a histogram to visualize the distribution of the predicted clusters for each iris species. By using different colors for each iris species in the histogram bars, we can observe the relative abundance of each species within the clusters.
•**Pair Plot:** The pair plot provides a matrix of scatter plots that showcase the relationships between different features in the dataset. By coloring the data points based on their predicted clusters and using different colors for each iris species, we can analyze how well the features separate the iris species and understand their distribution within the clusters.
•**Tree Plot:** We generate a dendrogram, also known as a tree plot, to visualize the hierarchical clustering structure of the dataset. By using different colors for each iris species, we can observe how the samples group together and form clusters based on their similarities. This plot helps us understand the organization of the data and the relationships between different iris species.

#*****Task 3: Exploratory Data Analysis - Retail*****

**Description:-**

	Exploratory Data Analysis (EDA) is a crucial step in the data analysis process, especially for business managers aiming to uncover insights and identify areas for improvement in their operations. In this particular analysis on the 'SampleSuperstore.csv' dataset, the objective is to identify weak areas in the retail business where profit can be increased.
	The initial steps involve loading and understanding the dataset, which contains information on various aspects of the superstore's operations, including sales, profits, discounts, regions, categories, and more. By performing EDA using Python in Google Colab, we can extract meaningful insights to address business problems.
	The analysis begins by examining the data based on different categories, regions, and segments. Visualizations such as pair plots, heatmaps, count plots, and distribution plots are used to explore relationships, correlations, and trends in the data. This provides a comprehensive understanding of the dataset and allows us to identify patterns and potential areas for improvement.
	Furthermore, the analysis dives into country, state, and city-wise deals, analyzing the profitability, discounts, and sales in each region. This helps identify specific regions or cities where attention should be focused to maximize profits and optimize operations.
	The quantity-wise analysis reveals insights into the impact of quantity on sales, profits, and discounts. By understanding the relationship between these variables, business managers can make informed decisions regarding inventory management, pricing strategies, and discount offers.
	Finally, the analysis concludes with a summary of the results and key findings. It highlights the weak areas in the business, such as regions or categories with low profits or excessive discounts, and provides recommendations for improvement.
	Overall, through exploratory data analysis, business managers can gain valuable insights into their retail operations, identify business problems, and make data-driven decisions to increase profitability and improve overall performance.
