# Wine-Quality-Prediction
Machine Learning model to predict the quality of wine using linear regression
This code appears to be a Python script that performs an analysis on a wine quality dataset ('winequality-red.csv'). The script uses various libraries, including NumPy, Matplotlib, Pandas, Seaborn, and scikit-learn, to explore the dataset, visualize data distributions, perform classification using logistic regression and an ensemble method (Extra Trees Classifier), and evaluate the model's performance. Let's break down the code step by step:

1. **Importing Libraries:** The code starts by importing necessary Python libraries, including NumPy, Matplotlib, Pandas, Seaborn, and specific modules and functions from scikit-learn. It also suppresses warnings using `filterwarnings(action='ignore')`.

2. **Loading Data:** The dataset is loaded from the 'winequality-red.csv' file into a Pandas DataFrame called 'wine'. A success message is printed.

3. **Data Exploration:**
   - `wine.head()`: Displays the first few rows of the dataset.
   - `wine.shape`: Prints the dimensions (number of rows and columns) of the dataset.
   - `wine.describe(include='all')`: Provides summary statistics for all columns, including mean, standard deviation, and quartile information.
   - `wine.isna().sum()`: Counts and prints the number of missing values in each column.
   - `wine.corr()`: Calculates and prints the correlation matrix between numerical columns.
   - `wine.groupby('quality').mean()`: Groups the data by 'quality' and calculates the mean values for each group.

4. **Data Visualization:**
   - Several count plots, bar plots, and density plots are created using Seaborn and Matplotlib to visualize the distribution of various features and the 'quality' target variable.
   - A KDE plot is used to visualize the distribution of 'quality' values greater than 2.
   - Histograms and box plots are generated to explore the distributions of various features.
   - A heatmap is created to visualize the correlation matrix.
   - A pairplot is generated to visualize pairwise relationships between numerical features.
   - A violin plot is used to visualize the relationship between 'quality' and 'alcohol' content.

5. **Data Preparation:**
   - A new binary column 'goodquality' is added to the DataFrame based on whether 'quality' is greater than or equal to 7 (1 if true, 0 if false).
   - The 'quality' and 'goodquality' columns are dropped from the DataFrame to create the feature matrix 'X' and target variable 'Y'.

6. **Feature Importance:**
   - An Extra Trees Classifier is trained on the features and target variable to calculate feature importances (`score`). The importance scores are printed.

7. **Data Splitting:**
   - The dataset is split into training and testing sets using `train_test_split()`. 70% of the data is used for training (`X_train` and `Y_train`), and 30% is used for testing (`X_test` and `Y_test`).

8. **Model Training and Prediction:**
   - A logistic regression model is created using `LogisticRegression()` and trained on the training data using `model.fit(X_train, Y_train)`.
   - Predictions are made on the test data using `model.predict(X_test)`.

9. **Model Evaluation:**
   - The accuracy of the model is calculated using `accuracy_score()` from scikit-learn.
   - A confusion matrix is generated using `confusion_matrix()` to evaluate the model's performance in classifying 'goodquality' wine samples.

In summary, this code performs an analysis of a wine quality dataset, explores the data, visualizes distributions, creates a binary classification task to predict 'goodquality' wine samples, trains a logistic regression model, and evaluates its performance using accuracy and a confusion matrix. It also explores feature importances using an Extra Trees Classifier.
