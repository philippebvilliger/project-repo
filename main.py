import pandas as pd  # We import pandas library for data manipulation
from sklearn.model_selection import train_test_split  # We import train_test_split that will allow us to split the dataset into a training set and a testing set

from modeling import (
    train_linear_regression,
    train_random_forest,
    train_gradient_boosting
)
# In the modeling.py file we defined functions for each ML model that returned the model, the train score ... we are importing these here.

from evaluation import evaluate_model, print_evaluation_results
# We import these two functions from the evaluation.py file that allow us to evaluate a regression model using R².

# ============================================================

df = pd.read_csv("data/fbref/processed/transfers_matched_complete.csv")
# We are now loading the final_dataset csv file i.e., the final merged dataset containing all matched players with their respective performance and transfer statistics
# We load this merged dataset into a dataframe df
# You can replace the file path with your own CSV file as long as it has the target variable i.e., after_GA_per_90 and a set of features such as minutes_played, goals, assists, xG, xA ...

# ============================================================

TARGET_COLUMN = "after_GA_per_90"
# We call this target column because this is the column you want to predict

X = df.drop(columns=[TARGET_COLUMN])
# X will designate the created dataframe without the target column thus containing only the information used to make the predictions not the result in itself obviously
y = df[TARGET_COLUMN]
# This is the target column i.e., what we are trying to predict
# By separating the inputs and the outputs, the ML model can now work properly

# ============================================================

# We choose 80% for training and 20% for testing.
# random_state ensures reproducibility.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
# train_test_split is a function that splits your dataset into two parts a training set and a testing set.
# As we know, X are the inputs and y is the target value i.e., the variable we want to predict.
# test_size = 0.2 means that we want the training set to be 80% of the data we have while the testing set only 20%.
# random_state=42 is an arbitrary random seed and it ensures the exact same random choices are made every time. It will always follow the same sequence of random choices.


