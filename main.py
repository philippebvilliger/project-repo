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

df = pd.read_csv("data/processed/transfers_matched_complete.csv")
# We are now loading the final_dataset csv file i.e., the final merged dataset containing all matched players with their respective performance and transfer statistics
# We load this merged dataset into a dataframe df
# You can replace the file path with your own CSV file as long as it has the target variable i.e., after_GA_per_90 and a set of features such as minutes_played, goals, assists, xG, xA ...


# ============================================================

TARGET_COLUMN = "after_G+A"
# We call this target column because this is the column you want to predict

X = df.drop(columns=[TARGET_COLUMN])
# X will designate the created dataframe without the target column thus containing only the information used to make the predictions not the result in itself obviously
X = X.select_dtypes(include=['number'])
# We do this in order to keep solely the numeric columns
y = df[TARGET_COLUMN]
# This is the target column i.e., what we are trying to predict
# By separating the inputs and the outputs, the ML model can now work properly

# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
# train_test_split is a function that splits your dataset into two parts a training set and a testing set.
# As we know, X are the inputs and y is the target value i.e., the variable we want to predict.
# test_size = 0.2 means that we want the training set to be 80% of the data we have while the testing set only 20%.
# random_state=42 is an arbitrary random seed and it ensures the exact same random choices are made every time. It will always follow the same sequence of random choices.
# This function returns the training inputs i.e., X_train this is 80% of the data of X
# It will also return the testing inputs i.e., X_test which is 20% of the data of X
# y_train which are the target values for each row of X_train i.e., after_GA_per_90
# and finally, y_test which are the target values for each row of X_test i.e., after_GA_per_90

# ============================================================

print("Training Linear Regression")
linear_model, linear_train_score, linear_test_score, linear_y_pred_test = train_linear_regression(X_train, y_train, X_test, y_test)
# With the variables we have just obtained thanks to the train_test_split function, we can now input them into the train_linear_regression function that we imported from our modeling.py file.
# This returns the ML model that was used so here LinearRegression(), the train_score i.e., the R² score for the training dataset,
# test_score i.e., the R² score for the testing dataset and y_pred_test, the predicted after_GA_per_90 for each player

print("Training Random Forest")
rf_model, rf_train_score, rf_test_score, rf_y_pred_test = train_random_forest(X_train, y_train, X_test, y_test)
# With the variables we have just obtained thanks to the train_test_split function, we can now input them into the train_random_forest function that we imported from our modeling.py file.
# This returns the ML model that was used so here RandomForestRegressor(), the train_score i.e., the R² score for the training dataset,
# test_score i.e., the R² score for the testing dataset and y_pred_test, the predicted after_GA_per_90 for each player

print("Training Gradient Boosting")
gb_model, gb_train_score, gb_test_score, gb_y_pred_test = train_gradient_boosting(X_train, y_train, X_test, y_test)
# With the variables we have just obtained thanks to the train_test_split function, we can now input them into the train_gradient_boosting function that we imported from our modeling.py file.
# This returns the ML model that was used so here GradientBoostingRegressor(), the train_score i.e., the R² score for the training dataset,
# test_score i.e., the R² score for the testing dataset and y_pred_test, the predicted after_GA_per_90 for each player

# ============================================================

r2_linear = evaluate_model(linear_model, X_test, y_test)
# We use the evaluate_model function that we import from our evaluation.py file
# The first input is the model so here linear_model that we have just generated thanks to the train_linear_regression function
# The second input contains the test features used by the model to make predictions and the third input contains the true after_GA_per_90 values for the test set.    
# This will return the R² score i.e., how well this model explains the variation in after_GA_per_90
print_evaluation_results("Linear Regression", r2_linear)
# We import the print function from the evaluation.py file in order to neatly display the results obtained

r2_rf = evaluate_model(rf_model, X_test, y_test)
# We use the evaluate_model function that we import from our evaluation.py file
# The first input is the model so here rf_model that we have just generated thanks to the train_random_forest function
# The second input contains the test features used by the model to make predictions and the third input contains the true after_GA_per_90 values for the test set.    
# This will return the R² score i.e., how well this model explains the variation in after_GA_per_90
print_evaluation_results("Random Forest", r2_rf)
# We import the print function from the evaluation.py file in order to neatly display the results obtained

r2_gb = evaluate_model(gb_model, X_test, y_test)
# We use the evaluate_model function that we import from our evaluation.py file
# The first input is the model so here gb_model that we have just generated thanks to the train_gradient_boosting function
# The second input contains the test features used by the model to make predictions and the third input contains the true after_GA_per_90 values for the test set.    
# This will return the R² score i.e., how well this model explains the variation in after_GA_per_90
print_evaluation_results("Gradient Boosting", r2_gb)
# We import the print function from the evaluation.py file in order to neatly display the results obtained

# ============================================================

print("\nAll models have been trained and evaluated successfully!")

