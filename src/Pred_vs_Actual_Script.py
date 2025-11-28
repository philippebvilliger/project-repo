import os   # we import the operating system module in order to access files from different folders of our directory
import sys  # we import the sys module in order to import our own modules across different folders


# ============================================================


current_file_path = os.path.abspath(__file__)
# this variable will give us the full path to this current script i.e., .../src/RF_Plot_Script.py

PROJECT_ROOT = os.path.dirname(os.path.dirname(current_file_path))
# os.path.dirname() gives us the folder of our current file i.e., src
# os.path.dirname(src) will give us the folder of src i.e., project-repo. So this is the PROJECT_ROOT

sys.path.append(PROJECT_ROOT)
# This function adds a folder to the possible import locations. So after you've added a folder there, any script can import modules from that folder.
# We add the project root i.e., project-repo/ to Python’s module search path.
# This will allow the script inside the src/ folder—to import modules located in the project root like modeling.py and other folders.
# So this allows imports such as "from modeling import train_random_forest"

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "transfers_matched_complete.csv")
# We call DATA_PATH the full path to the CSV file where we have all our merged data
# we get project-repo/data/processed/transfers_matched_complete.csv

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
# We call RESULTS_DIR the path to the results results folder i.e., project-repo/results

os.makedirs(RESULTS_DIR, exist_ok=True)
# In order to prevent an error where we'd get No directory, we use makedirs() which creates the directory in case it doesn't exist

# ============================================================

import pandas as pd   # We import pandas for data manipulation
import matplotlib.pyplot as plt  # We import matplotlib.pyplot to make plots
from sklearn.model_selection import train_test_split  # We import train_test_split to divide data into train/test groups

from modeling import (
    train_linear_regression,
    train_random_forest,
    train_gradient_boosting
)
# We import our own train_linear_regression, train_random_forest and train_gradient_boosting functions from modeling
# We can now freely access these thanks to the sys.path.append() 

# ============================================================


df = pd.read_csv(DATA_PATH)
# We read the csv file thanks to the path created earlier


df = df.loc[:, ~df.columns.str.contains("Unnamed")]
# We now know that when we export csv files, these may create useless columns like "Unnamed ..."
# df.loc[] allows us to keep all columns that don't contain "Unnamed" thanks to "~"

df = df[(df["after_G+A"].notna()) & (df["before_G+A"].notna())]
# After merging the file we know that it's very common to have missing values. So likely that either after_G+A or before_G+A is missing
# So we want to keep only rows where both are present
# Remember that if you want your model to learn it needs inputs and outputs

before_cols = [c for c in df.columns if c.startswith("before_")]
# We select all columns that start with "before_" to access features before the transfer

before_numeric = [c for c in before_cols if df[c].dtype != "object"]
# Remember that only numeric variables are of interest here we neglect factors such as nationality 

df = df.dropna(subset=before_numeric)
# We obviously discard any rows containing missing values among the columns of interest as ML training is impossible on missing data

X = df[before_numeric]   
# The variable X represents our input features for the model's training i.e., before-transfer stats

y = df["after_G+A"]     
# y represents the target values we want to predict so after-transfer stats


# ============================================================


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
# We use the train_test_split() function to split the dataset into training and test sets
# As we know, X are the inputs and y is the target value i.e., the variable we want to predict.
# test_size = 0.2 means that we want the training set to be 80% of the data we have while the testing set only 20%.
# random_state=50 is an arbitrary random seed and it ensures the exact same random choices are made every time. It will always follow the same sequence of random choices.
# This function returns the training inputs i.e., X_train this is 80% of the data of X
# It will also return the testing inputs i.e., X_test which is 20% of the data of X
# y_train which are the target values for each row of X_train i.e., after_GA_per_90
# and finally, y_test which are the target values for each row of X_test i.e., after_GA_per_90

# ============================================================


lr_model, lr_train_r2, lr_test_r2, lr_pred = train_linear_regression(
    X_train, y_train, X_test, y_test
)
# We now use our train_linear_regression() that we imported from the modeling.py file.
# The inputs are the recently obtained training inputs, testing inputs, target values for each row of X_train and the target values for each row of X_test
# We return the ML model that was used so here LinearRegression(), the train_score i.e., the R² score for the training dataset
# test_score i.e., the R² score for the testing dataset and y_pred_test, the predicted after_GA_per_90 for each player

rf_model, rf_train_r2, rf_test_r2, rf_pred = train_random_forest(
    X_train, y_train, X_test, y_test
)
# We now use our train_random_forest() that we imported from the modeling.py file.
# The inputs are the recently obtained training inputs, testing inputs, target values for each row of X_train and the target values for each row of X_test
# We return the ML model that was used so here RandomForestRegression(), the train_score i.e., the R² score for the training dataset
# test_score i.e., the R² score for the testing dataset and y_pred_test, the predicted after_GA_per_90 for each player

gb_model, gb_train_r2, gb_test_r2, gb_pred = train_gradient_boosting(
    X_train, y_train, X_test, y_test
)
# We now use our train_gradient_boosting() that we imported from the modeling.py file.
# The inputs are the recently obtained training inputs, testing inputs, target values for each row of X_train and the target values for each row of X_test
# We return the ML model that was used so here GradientBoostingRegression(), the train_score i.e., the R² score for the training dataset
# test_score i.e., the R² score for the testing dataset and y_pred_test, the predicted after_GA_per_90 for each player

# ============================================================

# We want to write one script that generates the predicted vs actual values for each of the three ML models
# In order to do this we will have to define a specific function called plot_pred_vs_actual()

def plot_pred_vs_actual(y_true, y_pred, model_name, output_name):

    # The purpose of this function is to create a predicted vs actual scatter plot for a given ML model based on the inputs
    # y_true is the true target values (y_test). These are the real after_G+A outputs from the test set i.e., the actual values the model tries to predict    
    # y_pred are the predicted values generated by the ML models. These are the estimates produced by each ML model after training.
    # The third input is simply the model name that will be used for the title of the plot
    # The output_name will be the name of the newly saved plot

    plt.figure(figsize=(8, 6))
    # We create a blank canvas of width 8 and height 6


    plt.scatter(y_true, y_pred, alpha=0.6)
    # In order to have each point represent a player we select y_true, 
    # the true target values as the x-axis and the y_pred, the predicted values for a given model as the y-axis.
    # alphas is just to determine how opaque each individual point is so, 0.6 makes them slightly see-through.

    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--")
    # plt.plot() is used to draw the perfect prediction line. This line shows what predictions
    # would look like if the model were perfect, i.e., y = x.
    # The first interval sets the minimum and maximum values on the x-axis.
    # The second interval sets the corresponding minimum and maximum values on the y-axis 
    # which creates the diagonal y = x line and makes it span the full range of our actual values.

    plt.title(f"Predicted vs Actual ({model_name})")
    plt.xlabel("Actual after_G+A")
    plt.ylabel("Predicted after_G+A")
    # Hree we are simply adding the title and axes labels

    plt.tight_layout()
    # This basically tells matplotlib to make everything fit nicely so that nothing is squished


    full_output_path = os.path.join(RESULTS_DIR, output_name)
    # This line creates the full path where the plot will be saved i.e., project-repo/results/output_name
    plt.savefig(full_output_path, dpi=300)
    # This saves our newly created plot to the output path we just determined.
    # dpi meaning dots per inch is the resolution of the plot

    print(f"Saved: {full_output_path}")
    # Message displayed if all goes well


# ============================================================


plot_pred_vs_actual(y_test, lr_pred, "Linear Regression", "LR_pred_vs_actual.png")
plot_pred_vs_actual(y_test, rf_pred, "Random Forest", "RF_pred_vs_actual.png")
plot_pred_vs_actual(y_test, gb_pred, "Gradient Boosting", "GB_pred_vs_actual.png")
# for each ML model, we execute the plot_pred_vs_actual() function with the appropriate inputs
