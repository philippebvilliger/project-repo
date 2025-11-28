import os   # we import the operating system module in order to access files from different folders of our directory
import sys  # we import the sys module in order to import our own modules across different folders

# ============================================================

current_file_path = os.path.abspath(__file__)
# this variable will give us the full path to this current script i.e., .../src/LR_Coefficient_Plot_Script.py

PROJECT_ROOT = os.path.dirname(os.path.dirname(current_file_path))
# os.path.dirname() gives us the folder of our current file i.e., src
# os.path.dirname(src) will give us the folder of src i.e., project-repo. 

sys.path.append(PROJECT_ROOT)
# This function adds a folder to the possible import locations. So after you've added a folder there, any script can import modules from that folder.
# We add the project root i.e., project-repo/ to Python’s module search path.
# This will allow the script inside the src/ folder—to import modules located in the project root like modeling.py and other folders.
# So this allows imports such as "from modeling import train_linear_regression"


DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "transfers_matched_complete.csv")
# We call DATA_PATH the full path to the CSV file where we have all our merged data
# we get project-repo/data/processed/transfers_matched_complete.csv

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
# We call RESULTS_DIR the path to the results folder i.e., project-repo/results

os.makedirs(RESULTS_DIR, exist_ok=True)
# In order to prevent an error where we'd get No directory, we use makedirs() which creates the directory in case it doesn't exist

# ============================================================

import pandas as pd                     # We import pandas for data manipulation
import matplotlib.pyplot as plt         # We import matplotlib.pyplot to make plots
from sklearn.model_selection import train_test_split   # We import train_test_split to divide data into train/test groups
from modeling import train_linear_regression
# We import our own train_linear_regression function from modeling
# We can now freely access this thanks to the sys.path.append() 

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
    X, y, test_size=0.20, random_state=50
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

# ============================================================

coefficients = lr_model.coef_
# coef_ is an attribute automatically created after training. 
# It stores the coefficient learned for each input feature in the Linear Regression model.
# We name this variable coefficients

feature_names = X.columns
# feature_names will be the names of the different columns in our input features

sorted_idx = coefficients.argsort()[::-1]
# argsort()[::-1] this first sorts the positions in ascending order and then reverses it 
# as we want the highest value i.e., most important feature to be at the top of the plot

top_features = feature_names[sorted_idx][:15]
top_coeffs = coefficients[sorted_idx][:15]
# For readibility reasons, we limit the plot to 15 most important features and their respective values

# ============================================================

plt.figure(figsize=(10, 8))
# We create a blank canvas of width 10 and height 8

plt.barh(top_features, top_coeffs)
# This creates a horizontal bar chart as this is more appropriate than a vertical bar chart in this scenario.
# top_features will be the name of each bar i.e., the name of the different columns in the input features
# top_coeffs will be their respective coefficients illustrated as bars

plt.gca().invert_yaxis()
# matplotlib would put the first bar i.e., with lowest importance at the top of the chart and the last at the bottom however we want the inverse
# hence why we invert the order thanks to inver_yaxis()
# gca() returns the current Axes object this allows us to apply invert_yaxis() to it.

plt.title("Top 15 Coefficients (Linear Regression)")
plt.xlabel("Coefficient Value")
plt.ylabel("Feature")
# Here we are simply adding the title and the axes labels

plt.tight_layout()
# This basically tells matplotlib to make everything fit nicely so that nothing is squished

output_path = os.path.join(RESULTS_DIR, "LR_coefficients.png")
# This is the path where the output will be saved
plt.savefig(output_path, dpi=300)
# This saves our newly created plot to the output path we just determined.
# dpi meaning dots per inch is the resolution of the plot

print(f"Saved: {output_path}")
# Message displayed if all goes well
