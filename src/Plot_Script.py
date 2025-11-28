import pandas as pd  # We import pandas for tabular data manipulation
import matplotlib.pyplot as plt  # We import matplotlib to help us create plots
from sklearn.model_selection import train_test_split  # We import train_test_split to split data into train and test sets 

from modeling import train_random_forest  
# We import our previously created function from modeling.py.

# =============================================

df = pd.read_csv("data/processed/transfers_matched_complete.csv")
# We load the merged matched dataset
