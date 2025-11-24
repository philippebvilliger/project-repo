import pandas as pd # you will import pandas dictionnary for data manipulation
from sklearn.preprocessing import StandardScaler 
# you will use the StandardScaler dictionnary to normalize numeric in order to be compatible with ML models

# ============================================================


def load_master_data(path="data/processed/transfers_matched_complete.csv"):
# This allows us to load the player-transfer dataset that we have created after combining both features.
# this removes rows with missing target values e.g., after-season performance
# We will then have a fully cleanded dataframe that can be used for train/test splitting

    
    df = pd.read_csv(path) # We read the combined file into a dataframe

    
    df = df[df["after_GA_per_90"].notna()]
    # As mentioned, we only keep rows where the AFTER-season GA_per_90 exists

    return df # We now have the new dataframe with no missing data


# ============================================================


def split_train_test(df): # We define a function that splits the dataset for testing
    # Remember, we train it on old transfers i.e., (2017-2022)
    # And we test it on newest trasnfers i.e., (2023-2024)
    

   
    train = df[df["transfer_year"] <= 2022]
    # So we select train as the rows where the transfer year is 2022 and below

    
    test = df[df["transfer_year"] >= 2023]
    # So we select test as the rows where the transfer year is 2023 and beyond

    
    target = "after_GA_per_90"
    # target is the name given to the performance metric we seek to predict 
    # i.e, goals and assets per 90 minutes after the transfer

    
    y_train = train[target]
    y_test = test[target]
    # We now extract this target column from the training set as well as the testing set

    X_train = train.drop(columns=[target])
    X_test = test.drop(columns=[target])
    # We define a new training and testing dataset such that the target column is now removed from them
    # Remember a ML model shouldn't have access to the value it's trying to predict.
    # If we left it, the model would already know the expected output and would just reproduce it directly without evaluating the relationships between variables

    return X_train, X_test, y_train, y_test
    # This returns the 4 datasets of interest.
    # 2 training datasets one with the expected predicted value and one without
    # 2 testing datasets one with the expected predicated value and one without
    # We create x and y (the expected answer) because the whole purpose of the model is to learn how to get y from x


# ============================================================


def prepare_features(X_train, X_test):
     # ML Models only numeric data so this function's purpose will be to prepare the dataframe for Machine Learning
     # It will first drop all non_numeric columns
     # It will then normalize the numeric data in order to be compatible with ML models
     # Important to note that the data values have vastly different scales e.g., transfer sums are in millions while goals in units
   

    
    non_numeric = X_train.select_dtypes(include=["object"]).columns
    # This finds all columns that contain text instead of numbers

    X_train = X_train.drop(columns=non_numeric)
    X_test = X_test.drop(columns=non_numeric)
    # We drop all non numeric columns


    scaler = StandardScaler()
    # scaler is an object of the StandardScaler class that will allow us to use functions from this class

    X_train_scaled = scaler.fit_transform(X_train)
    # fit_transform is the combination of the fit() function that computes the mean and standard deviation of each column
    # and transform() allows us to normalize the data based on what we've just estimated
    # This function belongs to the StandardScaler class hence the use of scaler here

    X_test_scaled = scaler.transform(X_test)
    # Remember to only use the fit function on the training and not testing because we don't want the ML Model to learn anything from the testing dataset.
    # The Scaler is using the mean and standard deviations from the training set to normalize the testing set
    # It's applying the scaling it learned on the testing dataset which is accurate with reality

    X_train_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    # We make new dataframes for our newly normalized/updated data

    return X_train_df, X_test_df
    # We return these two datframes

