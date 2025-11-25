import pandas as pd  #  We import the pandas library for data manipulation
from sklearn.linear_model import LinearRegression   # We import our first model from sklearn's linear_model module
from sklearn.ensemble import RandomForestRegressor  # We import our second model from sklearn's ensemble module
from sklearn.ensemble import GradientBoostingRegressor  # We import our second model from sklearn's ensemble module
from sklearn.metrics import r2_score  # We import the r2 in order to be able to the evaluate ML performance from the metrics module

# ============================================================

def train_linear_regression(X_train, y_train, X_test, y_test):
    # We define a function that trains the linear regression on the training data.

    model = LinearRegression()  
    # model will be the name of the object of the LinearRegression class
    # This object will allow us to make the best linear equation between x and y .

    model.fit(X_train, y_train)
    # The model learns the optimal coefficients such that X_train explain y_train 
    # It finds the best linear equation that predicts after_GA_per_90 the target, we seek to predict.

    y_pred_train = model.predict(X_train) 
    y_pred_test = model.predict(X_test)
    # predict() takes input features x and generates the model's estimated output based on what it learned during training
    # So here this concretely means that we will have one predicted value of after_GA_per_90 per player
    # We do this for the training and testing datasets

    train_score = r2_score(y_train, y_pred_train)
    test_score = r2_score(y_test, y_pred_test)
    # Remember that we use R² to measure predictive power.
    # This measures accuracy: 1.0 = perfect, 0 = bad, negative = very bad.
    # E.g., if R² = 1.0, then, the model explains 100% of the variation in after_GA_per_90.
    # The r2_score function compares the real values of after_GA_per_90 from the dataset with the predicted values produced from the model as inputs

    return model, train_score, test_score, y_pred_test
    # We return the ML model that was used so here LinearRegression(), the train_score i.e., the R² score for the training dataset
    # test_score i.e., the R² score for the testing dataset and y_pred_test, the predicted after_GA_per_90 for each player


# ============================================================


def train_random_forest(X_train, y_train, X_test, y_test):
    # We define a  function that trains the random forest Model on the training data.

    model = RandomForestRegressor(
        n_estimators=300,       # We select 300 trees here because it gives excellent performance with low risk of overfitting
        max_depth=None,         # This means there's no maximum depth for a given tree. It can split forever until every leaf has players with identical after_GA_per_90
                                # We can't be more precise in the prediction based on the information we have
        random_state=50,        # 50 is an arbitrary random seed and it ensures the exact same random choices are made every time
                                # So this will always follow the same sequence of random choices
        n_jobs=-1               # This just ensures the model runs as fast as possible
    )
    # model will be the name of the object of the RandomForestRegressor class
    # The principle of the Random Forest model is to generate many trees where each tree predicts the target variable i.e., the player's after_GA_per_90 here.
    # The model will then average all the trees predictions in order to generate one final prediction


    model.fit(X_train, y_train)
    # fit() is the phase where the model builds its understanding of the data
    # So the model analyzes X_train the input as well as y_train the real values
    # It builds many different decision trees and combines their predictions to estimate after_GA_per_90

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    # predict() takes input features x and generates the model's estimated output based on what it learned during training
    # So here this concretely means that we will have one predicted value of after_GA_per_90 per player
    # We do this for the training and testing datasets

    train_score = r2_score(y_train, y_pred_train)
    test_score = r2_score(y_test, y_pred_test)
    # The r2_score function compares the real values of after_GA_per_90 from the dataset with the predicted values produced from the model as inputs

    return model, train_score, test_score, y_pred_test
    # We return the ML model that was used so here RandomForestRegression(), the train_score i.e., the R² score for the training dataset
    # test_score i.e., the R² score for the testing dataset and y_pred_test, the predicted after_GA_per_90 for each player


# ============================================================


def train_gradient_boosting(X_train, y_train, X_test, y_test):
    # We define a train function for the gradient boosting ML Model


    model = GradientBoostingRegressor(
        n_estimators=300,        # We select 300 sequential tree corrections, which improves accuracy while keeping overfitting under control as each tree is shallow (low max_depth)
        learning_rate=0.05,      # This controls how much each new tree is allowed to correct the errors of the previous ones.
                                 # We use 0.05 here because it's small which makes the model learn more slowly and carefully ultimately reducing overfitting
        max_depth=3,             # This controls how complex each tree is. 3 is small and ideal as many small trees added together can learn relationships without overfitting
        random_state=70          # 70 is an arbitrary random seed and it ensures the exact same random choices are made every time
                                 # So this will always follow the same sequence of random choices
    )
    # model will be the name of the object of the GradientBoostingRegressor class
    # Gradient Boosting builds small trees one after another and each tree corrects the errors of the previous one.
    # This allows the model to become very accurate at predicting the target variable i.e., after_GA_per_90


    model.fit(X_train, y_train)
    # fit() is the phase where the model builds its understanding of the data
    # So the model analyzes X_train the input as well as y_train the real values
    # It builds many small trees one after another, and each new tree corrects the errors made by the previous ones. 
    # Eventually these sequential corrections allow the model to accurately predict the target i.e., after_GA_per_90

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    # predict() takes input features x and generates the model's estimated output based on what it learned during training
    # So here this concretely means that we will have one predicted value of after_GA_per_90 per player
    # We do this for the training and testing datasets

    train_score = r2_score(y_train, y_pred_train)
    test_score = r2_score(y_test, y_pred_test)
    # # The r2_score function compares the real values of after_GA_per_90 from the dataset with the predicted values produced from the model as inputs

    return model, train_score, test_score, y_pred_test
    # We return the ML model that was used so here GradientBoostingRegression(), the train_score i.e., the R² score for the training dataset
    # test_score i.e., the R² score for the testing dataset and y_pred_test, the predicted after_GA_per_90 for each player


# ============================================================

def train_all_models(X_train, y_train, X_test, y_test):
    # This function trains all 3 models i.e., Linear Regression, Random Forest and Gradient Boosting on the training data.

    results = {}
    # We create a dictionary named results because for each model, we want a single object that contains everything we need from the model
    # i.e., the trained model itself(coefficients, internal settings ...), training R² score, testing R² score and the predictions on the test set

    lr_model, lr_train, lr_test, lr_pred = train_linear_regression(
        X_train, y_train, X_test, y_test
    ) # These are the names given to the outputs of the train_linear_regression function

    results["Linear Regression"] = {
        "model": lr_model,
        "train_r2": lr_train,
        "test_r2": lr_test,
        "predictions": lr_pred
    } # We then stock them in the dictionary notice that there is the main dictionnary where the key is the name of the model and the value is the output of the executed function above
      # We then have a secondary dictionary where each key is for the respective 4 outputs of the function above

    rf_model, rf_train, rf_test, rf_pred = train_random_forest(
        X_train, y_train, X_test, y_test
    ) # These are the names given to the outputs of the train_random_forest function

    results["Random Forest"] = {
        "model": rf_model,
        "train_r2": rf_train,
        "test_r2": rf_test,
        "predictions": rf_pred
    } # We then stock them in the dictionary notice that there is the main dictionnary where the key is the name of the model and the value is the output of the executed function above
      # We then have a secondary dictionary where each key is for the respective 4 outputs of the function above

    gb_model, gb_train, gb_test, gb_pred = train_gradient_boosting(
        X_train, y_train, X_test, y_test
    ) # These are the names given to the outputs of the train_gradient_boosting function

    results["Gradient Boosting"] = {
        "model": gb_model,
        "train_r2": gb_train,
        "test_r2": gb_test,
        "predictions": gb_pred
    } # We then stock them in the dictionary notice that there is the main dictionnary where the key is the name of the model and the value is the output of the executed function above
      # We then have a secondary dictionary where each key is for the respective 4 outputs of the function above

    return results
    # We then return the entire dictionary