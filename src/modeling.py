import pandas as pd  #  We import the pandas library for data manipulation
from sklearn.linear_model import LinearRegression   # We import our first model from sklearn's linear_model module
from sklearn.ensemble import RandomForestRegressor  # We import our second model from sklearn's ensemble module
from sklearn.ensemble import GradientBoostingRegressor  # We import our second model from sklearn's ensemble module
from sklearn.metrics import r2_score  # We import the r2 in order to be able to the evaluate ML performance from the metrics module

# ============================================================

def train_linear_regression(X_train, y_train, X_test, y_test):
    # We define a train function for the linear regression 

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
    # E.g., if R² = 1.0, then, the model explains 100% of the variation in the target.
    # The r2_score function takes the real values of after_GA_per_90 from the dataset and the predicted values produced from the model as inputs

    return model, train_score, test_score, y_pred_test


# ============================================================


def train_random_forest(X_train, y_train, X_test, y_test):
    # We define a train function for the random forest ML Model

    model = RandomForestRegressor(
        n_estimators=300,       # We select 300 trees here because it gives excellent performance with low risk of overfitting
        max_depth=None,         # This means there's no maximum depth for a given tree. It can split forever until every leaf has players with identical after_GA_per_90
                                # We can't be more precise in the prediction based on the information we have
        random_state=50,        # 50 is an arbitrary random seed and it ensures the exact same random choices are made every time
                                # So this will always follow the same sequence of random choices
        n_jobs=-1               # This just ensures the model runs as fast as possible
    )
    # model will be the name of the object of the RandomForestRegressor class
    # The principle of the Random Forest model is to generate many trees where each tree predicts a certain feature. Here the player's after_GA_per_90.
    # The model will then average all the trees predictions in order to generate one final prediction


    model.fit(X_train, y_train)
    # The model learns from the training data.

   