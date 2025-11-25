from sklearn.metrics import r2_score
# We import the r2_score function to compute the R² value


def evaluate_model(model, X_test, y_test):
    # We create an evaluate_Model that will enable us to evaluate a regression model using the R².
    # The first input is the model this can be 1 out the 3 ML Models we used i.e., LinearRegression(), RandomForestRegressor() and GradientBoostingRegressor()
    # The second input contains the test features used by the model to make predictions and the third input contains the true after_GA_per_90 values for the test set.    
    # This will return the R² score i.e., how well the model explains the variation in after_GA_per_90
   

    y_pred = model.predict(X_test)
    # predict() takes input features x (e.g., age, minutes_played, before_GA_per_90 ...) and generates the model's estimated output based on what it learned during training
    # So here this concretely means that we will have one predicted value of after_GA_per_90 per player
    # We only do this on the testing dataset here as we wish to see how well the model performs on new unseen data
    
    r2 = r2_score(y_test, y_pred)
    # We use the r2_score function to test how well the model explains the variation in after_GA_per_90
    # Here we compare the predicted values we determined before y_pred with the real values y_test

    return r2
    # We return this r2 value found


def print_evaluation_results(model_name, r2_value):
    # The purpose of this function is simply to display the previous results more neatly
    # The inputs are the model name for one of the three ML Models and its respective R² score
   
    print(f"Model: {model_name} - R² score: {r2_value:.4f}")
    # This gives us a statement where the R² score is shown with 4 digits after the decimal point

