# Project Proposal

## Project Title:
Predicting Football Player Performance Based on Transfer Fees

## Category:
Sports Analytics & Machine Learning

## Problem statement or motivation:
In recent decades, clubs have been spending exorbitant sums of money in the transfer market, but high fees haven't always guaranteed high performance. Some expensive signings drastically underperform while cheaper signings excel. The question therefore is: Can we predict a player's future performance based on their transfer fee and historical data?

## Planned approach and technologies:
I will be collecting transfer fee data from Transfermarkt and performance statistics from FBref. I will focus on approximately 700-1,000 attacker and midfielder transfers from 2017-2024 with transfer fees of €5 million or above in the top 5 European leagues (Premier League, La Liga, Serie A, Bundesliga, Ligue 1). I will only be focusing on attackers and midfielders, using goals and assists per 90 minutes as my performance metric, given the different priorities different positions have e.g., a defender is focused on defending while an attacker is focused on scoring goals and passing. Predictor variables will include inflation-adjusted transfer fee, age, position, and previous season performance… The data will be split using an 80/20 train/test validation approach, training on 2017-2022 transfers and testing on 2023-2024 transfers to ensure the model is evaluating unseen, new data. I will be using Linear Regression to test whether expensive players perform proportionally to their transfer fee. Random Forest will allow me to capture non-linear patterns, e.g., diminishing returns where a €20M signing may only perform 1.3x better than a €10M player instead of 2x. Finally, I will be using Gradient Boosting to provide predictions by building trees that focus on correcting previous mistakes, e.g., accounting for subtle patterns such as how league quality interacts with transfer fee. I will be using statistical metrics such as R² to evaluate the efficacy of these ML models. Libraries such as pandas will be used for data management and matplotlib for graphs and diagrams.

 ## Expected challenges and how you’ll address them:
A first challenge may be the lack of present data. Some transfers may have missing or incomplete performance data due to injuries or limited playing time. I can address this by filtering for players with a minimum of, e.g., 500 minutes played in their first season and reporting those excluded. Another challenge is transfer fee inflation, given that transfer fees have dramatically increased from 2017 to 2024, making unaccounted-for comparisons is misleading. I will have to adjust all fees for inflation using appropriate indices and create relative spending features that compare each transfer fee to the league average for that specific season. Finally, the limited sample size for extreme values may heavily skew my findings, i.e., few transfers exceed €80 million, potentially leading to poor predictions for the highest-fee players. I will address this by ensuring the model doesn't overfit to the few expensive transfers and separately evaluating predictions for high-fee players to assess reliability.


 ## Success criteria (how will you know it’s working?):
 These are the success criteria: I successfully collect 700-1,000 attacker and midfielder transfers for the dataset. All three ML models train successfully without errors and generate predictions on the test set that don't overfit the training sample. The gap between training and test set performance is small (within 0.10 R² difference), confirming the model generalizes to new data rather than overfitting. The models achieve a test set R² above 0.30 for goals+assists per 90 minutes, demonstrating meaningful predictive power. I identify which features actually predict performance, i.e., transfer fee or previous season performance. I understand the relationship between the features and performance; whether it is linear or has diminishing returns. The code is coherent, legible, and correct, conforming to PEP 8 standards. Finally, I am able to highlight players who overperformed and underperformed relative to their fee-based predictions, demonstrating that the model has practical real-world value.


  ## Stretch goals:
  If time permits, I will analyze whether transfer performance improves in the second season, testing if players need an adaptation period before delivering full value.
