# Football Transfer Performance Prediction: Model Comparison

## Research Question
Which Machine Learning model best predicts a player's Goals + Assists in the season following a transfer:
Linear Regression, Random Forest Regressor or Gradient Boosting Regressor?

## Setup

# Create and activate environment
conda env create -f environment.yml
conda activate football-transfer-project

## Usage

python main.py

Expected output:  R² comparison between the three models

## Project Structure

project-repo/

├── main.py                       # Main entry point
├── modeling.py                   # Model training 
├── evaluation.py                 # Evaluation metrics
├── src/                          # Additional scripts (plots and data filtering/merging)  
│  
├── data/                         
│   └── processed/                
│       └── transfers_matched_complete.csv   # Final merged dataset with performance statistics and transfer fees
│
├── results/  # Output plots and metrics
│
├── environment.yml               # Dependencies  
├── README.md                     # Setup and usage instructions 
├── PROPOSAL.md                   # Initial project proposal  
└── project_report.pdf            # Final report  

## Results

R² scores on the test set:

- Linear Regression: R² = 0.1900 
- Random Forest: R² = 0.1725 
- Gradient Boosting: R² = 0.0807
- **Winner:** Linear Regression

## Requirements

- Python 3.10
- scikit-learn
- pandas
- numpy
- matplotlib
- joblib
- jupyter
