# Loan Prediction with Machine Learning A data-driven approach
# Overview
This repository contains the code for a machine-learning project focused on loan prediction. The goal of the project is to predict whether a loan applicant is likely to be approved or not based on various features such as income, credit score, loan amount, etc. The project utilizes the Python programming language and several popular libraries for data manipulation, visualization, and machine learning. <br>
# Dataset
The dataset used for this project is stored in the train.csv file and contains information about loan applicants. It includes features like age, gender, income, credit score, loan amount, and loan approval status. The dataset has been preprocessed to handle missing values, categorical variables, and feature scaling.<br>
# Requirements
Python 3.x <br>
NumPy <br>
Pandas <br>
Matplotlib <br>
Scikit-learn <br>
You can install the required dependencies using the following command: <br>
pip install -r requirements.txt <br>

# Code Structure
The repository is organized as follows: <br>

loan.py: Python file containing the main project code.<br>
train.csv: CSV file containing the training dataset. <br>
requirements.txt: List of required Python libraries. <br>

# Implementation
Data Preprocessing: Handling missing values, encoding categorical variables using one-hot encoding, and feature scaling using standardization. <br>

Model Training: Implementing logistic regression for loan prediction. The model is trained on the preprocessed data and evaluated using 10-fold cross-validation. <br>

Model Tuning: Utilizing Grid Search to find the best hyperparameters for logistic regression, optimizing model performance. <br>

# Results
The trained logistic regression model achieved an accuracy of 93% on the test set with 10-fold cross-validation. After hyperparameter tuning using Grid Search, the model's performance was further improved to 98% accuracy. <br>

# How to Use
To run the project, follow these steps:<br>

Clone this repository to your local machine. <br>
Install the required dependencies using pip install -r requirements.txt. <br>
Open the loan.py using IDE and execute the cells.
