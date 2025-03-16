import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
# Example: Load data from CSV
df = pd.read_csv('IPL.csv')

# Assuming relevant columns are: 'match_id', 'over', 'runs_left', 'wickets_left', 'total_runs_x', 'ball_left'
# Filter relevant columns
df = df[['match_id','batting_team','bowling_team','city','over', 'runs_left', 'wickets_left', 'total_runs_x', 'ball_left','last_five','result']]

# Drop rows with missing values if any
df.dropna(inplace=True)
# Define preprocessing steps using ColumnTransformer
preprocessor = ColumnTransformer([
    ('ohe', OneHotEncoder(handle_unknown='ignore'), ['batting_team','bowling_team','city']),  # Example with OneHotEncoder for match_id
    ('scaler', StandardScaler(), ['match_id','over', 'runs_left', 'wickets_left', 'total_runs_x', 'ball_left','last_three_overs_runs'])
])

# Define Logistic Regression model
log_reg = LogisticRegression()

# Create pipeline
pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', log_reg)
])
# Separate features and target variable
# Separate features and target variable
X = df.drop(columns=['result'])  # Drop 'result' column as it's the target variable
y = df['result']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
# Predict win probabilities for test dataain)
pipe.fit(X_train,y_train)
# Predict probabilities for test data
win_probabilities = pipe.predict_proba(X_test)

# Assuming class 1 is 'win', extract probabilities for win for both teams
win_probabilities_batting = win_probabilities[:, 1]
win_probabilities_bowling = 1 - win_probabilities_batting  # Assuming binary outcome (win or lose)

# Add win probabilities to X_test for plotting (assuming 'over' is still part of X_test)
X_test['win_probability_batting'] = win_probabilities_batting
X_test['win_probability_bowling'] = win_probabilities_bowling

# Plot win probability per over for both teams
plt.figure(figsize=(10, 6))
plt.plot(X_test['over'], X_test['win_probability_batting'], marker='o', linestyle='-', color='b', label='Batting Team')
plt.plot(X_test['over'], X_test['win_probability_bowling'], marker='o', linestyle='-', color='r', label='Bowling Team')
plt.xlabel('Over')
plt.ylabel('Win Probability')
plt.title('Predicted Win Probability Per Over of the Match')
plt.ylim([0, 1])
plt.legend()
plt.grid(True)
plt.tight_layout()
st.write(plt.show())
