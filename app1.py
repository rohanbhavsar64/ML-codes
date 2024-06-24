import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Load data
match = pd.read_csv('matches.csv')
delivery = pd.read_csv('deliveries.csv')

# Data preprocessing
delivery['total_runs'] = delivery['total_runs'].astype(int)
delivery['match_id'] = delivery['match_id'].astype(int)

total_score_df = delivery.groupby(['match_id', 'inning']).sum()['total_runs'].reset_index()
total_score_df = total_score_df[total_score_df['inning'] == 1]

match_df = match.merge(total_score_df[['match_id', 'total_runs']], left_on='id', right_on='match_id')

teams = [
    'Sunrisers Hyderabad',
    'Mumbai Indians',
    'Royal Challengers Bangalore',
    'Kolkata Knight Riders',
    'Kings XI Punjab',
    'Chennai Super Kings',
    'Rajasthan Royals',
    'Delhi Capitals'
]

match_df['team1'] = match_df['team1'].str.replace('Delhi Daredevils', 'Delhi Capitals')
match_df['team2'] = match_df['team2'].str.replace('Delhi Daredevils', 'Delhi Capitals')

match_df['team1'] = match_df['team1'].str.replace('Deccan Chargers', 'Sunrisers Hyderabad')
match_df['team2'] = match_df['team2'].str.replace('Deccan Chargers', 'Sunrisers Hyderabad')

match_df = match_df[match_df['team1'].isin(teams)]
match_df = match_df[match_df['team2'].isin(teams)]

match_df.shape
match_df = match_df[match_df['dl_applied'] == 0]
match_df = match_df[['match_id', 'city', 'winner', 'total_runs']]

delivery_df = match_df.merge(delivery, on='match_id')
delivery_df = delivery_df[delivery_df['inning'] == 2]

groups = delivery_df.groupby('match_id')

match_ids = delivery_df['match_id'].unique()
last_five = []
for id in match_ids:
    last_five.extend(groups.get_group(id).rolling(window=18).sum()['total_runs'].values.tolist())
delivery_df['last_five'] = last_five

delivery_df['city'].value_counts().keys()
delivery_df.groupby('match_id').cumsum()['total_runs']
delivery_df['current score'] = delivery_df.groupby('match_id').cumsum()['total_runs']
delivery_df['runs_left'] = delivery_df['total_runs'] + 1 - delivery_df['current score']
delivery_df['ball_left'] = 120 - ((delivery_df['over'] - 1) * 6 + delivery_df['ball'])
delivery_df['player_dismissed'] = delivery_df['player_dismissed'].fillna(0)
delivery_df['player_dismissed'] = delivery_df['player_dismissed'].apply(lambda x: x if x == 0 else 1)
delivery_df['player_dismissed'] = delivery_df['player_dismissed']

wickets = delivery_df.groupby('match_id').cumsum()['player_dismissed'].values
delivery_df['wickets_left'] = 10 - wickets

groups = delivery_df.groupby('match_id')

match_ids = delivery_df['match_id'].unique()
last_five = []
for id in match_ids:
    last_five.extend(groups.get_group(id).rolling(window=18).sum()['player_dismissed'].values.tolist())
delivery_df['last_five_wicket'] = last_five

delivery_df['crr'] = (delivery_df['current score'] * 6) / (120 - delivery_df['ball_left'])
delivery_df['rrr'] = (delivery_df['runs_left'] * 6) / (delivery_df['ball_left'])

def result(raw):
    return 1 if (raw['batting_team'] == raw['winner']) else 0

delivery_df['result'] = delivery_df.apply(result, axis=1)

final_df = delivery_df[['batting_team', 'bowling_team', 'city', 'batsman', 'non_striker', 'runs_left', 'ball_left', 'wickets_left', 'total_runs', 'crr', 'rrr', 'esult', 'last_five', 'last_five_wicket']]
final_df = final_df.sample(final_df.shape[0])
final_df = final_df[final_df['ball_left']!= 0]

final_df['batsman'] = final_df['batsman'].str.split(' ').str.get(-1)
final_df['non_striker'] = final_df['non_striker'].str.split(' ').str.get(-1)

final_df.dropna(inplace=True)
final_df.to_csv('result.csv')

x = final_df.drop(columns='result')
y = final_df['result']

from sklearn.model_selection import train_test
st.write(fig.show())
