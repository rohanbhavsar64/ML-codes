import numpy as np
import pandas as pd
match = pd.read_csv('matches.csv')
delivery = pd.read_csv('deliveries.csv')
total_score_df = delivery.groupby(['match_id','inning']).sum()['total_runs'].reset_index()
total_score_df = total_score_df[total_score_df['inning'] == 1]
match_df = match.merge(total_score_df[['match_id','total_runs']],left_on='id',right_on='match_id')
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
match_df['team1'] = match_df['team1'].str.replace('Delhi Daredevils','Delhi Capitals')
match_df['team2'] = match_df['team2'].str.replace('Delhi Daredevils','Delhi Capitals')

match_df['team1'] = match_df['team1'].str.replace('Deccan Chargers','Sunrisers Hyderabad')
match_df['team2'] = match_df['team2'].str.replace('Deccan Chargers','Sunrisers Hyderabad')
match_df = match_df[match_df['team1'].isin(teams)]
match_df = match_df[match_df['team2'].isin(teams)]
match_df.shape
match_df = match_df[match_df['dl_applied'] == 0]
match_df = match_df[['match_id','city','winner','total_runs']]
delivery_df = match_df.merge(delivery,on='match_id')
delivery_df = delivery_df[delivery_df['inning'] == 2]
groups = delivery_df.groupby('match_id')

match_ids = delivery_df['match_id'].unique()
last_five = []
for id in match_ids:
      last_five.extend(groups.get_group(id).rolling(window=18).sum()['total_runs_y'].values.tolist())
delivery_df['last_five']=last_five
delivery_df['city'].value_counts().keys()
delivery_df.groupby('match_id').cumsum()['total_runs_y']
delivery_df['current score']=delivery_df.groupby('match_id').cumsum()['total_runs_y']
delivery_df['runs_left']=delivery_df['total_runs_x']+1-delivery_df['current score']
delivery_df['ball_left']=120-((delivery_df['over']-1)*6+delivery_df['ball'])
delivery_df['player_dismissed']=delivery_df['player_dismissed'].fillna(0)
delivery_df['player_dismissed']=delivery_df['player_dismissed'].apply(lambda x:x if x == 0 else 1)
delivery_df['player_dismissed']=delivery_df['player_dismissed']
wickets=delivery_df.groupby('match_id').cumsum()['player_dismissed'].values
delivery_df['wickets_left']=10-wickets
groups = delivery_df.groupby('match_id')

match_ids = delivery_df['match_id'].unique()
last_five = []
for id in match_ids:
    last_five.extend(groups.get_group(id).rolling(window=18).sum()['player_dismissed'].values.tolist())
delivery_df['last_five_wicket']=last_five
delivery_df['crr']=(delivery_df['current score']*6)/(120-delivery_df['ball_left'])
delivery_df['rrr']=(delivery_df['runs_left']*6)/(delivery_df['ball_left'])
delivery_df
def result(raw):
    return 1 if(raw['batting_team']==raw['winner']) else 0
delivery_df['result']=delivery_df.apply(result,axis=1)
final_df = delivery_df[['batting_team','bowling_team','city','batsman','non_striker','runs_left','ball_left','wickets_left','total_runs_x','crr','rrr','result','last_five','last_five_wicket']]
final_df = final_df.sample(final_df.shape[0])
final_df=final_df[final_df['ball_left']!=0]
final_df['batsman']=final_df['batsman'].str.split(' ').str.get(-1)
final_df['non_striker']=final_df['non_striker'].str.split(' ').str.get(-1)
final_df.dropna(inplace=True)
final_df.to_csv('result.csv')
x=final_df.drop(columns='result')
y=final_df['result']
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest= train_test_split(x,y,test_size=0.2,random_state=1)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder()
trf = ColumnTransformer([
    ('trf',OneHotEncoder(sparse_output=False,handle_unknown = 'ignore'),['batting_team','bowling_team','city','batsman','non_striker'])],remainder='passthrough')
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
pipe=Pipeline(
    steps=[
        ('step1',trf),
        ('step2',LogisticRegression())
    ])
pipe.fit(xtrain,ytrain)
pipe.predict_proba(xtest)[1]

n=pipe.predict_proba(pd.DataFrame(columns=['batting_team','bowling_team','city','batsman','non_striker','runs_left','ball_left','wickets_left','total_runs_x','crr','rrr','last_five','last_five_wicket'],data=np.array(['Royal Challengers Bangalore','Chennai Super Kings','Indore','Dhoni','Sundar',63,42,7,2,11.23,9.00,33,2.0]).reshape(1,13))).astype(float)

print("Win Chances of Batting team is:", n[0][1]*100,"%")
print("Win Chances of Bowling team is:", n[0][0]*100,"%")
final_df['city'].unique()
def match_progression(x_df,match_id,pipe):
    match = x_df[x_df['match_id'] == match_id]
    match = match[(match['ball'] == 6)]
    temp_df = match[['batting_team','bowling_team','city','batsman', 'non_striker','runs_left','ball_left','wickets_left','total_runs_x','crr','rrr','last_five_wicket', 'last_five']].dropna()
    temp_df = temp_df[temp_df['ball_left'] != 0]
    result = pipe.predict_proba(temp_df)
    temp_df['lose'] = np.round(result.T[0]*100,1)
    temp_df['win'] = np.round(result.T[1]*100,1)
    temp_df['end_of_over'] = range(1,temp_df.shape[0]+1)
    
    target = temp_df['total_runs_x'].values[0]
    runs = list(temp_df['runs_left'].values)
    new_runs = runs[:]
    runs.insert(0,target)
    temp_df['runs_after_over'] = np.array(runs)[:-1] - np.array(new_runs)
    wickets = list(temp_df['wickets_left'].values)
    new_wickets = wickets[:]
    new_wickets.insert(0,10)
    wickets.append(0)
    w = np.array(wickets)
    nw = np.array(new_wickets)
    temp_df['wickets_in_over'] = (nw - w)[0:temp_df.shape[0]]
    
    print("Target-",target)
    temp_df = temp_df[['end_of_over','runs_after_over','wickets_in_over','lose','win']]
    return temp_df,target


import streamlit as st
a=st.number_input()


temp_df,target = match_progression(delivery_df,a,pipe)
temp_df


import matplotlib.pyplot as plt
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(x=temp_df['end_of_over'], y=temp_df['wickets_in_over'], mode='markers', marker=dict(color='red')))
fig.add_trace(go.Scatter(x=temp_df['end_of_over'], y=temp_df['win'], mode='lines', line=dict(color='#00a65a', width=4)))
fig.add_trace(go.Scatter(x=temp_df['end_of_over'], y=temp_df['lose'], mode='lines', line=dict(color='red', width=4)))
fig.add_trace(go.Bar(x=temp_df['end_of_over'], y=temp_df['runs_after_over']))
fig.update_layout(title='Target-' + str(target))

st.write(fig.show())
