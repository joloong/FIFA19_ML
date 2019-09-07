import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


data = pd.read_csv('../input/data.csv')

# Try to get some useful features and this featuref is not null
useful_feat = ['Name',
               'Age',
               'Photo',
               'Nationality',
               'Flag',
               'Overall',
               'Potential',
               'Club',
               'Club Logo',
               'Value',
               'Wage',
               'Preferred Foot',
               'International Reputation',
               'Weak Foot',
               'Skill Moves',
               'Work Rate',
               'Body Type',
               'Position',
               'Joined',
               'Contract Valid Until',
               'Height',
               'Weight',
               'Crossing',
               'Finishing',
               'HeadingAccuracy',
               'ShortPassing',
               'Volleys',
               'Dribbling',
               'Curve',
               'FKAccuracy',
               'LongPassing',
               'BallControl',
               'Acceleration',
               'SprintSpeed',
               'Agility',
               'Reactions',
               'Balance',
               'ShotPower',
               'Jumping',
               'Stamina',
               'Strength',
               'LongShots',
               'Aggression',
               'Interceptions',
               'Positioning',
               'Vision',
               'Penalties',
               'Composure',
               'Marking',
               'StandingTackle',
               'SlidingTackle',
               'GKDiving',
               'GKHandling',
               'GKKicking',
               'GKPositioning',
               'GKReflexes']

df = pd.DataFrame(data, columns=useful_feat)

# Include all the player except goalkeeper
vals = ['RF', 'ST', 'LW', 'RCM', 'LF', 'RS', 'RCB', 'LCM', 'CB',
        'LDM', 'CAM', 'CDM', 'LS', 'LCB', 'RM', 'LAM', 'LM', 'LB', 'RDM',
        'RW', 'CM', 'RB', 'RAM', 'CF', 'RWB', 'LWB']
ml_players = df.loc[df['Position'].isin(vals) & df['Position']]

# choose all the columns we need
ml_cols = ['Crossing',
           'Finishing',
           'HeadingAccuracy',
           'ShortPassing',
           'Volleys',
           'Dribbling',
           'Curve',
           'FKAccuracy',
           'LongPassing',
           'BallControl',
           'Acceleration',
           'SprintSpeed',
           'Agility',
           'Reactions',
           'Balance',
           'ShotPower',
           'Jumping',
           'Stamina',
           'Strength',
           'LongShots',
           'Aggression',
           'Interceptions',
           'Positioning',
           'Vision',
           'Penalties',
           'Composure',
           'Marking',
           'StandingTackle',
           'SlidingTackle',
           'Overall'
           ]


df_ml = pd.DataFrame(data=ml_players, columns=ml_cols)

# Train test split
y = df_ml['Overall']
X = df_ml[['Crossing',
           'Finishing',
           'HeadingAccuracy',
           'ShortPassing',
           'Volleys',
           'Dribbling',
           'Curve',
           'FKAccuracy',
           'LongPassing',
           'BallControl',
           'Acceleration',
           'SprintSpeed',
           'Agility',
           'Reactions',
           'Balance',
           'ShotPower',
           'Jumping',
           'Stamina',
           'Strength',
           'LongShots',
           'Aggression',
           'Interceptions',
           'Positioning',
           'Vision',
           'Penalties',
           'Composure',
           'Marking',
           'StandingTackle',
           'SlidingTackle']]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=101)

lm = LinearRegression()
lm.fit(X_train, y_train)
predictions = lm.predict(X_test)
plt.scatter(y_test, predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted y')

# Get the effect of every parameter
coeffecients = pd.DataFrame(lm.coef_, X.columns)
coeffecients.columns = ['Coeffecient']
print(coeffecients)
