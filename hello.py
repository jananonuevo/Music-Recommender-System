import itertools
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import pandas as pd
import io
sys.path.append('..')

import lightfm
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm import cross_validation

data = pd.read_csv('dummy_ratings.csv')
users = pd.read_csv('dummy_users.csv')
songs= pd.read_csv('dummy_songs.csv')

from lightfm.evaluation import precision_at_k
from lightfm.evaluation import recall_at_k
from lightfm.evaluation import auc_score

def calculate_metrics(model, interactions, train, item_features, user_features, k):
    precision = precision_at_k(model, interactions, train, item_features=item_features, user_features=user_features, k=k).mean()
    recall = recall_at_k(model, interactions, train, item_features=item_features, user_features=user_features, k=k).mean()
    auc = auc_score(model, interactions, None, item_features=item_features, user_features=user_features).mean()
    return precision, recall, auc

#tried lowering learning,weights in creasing th number of features etc.DOES NOT HAVE SIGNIFICANT IMPACT ON RESULTS, NEED TO IMPROVE DATASET
# default number of recommendations
K = 10
# percentage of data used for testing
TEST_PERCENTAGE = 0.3
# model learning rate
LEARNING_RATE = 0.01
# no of latent factors
NO_COMPONENTS = 200
# no of epochs to fit model
NO_EPOCHS = 30
# no of threads to fit model
NO_THREADS = 8

# regularisation for both user and item features
ITEM_ALPHA = 0
USER_ALPHA = 0

checkpoint = 'lightFM_hybrid'
# seed for pseudonumber generations
SEED = 42

songs=songs.drop(['link','userTotal','popularity'
], axis=1)  # axis=1 drops irrelevant columns
songs.dtypes

dataset = Dataset()
users_cols = users.columns[1:].tolist()
songs_cols =  songs.columns[3:].tolist()

all_user_features = np.concatenate([users[col].unique() for col in users_cols]).tolist()
all_item_features = np.concatenate([songs[col].unique() for col in songs_cols]).tolist()

dataset.fit(
    users=users['userID'],
    items=songs['songID'],
    user_features=all_user_features,
    item_features=all_item_features
)

# number of unique users and items should be 50
num_users, num_items = dataset.interactions_shape()
print(f'Num users: {num_users}, num_items: {num_items}.')

(interactions, weights) = dataset.build_interactions(zip(data['userID'], data['songID']))

def item_feature_generator():
    for i, row in songs.iterrows():
        features = row.values[3:]
        yield (row['songID'], features)

def user_feature_generator():
    for i, row in users.iterrows():
        features = row.values[1:]
        yield (row['userID'], features)

item_features = dataset.build_item_features((item_id, item_feature) for item_id, item_feature in item_feature_generator())
user_features = dataset.build_user_features((user_id, user_feature) for user_id, user_feature in user_feature_generator())

train_interactions, test_interactions = cross_validation.random_train_test_split(
    interactions, test_percentage=TEST_PERCENTAGE,
    random_state=np.random.RandomState(SEED)
)

uids, iids, data_interaction = cross_validation._shuffle(interactions.row, interactions.col, interactions.data, np.random.RandomState(SEED))

cutoff = int((1.0 - TEST_PERCENTAGE) * len(uids))
test_idx = slice(cutoff, None)
train_idx = slice(None, cutoff)

test_uids, test_iids = uids[test_idx], iids[test_idx]
train_uids, train_iids = uids[train_idx], iids[train_idx]

import pickle

# Replace "model.pkl" with the actual filename of your pickle file
with open("lightFM_hybrid (1).pickle", "rb") as file:
  model = pickle.load(file)

from IPython.display import display_html
def display_side_by_side(*args):
    html_str=''
    for df in args:
        html_str+=df.to_html()
    display_html(html_str.replace('table','table style="display:inline"'),raw=True)

def computePersonalityScore(q1, q2, q3, q4, q5, q6, q7, q8, q9, q10):
    import csv
    import pandas as pd

    extraversion = 3
    agreeableness = 3
    openness = 3
    conscientiousness = 3
    neuroticism = 3

    userid = "U101"
    data = pd.DataFrame({
            'userID': userid,
            'E_Score': extraversion,
            'E_High': 0,
            'E_Avg': 0,
            'E_Low': 0,
            'A_Score': agreeableness,
            'A_High': 0,
            'A_Avg': 0,
            'A_Low': 0,
            'N_Score': neuroticism,
            'N_High': 0,
            'N_Avg': 0,
            'N_Low': 0,
            'C_Score': conscientiousness,
            'C_High': 0,
            'C_Avg': 0,
            'C_Low': 0,
            'O_Score': openness,
            'O_High': 0,
            'O_Avg': 0,
            'O_Low': 0,
            'genre_EDM': 0,
            'genre_country': 0,
            'genre_indie': 0,
            'genre_metal': 0,
            'genre_pop': 0,
            'genre_pop-punk': 0,
            'genre_rap': 0,
            'genre_rock': 0,
            'genre_singer-songwriter': 0,
            'genre_soul': 0
            }, index=['userID'])

    #0: E, 1: A, 2: O, 3: C, 4: N
    self_ratings = [0,0,0,0,0]

    self_ratings[0] = ((((extraversion / 2) - 3.2) / 0.8) * 10) + 50
    self_ratings[1] = ((((agreeableness / 2) - 3.8) / 0.6) * 10) + 50
    self_ratings[2] = ((((openness / 2) - 3.7) / 0.7) * 10) + 50
    self_ratings[3] = ((((conscientiousness / 2) - 3.6) / 0.7) * 10) + 50
    self_ratings[4] = ((((neuroticism / 2) - 3.0) / 0.8) * 10) + 50

    print(str(self_ratings) +"\n")

    for i in range(0, 4):
        if self_ratings[i] < 19:
            if i == 0:
                data['E_Low'] = 1
            elif i == 1:
                data['A_Low'] = 1
            elif i == 2:
                data['O_Low'] = 1
            elif i == 3:
                data['C_Low'] = 1
            elif i == 4:
                data['N_Low'] = 1
        elif self_ratings[i] >= 19 and self_ratings[i] <= 28:
            if i == 0:
                data['E_Avg'] = 1
            elif i == 1:
                data['A_Avg'] = 1
            elif i == 2:
                data['O_Avg'] = 1
            elif i == 3:
                data['C_Avg'] = 1
            elif i == 4:
                data['N_Avg'] = 1
        elif self_ratings[i] > 28:
            if i == 0:
                data['E_High'] = 1
            elif i == 1:
                data['A_High'] = 1
            elif i == 2:
                data['O_High'] = 1
            elif i == 3:
                data['C_High'] = 1
            elif i == 4:
                data['N_High'] = 1

    print(data)

    data.to_csv('dummy_users.csv', mode='a', index=False, header=False)


def generateRecommendations(model, userIDs, k):
    mapper_to_internal_ids = dataset.mapping()[2]
    mapper_to_external_ids = {v: k for k, v in mapper_to_internal_ids.items()}

    user_mapper_to_internal = dataset.mapping()[0]
    user_mapper_to_external = {v: k for k, v in user_mapper_to_internal.items()}
    userIDs = np.vectorize(user_mapper_to_internal.get)(userIDs)

    for userID in userIDs:

        print(f'Songs liked by user {user_mapper_to_external[userID]}:')

        train_item_ids = [iid for uid, iid in zip(train_uids, train_iids) if uid == userID]
        songs_liked = songs[songs['songID'].isin(np.vectorize(mapper_to_external_ids.get)(train_item_ids))]

        display_side_by_side(songs_liked[['songID', 'name','artists']])

        scores = model.predict(
            int(userID),
            list(mapper_to_internal_ids.values()),
            user_features=user_features,
            item_features=item_features
        )
        top_k_indices = np.argsort(-scores)[:k]
        print(f'songs recommended to user {user_mapper_to_external[userID]}:')

        songs_recommended = songs[songs['songID'].isin(np.vectorize(mapper_to_external_ids.get)(top_k_indices))]
        songs_recommended['liked'] = songs_recommended['songID'].isin(songs_liked['songID'])
        display_side_by_side(songs_recommended[['songID', 'name','artists','liked']])

#userid_na_gagamitin = type(np.random.choice(users['userID'], 1))
filtered_userids = users.loc[users['userID'] == 'U50']
convert_to_numpyndarray = filtered_userids.to_numpy() 

generateRecommendations(model, users['userID'][users['userID'] == 'U49'], 3)

