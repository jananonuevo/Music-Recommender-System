from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
import pandas as pd

import lightfm
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm import cross_validation

data = pd.read_csv('ratings.csv')
users = pd.read_csv('users.csv')
songs= pd.read_csv('songs.csv')

from lightfm.evaluation import precision_at_k
from lightfm.evaluation import recall_at_k
from lightfm.evaluation import auc_score

app = Flask(__name__)  # Create the Flask app instance

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

with open("lightFM_hybrid.pickle", "rb") as file:
    model = pickle.load(file)

def reverseScore(r):
    if(r == 5):
        r = 1
    elif(r == 4):
        r = 2
    elif(r == 3):
        r = 3
    elif(r == 2):
        r = 4
    elif(r == 1):
        r = 5
    return r

def computePersonalityScore(q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, top_genres_user):
    #compute for the user bf dimension mean here
    extraversion = reverseScore(q1) + q6
    agreeableness = q2 + reverseScore(q7)
    openness = reverseScore(q5) + q10
    conscientiousness = reverseScore(q3) + q8
    neuroticism = reverseScore(q4) + q9

    #userid = "U" +str(getLastUserID + 1)
    df_personality_profile =  {
            'E_High': 0,
            'E_Avg': 0,
            'E_Low': 0,
            'A_High': 0,
            'A_Avg': 0,
            'A_Low': 0,
            'N_High': 0,
            'N_Avg': 0,
            'N_Low': 0,
            'C_High': 0,
            'C_Avg': 0,
            'C_Low': 0,
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
            }

    #--0: Extraversion, 1: Agreeableness, 2: Openness, 3: Conscientiousness, 4: Neuroticism
    self_ratings = [0,0,0,0,0]

    self_ratings[0] = ((((extraversion / 2) - 2.142857143) / 0.5267566112) * 10) + 50
    self_ratings[1] = ((((agreeableness / 2) - 3.376190476) / 0.7807200584) * 10) + 50
    self_ratings[2] = ((((openness / 2) - 3.476190476) / 0.7977762133) * 10) + 50
    self_ratings[3] = ((((conscientiousness / 2) - 3.138095238) / 0.7019726261) * 10) + 50
    self_ratings[4] = ((((neuroticism / 2) - 3.152380952) / 0.9331599787) * 10) + 50

    print(str(self_ratings) +"\n")

    for i in range(0, 4):
        if self_ratings[i] < 19:
            if i == 0:
                df_personality_profile['E_Low'] = 1
            elif i == 1:
                df_personality_profile['A_Low'] = 1
            elif i == 2:
                df_personality_profile['O_Low'] = 1
            elif i == 3:
                df_personality_profile['C_Low'] = 1
            elif i == 4:
                df_personality_profile['N_Low'] = 1
        elif self_ratings[i] >= 19 and self_ratings[i] <= 28:
            if i == 0:
                df_personality_profile['E_Avg'] = 1
            elif i == 1:
                df_personality_profile['A_Avg'] = 1
            elif i == 2:
                df_personality_profile['O_Avg'] = 1
            elif i == 3:
                df_personality_profile['C_Avg'] = 1
            elif i == 4:
                df_personality_profile['N_Avg'] = 1
        elif self_ratings[i] > 28:
            if i == 0:
                df_personality_profile['E_High'] = 1
            elif i == 1:
                df_personality_profile['A_High'] = 1
            elif i == 2:
                df_personality_profile['O_High'] = 1
            elif i == 3:
                df_personality_profile['C_High'] = 1
            elif i == 4:
                df_personality_profile['N_High'] = 1

    for i in range(len(top_genres_user)):
        if top_genres_user[i] == 'edm':
            df_personality_profile['genre_EDM'] = 1
        elif top_genres_user[i] == 'country':
            df_personality_profile['genre_country'] = 1
        elif top_genres_user[i] == 'indie':
            df_personality_profile['genre_indie'] = 1
        elif top_genres_user[i] == 'metal':
            df_personality_profile['genre_metal'] = 1
        elif top_genres_user[i] == 'pop':
            df_personality_profile['genre_pop'] = 1
        elif top_genres_user[i] == 'poppunk':
            df_personality_profile['genre_pop-punk'] = 1
        elif top_genres_user[i] == 'rap':
            df_personality_profile['genre_rap'] = 1
        elif top_genres_user[i] == 'rock':
            df_personality_profile['genre_rock'] = 1
        elif top_genres_user[i] == 'singersongwriter':
            df_personality_profile['genre_singer-songwriter'] = 1
        elif top_genres_user[i] == 'soul':
            df_personality_profile['genre_soul'] = 1

    return df_personality_profile

def newUserRecommendation(model, dataset, userID=None, new_user_feature=None, k=10):
      userID = 107
      mapper_to_internal_ids = dataset.mapping()[2]
      mapper_to_external_ids = {v: k for k, v in mapper_to_internal_ids.items()}
      user_mapper_to_internal = dataset.mapping()[0]
      user_mapper_to_external = {v: k for k, v in user_mapper_to_internal.items()}

      dataset.fit(users=[userID], items=songs['songID'], user_features=new_user_feature, item_features=all_item_features)

      new_user_feature = [userID,new_user_feature]
      new_user_feature = dataset.build_user_features([new_user_feature], normalize=False)
      
      userID_map = dataset.mapping()[0][userID]
      scores = model.predict(
        userID_map,
        list(dataset.mapping()[1].values()),
        user_features=new_user_feature,
        item_features=item_features
      )
      top_k_indices = np.argsort(-scores)[:k]
      print(f'songs recommended to user {userID}:')

      recommended_song_ids = np.vectorize(mapper_to_external_ids.get)(top_k_indices)
      songs_recommended = songs[songs['songID'].isin(recommended_song_ids)]
      return songs_recommended[['songID', 'name', 'artists']]
    

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/", methods=['POST'])
def get_recos():
    with app.app_context():
        q1 = int(request.form['q1'])
        q2 = int(request.form['q2'])
        q3 = int(request.form['q3'])
        q4 = int(request.form['q4'])
        q5 = int(request.form['q5'])
        q6 = int(request.form['q6'])
        q7 = int(request.form['q7'])
        q8 = int(request.form['q8'])
        q9 = int(request.form['q9'])
        q10 = int(request.form['q10'])
        top_genres_user = request.form.getlist('genre')

        #recommendations = generateRecommendations(model, users['userID'][users['userID'] == userrid], 10)
        recommendations = newUserRecommendation(model, dataset, new_user_feature=computePersonalityScore(q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,top_genres_user))
        return render_template("index.html", htmlstr=recommendations.to_html())

if __name__ == "__main__":
  app.run(debug=True)

