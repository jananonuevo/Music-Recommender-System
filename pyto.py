# from flask import Flask, render_template

#app = Flask(__name__)  # Create the Flask app instance

# @app.route("/")
# def hello():
  # message = "Hello from Flask with Templates!"
  # return render_template('index.html', message=message)

# if __name__ == "__main__":
  # app.run(debug=True)

from flask import Flask, render_template, request
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

import pickle

# Replace "model.pkl" with the actual filename of your pickle file
with open("lightFM_hybrid (1).pickle", "rb") as file:
    model = pickle.load(file)

#from IPython.display import display_html
#def display_side_by_side(*args):
 #   html_str = pd.DataFrame({args})
  #  return render_template("index.html", htmlstr=html_str)

def generateRecommendations(model, userIDs, k):
    mapper_to_internal_ids = dataset.mapping()[2]
    mapper_to_external_ids = {v: k for k, v in mapper_to_internal_ids.items()}

    user_mapper_to_internal = dataset.mapping()[0]
    user_mapper_to_external = {v: k for k, v in user_mapper_to_internal.items()}
    userIDs = np.vectorize(user_mapper_to_internal.get)(userIDs)

    for userID in userIDs:

        #print(f'Songs liked by user {user_mapper_to_external[userID]}:')

        train_item_ids = [iid for uid, iid in zip(train_uids, train_iids) if uid == userID]
        songs_liked = songs[songs['songID'].isin(np.vectorize(mapper_to_external_ids.get)(train_item_ids))]
        
        #display_side_by_side(songs_liked[['songID', 'name','artists']])

        scores = model.predict(
            int(userID),
            list(mapper_to_internal_ids.values()),
            user_features=user_features,
            item_features=item_features
        )
        top_k_indices = np.argsort(-scores)[:k]
        #print(f'songs recommended to user {user_mapper_to_external[userID]}:')

        songs_recommended = songs[songs['songID'].isin(np.vectorize(mapper_to_external_ids.get)(top_k_indices))]
        songs_recommended['liked'] = songs_recommended['songID'].isin(songs_liked['songID'])
        
        #with app.app_context(): 
        return songs_recommended[['songID', 'name','artists','liked']]

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/", methods=['POST'])
def get_recos():
    with app.app_context():
        userrid = request.form['text']
        recommendations = generateRecommendations(model, users['userID'][users['userID'] == userrid], 10)
        return render_template("index.html", htmlstr=recommendations.to_html())

if __name__ == "__main__":
  app.run(debug=True)

