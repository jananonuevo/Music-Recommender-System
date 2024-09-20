from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
import pandas as pd

import lightfm
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm import cross_validation

import os, uuid
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

app = Flask(__name__)  # Create the Flask app instance

with open("lightFM_hybrid.pickle", "rb") as file:
    model1 = pickle.load(file)

with open("modelwithoutipp.pickle", "rb") as file:
    model2 = pickle.load(file)

#PREPROCESS DATA
xls = pd.ExcelFile('surveyData.xlsx')
songsDF = pd.read_excel(xls, 'songs')
usersDF = pd.read_excel(xls, 'users')
interactionsDF = pd.read_excel(xls, 'interactions')

usersDF= pd.concat([usersDF, usersDF['Genres'].str.get_dummies(sep=', ')], axis=1)
usersDF= usersDF.rename(columns={'Indie / POV : Indie': 'Indie','Pop-Punk':'pop punk','Pop Punk':'pop punk'})
usersDF.columns = usersDF.columns.str.lower()

def getBigFiveScores(df, columns):
  for col in columns:
    dummies = pd.get_dummies(df[col], prefix=col, drop_first=False,dtype ='int32')
    if dummies.shape[1] == 1:
        raise ValueError(f"No dummy columns created for column: {col}")
    df = pd.concat([df, dummies], axis=1)
    df = df.drop(col, axis=1)

  return df

big_five_facets = ['extraversion', 'openness', 'agreeableness', 'conscientiousness', 'neuroticism']
usersDF = getBigFiveScores(usersDF.copy(), big_five_facets)
usersDF = usersDF.drop('genres',axis=1)
usersDF.rename(columns={'userid': 'userID'}, inplace=True)

interactionsDF = interactionsDF.melt(id_vars='userID', var_name='songID', value_name='isLiked')
interactionsDF = interactionsDF[['userID','songID','isLiked']]
duplicates = interactionsDF.duplicated(subset=['userID', 'songID'], keep='first')

df_duplicates = interactionsDF[duplicates]
interactionsDF = interactionsDF[~duplicates]

dataset = Dataset()

users_cols = usersDF.columns[1:].tolist()
songs_cols =  songsDF.columns[5:24].tolist()
all_user_features = np.concatenate([usersDF[col].unique() for col in users_cols]).tolist()
all_item_features = np.concatenate([songsDF[col].unique() for col in songs_cols]).tolist()

dataset.fit(
    users=usersDF['userID'],
    items=songsDF['id'],
    user_features=all_user_features,
    item_features=all_item_features
)

num_users, num_items = dataset.interactions_shape()

(interactions, weights) = dataset.build_interactions(zip(interactionsDF['userID'], interactionsDF['songID']))

def item_feature_generator():
    for i, row in songsDF.iterrows():
        #features =  (pd.Series(row.values[5:-1]) ) .fillna(0)
        features =  (pd.Series(row.values[5:24]) ) .fillna(0)
        yield (row['id'], features)

def user_feature_generator():
    for i, row in usersDF.iterrows():
        #features =  (pd.Series(row.values[1:]) ) .fillna(0)
        features =  (pd.Series(row.values[1:]) ) .fillna(0)
        yield (row['userID'], features)

item_features = dataset.build_item_features((item_id, item_feature) for item_id, item_feature in item_feature_generator())
user_features = dataset.build_user_features((user_id, user_feature) for user_id, user_feature in user_feature_generator())
#END PREPROCESS DATA

df_new_users =  {
            'userID': '',
            'name': '',
            'studentidno': '',
            'email': '',
            'educationallvl': '',
            'coursestrand': '',
            'q1':0,
            'q2':0,
            'q3':0,
            'q4':0,
            'q5':0,
            'q6':0,
            'q7':0,
            'q8':0,
            'q9':0,
            'q10':0,
            'genre_EDM': 0,
            'genre_country': 0,
            'genre_indie': 0,
            'genre_metal': 0,
            'genre_pop': 0,
            'genre_pop-punk': 0,
            'genre_rap': 0,
            'genre_rock': 0,
            'genre_singer-songwriter': 0,
            'genre_soul': 0,
            'extraversion': 0.0,
            'agreeableness': 0.0,
            'openness': 0.0,
            'conscientiousness': 0.0,
            'neuroticism': 0.0,
            'selfrating_E': 0.0,
            'selfrating_A': 0.0,
            'selfrating_O': 0.0,
            'selfrating_C': 0.0,
            'selfrating_N': 0.0,
            'top1': '',
            'top2': '',
            'top3': '',
            'top4': '',
            'top5': '',
            'top6': '',
            'top7': '',
            'top8': '',
            'top9': '',
            'top10': ''
            }   

connect_str = "DefaultEndpointsProtocol=https;AccountName=musicrecommender;AccountKey=xuY+OSYHySZDuIiMnbR5n6u0RvY7cjfqrkUupdG+KVJ5bFmR8N+PSyQPzxktS4SFFfW49mhGu/k0+AStYg8XXw==;EndpointSuffix=core.windows.net"
container_name = "csvs"

def getCSV(connect_str, container_name, blob_name):
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    container_client = blob_service_client.get_container_client(container_name)

    blob_client = container_client.get_blob_client(blob_name)

    download_stream = blob_client.download_blob()
    df = pd.read_csv(download_stream)

    return df

def append_row_to_csv_blob(connect_str, container_name, blob_name, new_row):
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    container_client = blob_service_client.get_container_client(container_name)

    blob_client = container_client.get_blob_client(blob_name)

    download_stream = blob_client.download_blob()
    df = pd.read_csv(download_stream)
    new_row = pd.DataFrame([new_row])
    new_df = pd.concat([df, new_row], ignore_index=True)

    csv_data = new_df.to_csv(index=False)

    blob_client.upload_blob(csv_data, overwrite=True)


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
    global df_new_users
    #compute for the user bf dimension mean here
    extraversion = reverseScore(q1) + q6
    agreeableness = q2 + reverseScore(q7)
    openness = reverseScore(q5) + q10
    conscientiousness = reverseScore(q3) + q8
    neuroticism = reverseScore(q4) + q9

    df_new_users['extraversion'] = extraversion
    df_new_users['agreeableness'] = agreeableness
    df_new_users['openness'] = openness
    df_new_users['conscientiousness'] = conscientiousness
    df_new_users['neuroticism'] = neuroticism

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

    df_new_users['selfrating_E'] = self_ratings[0]
    df_new_users['selfrating_A'] = self_ratings[1]
    df_new_users['selfrating_O'] = self_ratings[2]
    df_new_users['selfrating_C'] = self_ratings[3]
    df_new_users['selfrating_N'] = self_ratings[4]

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
            df_new_users['genre_EDM'] = 1
        elif top_genres_user[i] == 'country':
            df_personality_profile['genre_country'] = 1
            df_new_users['genre_country'] = 1
        elif top_genres_user[i] == 'indie':
            df_personality_profile['genre_indie'] = 1
            df_new_users['genre_indie'] = 1
        elif top_genres_user[i] == 'metal':
            df_personality_profile['genre_metal'] = 1
            df_new_users['genre_metal'] = 1
        elif top_genres_user[i] == 'pop':
            df_personality_profile['genre_pop'] = 1
            df_new_users['genre_pop'] = 1
        elif top_genres_user[i] == 'poppunk':
            df_personality_profile['genre_pop-punk'] = 1
            df_new_users['genre_pop-punk'] = 1
        elif top_genres_user[i] == 'rap':
            df_personality_profile['genre_rap'] = 1
            df_new_users['genre_rap'] = 1
        elif top_genres_user[i] == 'rock':
            df_personality_profile['genre_rock'] = 1
            df_new_users['genre_rock'] = 1
        elif top_genres_user[i] == 'singersongwriter':
            df_personality_profile['genre_singer-songwriter'] = 1
            df_new_users['genre_singer-songwriter'] = 1
        elif top_genres_user[i] == 'soul':
            df_personality_profile['genre_soul'] = 1
            df_new_users['genre_soul'] = 1

    return df_personality_profile

def getLargestNumber():
    new_users = getCSV(connect_str, container_name, "new_users.csv")
    userID = new_users['userID'].tolist()
    userID_remove_duplicates = list(set(userID))
    userID_remove_U = [element.replace('U', '') for element in userID_remove_duplicates]
    userID_convert_int = [int(item) for item in userID_remove_U if item.isdigit()]
    userID_largest_number = max(userID_convert_int)
    return userID_largest_number

def newUserRecommendation(model, userID=None, new_user_feature=None, k=10):
    global df_new_users
    
    userID = getLargestNumber() + 1
    df_new_users['userID'] = 'U' +str(userID)

    mapper_to_internal_ids = dataset.mapping()[2]
    mapper_to_external_ids = {v: k for k, v in mapper_to_internal_ids.items()}

    dataset.fit(users=[userID], items=songsDF['id'], user_features=new_user_feature, item_features=all_item_features)

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

    recommended_song_ids = np.vectorize(mapper_to_external_ids.get)(top_k_indices)
    songs_recommended = songsDF[songsDF['id'].isin(recommended_song_ids)]
      
    for row, i in zip(songs_recommended.itertuples(), range(1, 11)):
        song_id = row.id
        df_col = 'top' +str(i)
        df_new_users[df_col] = song_id

    return songs_recommended[['id', 'name', 'artists']]

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/", methods=['POST'])
def get_recos():
    global df_new_users
    if 'getrecos' in request.form:
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
            
            recommendations = newUserRecommendation(model1, new_user_feature=computePersonalityScore(q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,top_genres_user))
            recommendations2 = newUserRecommendation(model2, new_user_feature=computePersonalityScore(q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,top_genres_user))

            htmltable_string = """
            <h4 class='text-center'> IPP Based Model Results </h4>
            <table class='table table-striped table-dark'> 
                <thead>
                    <tr> 
                        <th scope='col'>Rank</th>
                        <th scope='col'>Name</th> 
                        <th scope='col'>Artist</th> 
                        <th scope='col'>Spotify Preview</th> 
                    </tr>
                </thead>
                <tbody>
            """
            for row, i in zip(recommendations.itertuples(), range(1, 11)):
                htmltable_string += """
                <tr> 
                    <th scope='row'>""" +str(i) +"""</th>
                    <td>""" +row.name +"""</td> 
                    <td>""" +row.artists +"""</td> 
                    <td> <a href='https://open.spotify.com/track/""" +row.id +"""'> Link to Spotify </a> </td> 
                </tr>"""
            htmltable_string += "</tbody></table><br><br>"

            htmltable_string += """
            <h4 class='text-center'> Without IPP Model Results </h4>
            <table class='table table-striped table-dark'> 
                <thead>
                    <tr> 
                        <th scope='col'>Rank</th>
                        <th scope='col'>Name</th> 
                        <th scope='col'>Artist</th> 
                        <th scope='col'>Spotify Preview</th> 
                    </tr>
                </thead>
                <tbody>
            """
            for row, i in zip(recommendations2.itertuples(), range(1, 11)):
                htmltable_string += """
                <tr> 
                    <th scope='row'>""" +str(i) +"""</th>
                    <td>""" +row.name +"""</td> 
                    <td>""" +row.artists +"""</td> 
                    <td> <a href='https://open.spotify.com/track/""" +row.id +"""'> Link to Spotify </a> </td> 
                </tr>"""
            htmltable_string += "</tbody></table>"

            df_new_users['q1'] = int(q1)
            df_new_users['q2'] = int(q2)
            df_new_users['q3'] = int(q3)
            df_new_users['q4'] = int(q4)
            df_new_users['q5'] = int(q5)
            df_new_users['q6'] = int(q6)
            df_new_users['q7'] = int(q7)
            df_new_users['q8'] = int(q8)
            df_new_users['q9'] = int(q9)
            df_new_users['q10'] = int(q10)
            df_new_users['name'] = request.form['name']
            df_new_users['studentidno'] = request.form['idno']
            df_new_users['email'] = request.form['email']
            df_new_users['educationallvl'] = request.form['edu']
            df_new_users['coursestrand'] = request.form['coursestrand']

            append_row_to_csv_blob(connect_str, container_name, "new_users.csv", df_new_users)

            return render_template("results.html", htmlstrr=htmltable_string)
    elif 'getsurvey' in request.form:
        df_newusers_surveyresults =  {
            'userID': df_new_users['userID'],
            'rqq1': int(request.form['rqq1']),
            'rqq2': int(request.form['rqq2']),
            'rqq3': int(request.form['rqq3']),
            'rdq1': int(request.form['rdq1']),
            'rdq2': int(request.form['rdq2']),
            'rdq3': int(request.form['rdq3']),
            'usq1': int(request.form['usq1']),
            'usq2': int(request.form['usq2']),
            'usq3': int(request.form['usq3'])
            }  
        append_row_to_csv_blob(connect_str, container_name, "newusers_surveyresults.csv", df_newusers_surveyresults)
        return render_template("index.html")

if __name__ == "__main__":
  app.run(debug=True)

