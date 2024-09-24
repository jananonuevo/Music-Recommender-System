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

#Load excel DATA
xls = pd.ExcelFile('surveyData.xlsx')
songsDF = pd.read_excel(xls, 'songs')
usersDF = pd.read_excel(xls, 'users')
interactionsDF = pd.read_excel(xls, 'interactions')

#Load Dataset
#Dataset Setup for both models ( Does not include new user)
item_features_columns = [
       'danceability',
       'energy', 'acousticness', 'instrumentalness', 'liveness', 'valence',
       'edm', 'country', 'indie', 'metal', 'pop', 'pop punk', 'rap', 'rock',
       'singer songwriter', 'soul','extraversion_high',
'extraversion_average', 'extraversion_low', 'agreeableness_high',
'agreeableness_average', 'agreeableness_low', 'conscientiousness_high',
'conscientiousness_average', 'conscientiousness_low',
'neuroticism_high', 'neuroticism_average', 'neuroticism_low',
'openness_high', 'openness_average', 'openness_low'
]

user_features_columns = [ 'country', 'edm', 'indie', 'metal', 'pop', 'pop punk', 'rap',
       'rock', 'singer songwriter', 'soul','extraversion_high',
'extraversion_average', 'extraversion_low', 'agreeableness_high',
'agreeableness_average', 'agreeableness_low', 'conscientiousness_high',
'conscientiousness_average', 'conscientiousness_low',
'neuroticism_high', 'neuroticism_average', 'neuroticism_low',
'openness_high', 'openness_average', 'openness_low'
       ]

IPPdataset= Dataset()
IPPdataset.fit(
            users=usersDF['userID'].unique(),
            items=songsDF['songID'].unique(),
            item_features = item_features_columns,
            user_features= user_features_columns)
noIPPdataset = Dataset()
noIPPdataset.fit(
            users=usersDF['userID'].unique(),
            items=songsDF['songID'].unique(),
            item_features = item_features_columns[:16],
            user_features= user_features_columns[:10])

IPP_item_features_lookup = [(song_id, {column: value for column, value in zip(item_features_columns, features)}) for song_id, *features in songsDF[['songID'] + item_features_columns].itertuples(index=False)]
IPP_user_features_lookup = [(user_id, {column: value for column, value in zip(user_features_columns, features)}) for user_id, *features in usersDF[['userID'] +user_features_columns].itertuples(index=False)]

noIPP_item_features_lookup = [(song_id, {column: value for column, value in zip(item_features_columns, features)}) for song_id, *features in songsDF[['songID'] + item_features_columns[:16]].itertuples(index=False)]
noIPP_user_features_lookup = [(user_id, {column: value for column, value in zip(user_features_columns, features)}) for user_id, *features in usersDF[['userID'] +user_features_columns[:10]].itertuples(index=False)]

IPP_item_features_list = IPPdataset.build_item_features(IPP_item_features_lookup,normalize=True)
IPP_user_features_list = IPPdataset.build_user_features(IPP_user_features_lookup,normalize=True)

noIPP_item_features_list = IPPdataset.build_item_features(noIPP_item_features_lookup,normalize=True)
noIPP_user_features_list = IPPdataset.build_user_features(noIPP_user_features_lookup,normalize=True)

#Model Setup
with open("modelwithIPP.pickle", "rb") as file:
    IPPmodel = pickle.load(file)

with open("modelnoIPP.pickle", "rb") as file:
    noIPPmodel = pickle.load(file)


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
            'edm': 0,
            'country': 0,
            'indie': 0,
            'metal': 0,
            'pop': 0,
            'pop punk': 0,
            'rap': 0,
            'rock': 0,
            'singer songwriter': 0,
            'soul': 0,
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
            'atop1': '',
            'atop2': '',
            'atop3': '',
            'atop4': '',
            'atop5': '',
            'atop6': '',
            'atop7': '',
            'atop8': '',
            'atop9': '',
            'atop10': '',
            'btop1': '',
            'btop2': '',
            'btop3': '',
            'btop4': '',
            'btop5': '',
            'btop6': '',
            'btop7': '',
            'btop8': '',
            'btop9': '',
            'btop10': ''
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
            'extraversion_high': 0,
            'extraversion_avg': 0,
            'extraversion_low': 0,
            'agreeableness_high': 0,
            'agreeableness_avg': 0,
            'agreeableness_low': 0,
            'neuroticism_high': 0,
            'neuroticism_avg': 0,
            'neuroticism_low': 0,
            'conscientiousness_high': 0,
            'conscientiousness_avg': 0,
            'conscientiousness_low': 0,
            'openness_high': 0,
            'openness_avg': 0,
            'openness_low': 0,
            'edm': 0,
            'country': 0,
            'indie': 0,
            'metal': 0,
            'pop': 0,
            'pop punk': 0,
            'rap': 0,
            'rock': 0,
            'singer songwriter': 0,
            'soul': 0
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
                df_personality_profile['extraversion_low'] = 1
            elif i == 1:
                df_personality_profile['agreeableness_low'] = 1
            elif i == 2:
                df_personality_profile['openness_low'] = 1
            elif i == 3:
                df_personality_profile['conscientiousness_low'] = 1
            elif i == 4:
                df_personality_profile['neuroticism_low'] = 1
        elif self_ratings[i] >= 19 and self_ratings[i] <= 28:
            if i == 0:
                df_personality_profile['extraversion_avg'] = 1
            elif i == 1:
                df_personality_profile['agreeableness_avg'] = 1
            elif i == 2:
                df_personality_profile['openness_avg'] = 1
            elif i == 3:
                df_personality_profile['conscientiousness_avg'] = 1
            elif i == 4:
                df_personality_profile['neuroticism_avg'] = 1
        elif self_ratings[i] > 28:
            if i == 0:
                df_personality_profile['extraversion_high'] = 1
            elif i == 1:
                df_personality_profile['agreeableness_high'] = 1
            elif i == 2:
                df_personality_profile['openness_high'] = 1
            elif i == 3:
                df_personality_profile['conscientiousness_high'] = 1
            elif i == 4:
                df_personality_profile['neuroticism_high'] = 1

    for i in range(len(top_genres_user)):
        if top_genres_user[i] == 'edm':
            df_personality_profile['edm'] = 1
            df_new_users['edm'] = 1
        elif top_genres_user[i] == 'country':
            df_personality_profile['country'] = 1
            df_new_users['country'] = 1
        elif top_genres_user[i] == 'indie':
            df_personality_profile['indie'] = 1
            df_new_users['indie'] = 1
        elif top_genres_user[i] == 'metal':
            df_personality_profile['metal'] = 1
            df_new_users['metal'] = 1
        elif top_genres_user[i] == 'pop':
            df_personality_profile['pop'] = 1
            df_new_users['pop'] = 1
        elif top_genres_user[i] == 'poppunk':
            df_personality_profile['pop punk'] = 1
            df_new_users['pop punk'] = 1
        elif top_genres_user[i] == 'rap':
            df_personality_profile['rap'] = 1
            df_new_users['rap'] = 1
        elif top_genres_user[i] == 'rock':
            df_personality_profile['rock'] = 1
            df_new_users['rock'] = 1
        elif top_genres_user[i] == 'singersongwriter':
            df_personality_profile['singer songwriter'] = 1
            df_new_users['singer songwriter'] = 1
        elif top_genres_user[i] == 'soul':
            df_personality_profile['soul'] = 1
            df_new_users['soul'] = 1

    return df_personality_profile

def getLargestNumber():
    new_users = getCSV(connect_str, container_name, "new_users.csv")
    userID = new_users['userID'].tolist()
    userID_remove_duplicates = list(set(userID))
    userID_remove_U = [element.replace('U', '') for element in userID_remove_duplicates]
    userID_convert_int = [int(item) for item in userID_remove_U if item.isdigit()]
    userID_largest_number = max(userID_convert_int)
    return userID_largest_number

def newUserRecommendation(model, model_type, dataset, userID=None, new_user_feature=None, k=10):
    global df_new_users

    userID = getLargestNumber() + 1
    df_new_users['userID'] = 'U' +str(userID)
    mapper_to_internal_ids = dataset.mapping()[2]
    mapper_to_external_ids = {v: k for k, v in mapper_to_internal_ids.items()}
    new_user_features_lookup= 0
    if model_type == 'A':
        dataset.fit_partial(
                users=[userID],
                user_features= user_features_columns)
        new_user_features_lookup = [(userID, {column: value for column, value in zip(user_features_columns, new_user_feature.values())})]
    else:
        dataset.fit_partial(
                    users=[userID],
                    user_features= user_features_columns[:10])
        new_user_features_lookup = [(userID, {column: value for column, value in zip(user_features_columns[:10], new_user_feature.values())})]

    new_user_feature = dataset.build_user_features(new_user_features_lookup, normalize=True)

    userID_map = dataset.mapping()[0][userID]
    n_items = len(dataset.mapping()[2])
    scores = model.predict(userID_map, np.arange(n_items),new_user_feature)
    top_k_indices = np.argsort(-scores)[:k]
    recommended_song_ids = np.vectorize(mapper_to_external_ids.get)(top_k_indices)
    songs_recommended = songsDF[songsDF['songID'].isin(recommended_song_ids)]
      
    for row, i in zip(songs_recommended.itertuples(), range(1, 11)):
        song_id = row.songID
        df_col = str(model_type) +'top' +str(i)
        df_new_users[df_col] = song_id

    print(songs_recommended[['songID', 'name', 'artists']])

    return songs_recommended[['songID', 'name', 'artists']]

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
            
            recommendations = newUserRecommendation(IPPmodel, 'A', IPPdataset, new_user_feature=computePersonalityScore(q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,top_genres_user))
            recommendations2 = newUserRecommendation(noIPPmodel, 'B', noIPPdataset, new_user_feature=computePersonalityScore(q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,top_genres_user))

            htmltable_string = """
            <h4 class='text-center'> IPP Based Model Results </h4>
            <table class='table table-striped table-dark'> 
                <thead>
                    <tr> 
                        <th scope='col'>Rank</th>
                        <th scope='col'>Name</th> 
                        <th scope='col'>Artist</th> 
                        <th scope='col'>Spotify Preview</th>
                        <th scope='col'>Like/Dislike</th> 
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
                    <td> <a href='https://open.spotify.com/track/""" +row.songID +"""'> Link to Spotify </a> </td> 
                    <td> <center><input class='form-check-input' type='checkbox' name='like_atop""" +str(i) +"""' id='defaultCheck1'></center> </td>
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
                        <th scope='col'>Like/Dislike</th> 
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
                    <td> <a href='https://open.spotify.com/track/""" +row.songID +"""'> Link to Spotify </a> </td> 
                    <td> <center><input class='form-check-input' type='checkbox' name='like_btop""" +str(i) +"""' id='defaultCheck1'></center> </td>
                </tr>"""
            htmltable_string += "</tbody></table></form>"

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
            'aq1': int(request.form['aq1']),
            'aq2': int(request.form['aq2']),
            'aq3': int(request.form['aq3']),
            'bq1': int(request.form['bq1']),
            'bq2': int(request.form['bq2']),
            'bq3': int(request.form['bq3']),
            'lcq1': request.form['lcq1'],
            'lcq2': request.form['lcq2'],
            'like_atop1': '1' if request.form.get('like_atop1') else '0',
            'like_atop2': '1' if request.form.get('like_atop2') else '0',
            'like_atop3': '1' if request.form.get('like_atop3') else '0',
            'like_atop4': '1' if request.form.get('like_atop4') else '0',
            'like_atop5': '1' if request.form.get('like_atop5') else '0',
            'like_atop6': '1' if request.form.get('like_atop6') else '0',
            'like_atop7': '1' if request.form.get('like_atop7') else '0',
            'like_atop8': '1' if request.form.get('like_atop8') else '0',
            'like_atop9': '1' if request.form.get('like_atop9') else '0',
            'like_atop10': '1' if request.form.get('like_atop10') else '0',
            'like_btop1': '1' if request.form.get('like_btop1') else '0',
            'like_btop2': '1' if request.form.get('like_btop2') else '0',
            'like_btop3': '1' if request.form.get('like_btop3') else '0',
            'like_btop4': '1' if request.form.get('like_btop4') else '0',
            'like_btop5': '1' if request.form.get('like_btop5') else '0',
            'like_btop6': '1' if request.form.get('like_btop6') else '0',
            'like_btop7': '1' if request.form.get('like_btop7') else '0',
            'like_btop8': '1' if request.form.get('like_btop8') else '0',
            'like_btop9': '1' if request.form.get('like_btop9') else '0',
            'like_btop10': '1' if request.form.get('like_btop10') else '0'
            }  
        append_row_to_csv_blob(connect_str, container_name, "newusers_surveyresults.csv", df_newusers_surveyresults)
        return render_template("index.html")

if __name__ == "__main__":
  app.run(debug=True)

