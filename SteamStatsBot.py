#!/usr/bin/env python
# coding: utf-8

# # <u>Similar Game Finder :D</u>
# ---------------------------------------------------------------------------------------------------------------
#                                 S I M I L A R   G A M E   F I N D E R
# ---------------------------------------------------------------------------------------------------------------

"""
**API REFERENCES**<
https://partner.steamgames.com/doc/webapi                          - Valve's steam API
https://wiki.teamfortress.com/wiki/User:RJackson/StorefrontAPI     - API Docs
https://steamapi.xpaw.me/#
https://steamspy.com/api.php                                       - Popularity and sales

**CODE REFERENCES**
https://nik-davis.github.io/posts/2019/steam-data-collection/
https://www.machinelearningplus.com/nlp/gensim-tutorial/#3howtocreateadictionaryfromalistofsentences  - gensim
"""


# standard library imports
import csv
import datetime as dt
import json
import os
import statistics
import time
from collections import Counter

# third-party imports
import numpy as np
import pandas as pd
import requests
from requests.exceptions import SSLError

# For regex
import re

# Make sure to install spacy, along with it's medium or large dataset
import spacy

# Make sure to install gensim
import gensim
from gensim.models import Word2Vec
from gensim import corpora
from gensim.test.utils import common_texts

# customisations - ensure tables show all columns
pd.set_option("max_columns", 100)

# ---------------------------------------------------------------------------------------------------------------
#                                 All-Purpose Function to Get API Requests
# ---------------------------------------------------------------------------------------------------------------

def get_request(url, parameters=None):
    """Return json-formatted response of a get request using optional parameters.

    Parameters
    ----------
    url : string
    parameters : {'parameter': 'value'}
        parameters to pass as part of get request

    Returns
    -------
    json_data
        json-formatted response (dict-like)
    """
    try:
        response = requests.get(url=url, params=parameters)
    except SSLError as s:
        print('SSL Error:', s)

        for i in range(5, 0, -1):
            print('\rWaiting... ({})'.format(i), end='')
            time.sleep(1)
        print('\rRetrying.' + ' ' * 10)

        # recusively try again
        return get_request(url, parameters)

    if response:
        return response.json()
    else:
        # response is none usually means too many requests. Wait and try again
        print('No response, waiting 10 seconds...')
        time.sleep(10)
        print('Retrying.')
        return get_request(url, parameters)

# ---------------------------------------------------------------------------------------------------------------
#                                    Generate List of App IDs Using SteamSpy
# ---------------------------------------------------------------------------------------------------------------

# url = "https://steamspy.com/api.php"
# parameters = {"request": "all"}

# request 'all' from steam spy and parse into dataframe
# json_data = get_request(url, parameters=parameters)
# steam_spy_all = pd.DataFrame.from_dict(json_data, orient='index')

# generate sorted app_list from steamspy data
# app_list = steam_spy_all[['appid', 'name']].sort_values('appid').reset_index(drop=True)

# export can be disabled to keep consistency across download sessions
# app_list.to_csv('../SimilarGameFinder/downloads/app_list.csv', index=False)

# instead read from stored csv
app_list = pd.read_csv('../SimilarGameFinder/downloads/kaggle_steam_dataset/steam.csv')

# display first few rows
app_list.head()

print("TOTAL DATABASE SIZE: ", len(app_list), "\n\n")


# ---------------------------------------------------------------------------------------------------------------
#                                           Define Download Logic
# ---------------------------------------------------------------------------------------------------------------

"""
Now that we have app_list, we can iterate over it to request individual app data from the srevers.
This is where we set the logic for that, and store the end data as a csv.
Since it takes a long time to retreive the data, we cannot attempt it all in one go.
We shall define a function to download and process the requests in batches, appending each one to an external file, and keeping track of the highest index in another.

This provides security (easy restart), and means we can complete the download over multiple sessions.
"""
def get_app_data(start, stop, parser, pause):
    """Return list of app data generated from parser.
    
    parser : function to handle request
    """
    app_data = []
    
    # iterate through each row of app_list, confined by start and stop
    for index, row in app_list[start:stop].iterrows():
        print('Current index: {}'.format(index), end='\r')
        
        appid = row['appid']
        name = row['name']

        # retrive app data for a row, handled by supplied parser, and append to list
        data = parser(appid, name)
        app_data.append(data)

        time.sleep(pause) # prevent overloading api with requests
    
    return app_data

def process_batches(parser, app_list, download_path, data_filename, index_filename,
                    columns, begin=0, end=-1, batchsize=100, pause=1):
    """Process app data in batches, writing directly to file.
    
    parser : custom function to format request
    app_list : dataframe of appid and name
    download_path : path to store data
    data_filename : filename to save app data
    index_filename : filename to store highest index written
    columns : column names for file
    
    Keyword arguments:
    
    begin : starting index (get from index_filename, default 0)
    end : index to finish (defaults to end of app_list)
    batchsize : number of apps to write in each batch (default 100)
    pause : time to wait after each api request (defualt 1)
    
    returns: none
    """
    print('Starting at index {}:\n'.format(begin))
    
    # by default, process all apps in app_list
    if end == -1:
        end = len(app_list) + 1
    
    # generate array of batch begin and end points
    batches = np.arange(begin, end, batchsize)
    batches = np.append(batches, end)
    
    apps_written = 0
    batch_times = []
    
    for i in range(len(batches) - 1):
        start_time = time.time()
        
        start = batches[i]
        stop = batches[i+1]
        
        app_data = get_app_data(start, stop, parser, pause)
        
        rel_path = os.path.join(download_path, data_filename)
        
        # writing app data to file
        with open(rel_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
            
            for j in range(3,0,-1):
                print("\rAbout to write data, don't stop script! ({})".format(j), end='')
                time.sleep(0.5)
            
            writer.writerows(app_data)
            print('\rExported lines {}-{} to {}.'.format(start, stop-1, data_filename), end=' ')
            
        apps_written += len(app_data)
        
        idx_path = os.path.join(download_path, index_filename)
        
        # writing last index to file
        with open(idx_path, 'w') as f:
            index = stop
            print(index, file=f)
            
        # logging time taken
        end_time = time.time()
        time_taken = end_time - start_time
        
        batch_times.append(time_taken)
        mean_time = statistics.mean(batch_times)
        
        est_remaining = (len(batches) - i - 2) * mean_time
        
        remaining_td = dt.timedelta(seconds=round(est_remaining))
        time_td = dt.timedelta(seconds=round(time_taken))
        mean_td = dt.timedelta(seconds=round(mean_time))
        
        print('Batch {} time: {} (avg: {}, remaining: {})'.format(i, time_td, mean_td, remaining_td))
            
    print('\nProcessing batches complete. {} apps written'.format(apps_written))


"""
Next, we need functions to handle and prepare the external files

**reset_index** is used for testing and demonstration; setting the index in the stored file to 0 will restart
the download process

**get_index** retrieves the index from file.

**prepare_data_file** readies the CSV for storing data. If index is 0, we need a blank csv. Otherwise, leave CSV alone.
"""
def reset_index(download_path, index_filename):
    """Reset index in file to 0."""
    rel_path = os.path.join(download_path, index_filename)
    
    with open(rel_path, 'w') as f:
        print(0, file=f)

def get_index(download_path, index_filename):
    """Retrieve index from file, returning 0 if file not found."""
    try:
        rel_path = os.path.join(download_path, index_filename)

        with open(rel_path, 'r') as f:
            index = int(f.readline())
    
    except FileNotFoundError:
        index = 0
        
    return index

def prepare_data_file(download_path, filename, index, columns):
    """Create file and write headers if index is 0."""
    if index == 0:
        rel_path = os.path.join(download_path, filename)

        with open(rel_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()

# ---------------------------------------------------------------------------------------------------------------
#                                           Download Steam Data
# ---------------------------------------------------------------------------------------------------------------
# - NOTE: Currently only using 2000 games from the 27000 game database.


def parse_steam_request(appid, name):
    """Unique parser to handle data from Steam Store API.
    
    Returns : json formatted data (dict-like)
    """
    url = "http://store.steampowered.com/api/appdetails/"
    parameters = {"appids": appid}
    
    json_data = get_request(url, parameters=parameters)
    json_app_data = json_data[str(appid)]
    
    if json_app_data['success']:
        data = json_app_data['data']
    else:
        data = {'name': name, 'steam_appid': appid}
        
    return data


# Set file parameters
download_path = './downloads/kaggle_steam_dataset'
steam_description_data = 'steam_app_data.csv'
steam_index = 'steam_index.txt'

steam_columns = [
    'type', 'name', 'steam_appid', 'required_age', 'is_free', 'controller_support',
    'dlc', 'detailed_description', 'about_the_game', 'short_description', 'fullgame',
    'supported_languages', 'header_image', 'website', 'pc_requirements', 'mac_requirements',
    'linux_requirements', 'legal_notice', 'drm_notice', 'ext_user_account_notice',
    'developers', 'publishers', 'demos', 'price_overview', 'packages', 'package_groups',
    'platforms', 'metacritic', 'reviews', 'categories', 'genres', 'screenshots',
    'movies', 'recommendations', 'achievements', 'release_date', 'support_info',
    'background', 'content_descriptors'
]


###### NOTE: DO NOT RUN COMMENTED LINES UNLESS YOU WANT TO KEEP ITERATING THROUGH GAME LIST AND WRITING TO CSV
# (Long Process)

# Overwrites last index for demonstration (would usually store highest index so can continue across sessions)
# reset_index(download_path, steam_index)

# Retrieve last index downloaded from file
# index = get_index(download_path, steam_index)
#
# # Wipe or create data file and write headers if index is 0
# prepare_data_file(download_path, steam_description_data, index, steam_columns)
#
# # Set end and chunksize for demonstration - remove to run through entire app list
# process_batches(
#     parser=parse_steam_request,
#     app_list=app_list,
#     download_path=download_path,
#     data_filename=steam_description_data,
#     index_filename=steam_index,
#     columns=steam_columns,
#     begin=index,
#     end=len(app_list),
#     batchsize=1000
# )

# inspect downloaded data
steam_game_data = pd.read_csv(download_path + '/' + steam_description_data)
steam_game_data.head()
print("Current Sample Size: ", len(steam_game_data), "\n\n")

# ---------------------------------------------------------------------------------------------------------------
#                                           Downloading SteamSpy Data
# ---------------------------------------------------------------------------------------------------------------
#  - NOTE: Currently only using 2000 games from the 27000 game database.

def parse_steamspy_request(appid, name):
    """Parser to handle SteamSpy API data."""
    url = "https://steamspy.com/api.php"
    parameters = {"request": "appdetails", "appid": appid}
    
    json_data = get_request(url, parameters)
    return json_data

steamspy_data = 'steamspy_data.csv'
steamspy_index = 'steamspy_index.txt'

steamspy_columns = [
    'appid', 'name', 'developer', 'publisher', 'score_rank', 'positive',
    'negative', 'userscore', 'owners', 'average_forever', 'average_2weeks',
    'median_forever', 'median_2weeks', 'price', 'initialprice', 'discount',
    'languages', 'genre', 'ccu', 'tags'
]


# ##### NOTE: DO NOT RUN THIS CODE BLOCK UNLESS YOU WANT TO KEEP ITERATING THROUGH GAME LIST AND WRITING TO CSV
# (Long Process)

# ONLY UNCOMMENT THE LINE BELOW TO RESTART THE PROCESS OF WRITING ALL THE GAME DATA TO CSV
# reset_index(download_path, steamspy_index)

# index = get_index(download_path, steamspy_index)
#
# # Wipe data file if index is 0
# prepare_data_file(download_path, steamspy_data, index, steamspy_columns)
#
# process_batches(
#     parser=parse_steamspy_request,
#     app_list=app_list,
#     download_path=download_path,
#     data_filename=steamspy_data,
#     index_filename=steamspy_index,
#     columns=steamspy_columns,
#     begin=index,
#     end=len(app_list),
#     batchsize=1000,
#     pause=0.1
# )


# inspect downloaded steamspy data
all_steamspy_game_data = pd.read_csv(download_path + '/' + steamspy_data)
all_steamspy_game_data.head()
# print(len(all_steamspy_game_data))

# ---------------------------------------------------------------------------------------------------------------
#                                           Gensim Word Embeddings
# ---------------------------------------------------------------------------------------------------------------
#  - Not currently needed, may come in handy in the future

# # Make directory if it does not exist
# path = "./models/"

# if os.path.exists(path) == False:
#     try:
#         os.mkdir(path)
#     except OSError:
#         print ("Creation of the directory %s failed" % path)
#     else:
#         print ("Successfully created the directory %s " % path)
        
# filename = './downloads/GoogleNews-vectors-negative300.bin.gz'
# google_news_model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)
# google_news_model.save("./models/google_news.model")

# ---------------------------------------------------------------------------------------------------------------
#                                           Get Game Descriptions
# ---------------------------------------------------------------------------------------------------------------
# Not currently in use, but is planned to be used for comparing a given description to other game descriptions using NLP

game_description_list = []

# TODO: VECTORIZE
for description in steam_game_data['detailed_description']:
    game_description_list.append(description)

# ---------- Debug ------------
# print(game_description_list[0])
# for d in game_description_list:
#     if "sport" in d:
#         print(d)
# -----------------------------

# Tokenize (split each game's desecription into words)
steam_game_texts = [[text for text in description.split()] for description in game_description_list]
# Create dictionary
steam_game_dictionary = corpora.Dictionary(steam_game_texts)
# print(steam_game_dictionary)

steam_corpus = [steam_game_dictionary.doc2bow(doc, allow_update=True) for doc in steam_game_texts]
word_counts = [[(steam_game_dictionary[id], count) for id, count in line] for line in steam_corpus]

# ---------------------------------------------------------------------------------------------------------------
#                                   Create Similarity Vectors Manually
# ---------------------------------------------------------------------------------------------------------------

# Grab tags column
tags = all_steamspy_game_data['tags']

# Create a list of all UNIQUE game tags
taglist = []

# TODO: VECTORIZE THIS CODE
for tag in tags:
    current_game_tags = re.findall("'([A-Za-z&\s'\-0-9]*)'", tag)
    for current_tag in current_game_tags:
        if current_tag in taglist:
            continue
        taglist.append(current_tag)

print("Full List of Unique Tags:\n", taglist, "\n\n")


# TODO: VECTORIZE

"""
Here, we are going through our unique taglist, counting the number of times a given tag appears with another tag. 
While doing this, we also count the total number of appearances of a given tag, stored in tag_appearances.
With that information, we then go through tag_similarities (stores a tag and the number of times it appears
with every other tag), dividing the number of times a given tag has appeared with another tag by the total appearances
of that tag.

E.g: If 'Action' appeared with 'FPS' 100 times, and 'Action' itself appeared 200 times, the similarity vector for 
'Action' and 'FPS' would be 0.5.
"""

tag_similarities = {}
tag_appearances = {}

count = 0
for tag in taglist:
    for game_tags in all_steamspy_game_data['tags']:
        if tag in game_tags:
            if not tag in tag_appearances:
                tag_appearances[tag] = 1
            else:
                tag_appearances[tag] += 1
            if not tag in tag_similarities:
                tag_similarities[tag] = {}
            current_tag_group = re.findall("'([A-Za-z&\s'\-0-9]*)'", game_tags)
            if tag in current_tag_group:
                current_tag_group.remove(tag)
            for current_tag in current_tag_group:
                if current_tag in tag_similarities[tag]:
                    tag_similarities[tag][current_tag] += 1
                else:
                    tag_similarities[tag][current_tag] = 1
                
    count += 1

for tag in tag_similarities:
    for sub_tag in tag_similarities[tag]:
        current_tag_correlations = tag_similarities[tag][sub_tag]
    for sub_tag in tag_similarities[tag]:
        current_tag_correlations = tag_similarities[tag][sub_tag]
        tag_similarities[tag][sub_tag] = current_tag_correlations / tag_appearances[tag]

# Printing first 3 similarity vectors
print("\nEXAMPLE; Some Similarity Vectors For The Tag 'Action':")
count = 0
for tag in tag_similarities:
    if count > 1:
        break;
    print(tag, ":")
    for sub_tag in tag_similarities[tag]:
        if count == 10:
            break;
        print(sub_tag, " --> ", tag_similarities[tag][sub_tag])
        count += 1
    print("\n-----------------------------------------------------------------\n")
    count += 1


# Tags we will give to our game
our_tags = ["Puzzle platformer", "Horror", "Story rich", "Dark", "2D", "Platformer", "Puzzle"]

# ONLY compare to games that include this list of tags. If empty, comapre to all.
must_include = ["2D"]

# TODO: Implement weights ?
weights = []

game_similarities = {}
current_game_similarity_score = 0
count = 0
continue_comparing = True;

"""
Go through all games in our csv, incrementing current_game_similarity_score by the corresponding similarity vector stored in 
tag_similarities, for each tag in our_tags, to each tag of a game. 
"""
for current_game_tags in all_steamspy_game_data['tags']:
    current_game_tags_tidy = re.findall("'([A-Za-z&\s'\-0-9]*)'", current_game_tags)
    if must_include:
        for must_include_tag in must_include:
            if not must_include_tag in current_game_tags_tidy:
                continue_comparing = False;
                break;
    if continue_comparing == False:
        continue_comparing = True;
        current_game_similarity_score = 0
        game_similarities[all_steamspy_game_data['name'][count]] = current_game_similarity_score
        count += 1
        continue;
    for our_tag in our_tags:
        if our_tag in tag_similarities:
            if our_tag in current_game_tags_tidy:
                current_game_similarity_score += 1
            for current_game_tag in current_game_tags_tidy:
                if current_game_tag in tag_similarities[our_tag]:
                    current_game_similarity_score += tag_similarities[our_tag][current_game_tag]
    game_similarities[all_steamspy_game_data['name'][count]] = current_game_similarity_score
    current_game_similarity_score = 0
    count += 1
    
# Printing the 20 most similar games
k = Counter(game_similarities)
high_similarity = k.most_common(20)

print("Top 20 Most Similar Games Were:\n")
for game in high_similarity:
    if game[1] > 0:
        print(game[0], " --> ", game[1])


#  ### IMPROVEMENTS
#  - Add WEIGHTS that user can define for each tag?
#  - Be more lenient towards similar tags? (e.g, "2D" and "2D Platformer")
#  - Add option to filter date created?
#  - Add Follower/Wishlist/Estimated Revenue
#  - Add option to add a description; use NLP to compare similarity?

# # ----------------------------------------------------------------------------------------------------------

# # Other Solutions
# Empty for now.
