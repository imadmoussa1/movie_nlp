import csv
import json
import pandas as pd
import numpy as np
import re
import shutil
import os
from clean_data import cleaner

# Read the Movies CSV data set in Dataframe,
# we used the converters for genres to loaded as List field
movies_df = pd.read_csv("movies_metadata.csv", converters={'genres': eval}, low_memory=False)
# Choosing the fields will be using in this project (genres, overview, title)
movies_df = movies_df[['genres', 'overview', 'title']]
# Getting the first genre from the list of genres in the data set
movies_df["genres"] = movies_df["genres"].apply(lambda x: [str(i['name']) for i in x] if len(x) > 0 else np.nan)
# Drop the row with Empty genres ( not useful in training)
movies_df = movies_df.dropna()
# Adding title of the movie to the description
movies_df['overview'] = movies_df['title'].astype(str).str.cat(movies_df['overview'], sep=' ')
# Clean the data before training
movies_df['overview'] = movies_df['overview'].apply(cleaner)
# Augment the data so each overview have all the possible genres
movies_df = movies_df.explode('genres')
# Get the Unique values of genres to be our genres Taxonomy
genres_label = movies_df.genres.unique()
# Print the count of each genre
print(genres_label)
print(movies_df)
print(movies_df['genres'].value_counts())

# Save the datframe in Data Folder with text files for each Genre
# This format will be use in The training model, faster loading in Tenserflow
os.mkdir("data/")
for genre in genres_label:
  df_train = movies_df[movies_df['genres'] == genre]
  file_name = "%s/%s.txt" % ("data/", genre)
  df_train["overview"].to_csv(file_name, header=None, index=None, sep='\t', mode='a')

# This simple csv format to be used in TFX example
movies_df.genres.replace(genres_label, [0, 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], inplace=True)
movies_df.set_index('genres', inplace=True)
movies_df.to_csv("tfx/data/movies_genre.csv", ",")
