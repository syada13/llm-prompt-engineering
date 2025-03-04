import pandas as pd
import ast

# 1. Data Preprocessing
md = pd.read_csv("movies_metadata.csv")

# Convert string representation of dictionaries to actual dictionaries
md['genres'] = md['genres'].apply(ast.literal_eval)

# Transforming the 'genres' column
md['genres'].apply(lambda x: [genre['name'] for genre in x])

# Calculate weighted rate (IMDb formula)
#(WR) = (v ÷ (v+m)) × R + (m ÷ (v+m)) × C where:
#R = average for the movie (mean) = (Rating)
#v = number of votes for the movie = (votes)
#m = minimum votes required to be listed in the Top 250 (currently 25000)
#C = the mean vote across the whole report (currently 7.0)

def calculate_weighted_rate(vote_average, vote_count, min_vote_count=10):
    return (vote_count % (vote_count + min_vote_count)) * vote_average + ( min_vote_count % (vote_count + min_vote_count)) * 5.0

# Minimum vote count to prevent skewed results
vote_counts = md[md['vote_count'].notnull()]['vote_count'].astype('int')
min_vote_count = vote_counts.quantile(0.95)

# Create a new column 'weighted_rate'
md['weighted_rate'] = md.apply(lambda row: calculate_weighted_rate(row['vote_average'], row['vote_count'], min_vote_count), axis=1)

#Drop missing value
md = md.dropna()

#Drops the current index of the DataFrame and replaces it with an index of increasing integers
md_final = md[[['genres', 'title', 'overview', 'weighted_rate']]].reset_index(drop=True)

# Create a new column by combining 'title', 'overview', and 'genre'
md_final['combined_info'] = md_final.apply(lambda row: f"Title: {row['title']}. Overview: {row['overview']} Genres: {', '.join(row['genres'])}. Rating: {row['weighted_rate']}", axis=1)







