# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 06:50:19 2024

@author: CC
"""

from flask import Flask, render_template, request
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MultiLabelBinarizer
import logging
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from collections import Counter

app = Flask(__name__, template_folder=r"C:\Users\CC\Documents\Mv recommender\Temp", static_folder=r"C:\Users\CC\Documents\Mv recommender\backdrop")


# Loading dataset
movies = pd.read_csv(r"C:\Users\CC\Documents\Mv recommender\ml-latest-small\movies.csv")

movies

movies.describe()

movies.info()

duplicates = movies[movies.duplicated()]
print(duplicates)

# Checking shapes of dataframes
print("Movies shape:", movies.shape)

# Combining genres into a single string
genres_text = ' '.join(movies['genres'].explode())

# Generating a word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(genres_text)
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Movie Genres Word Cloud')
plt.show()

# Calculating the frequency of each genre
genre_counts = movies.explode('genres')['genres'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(genre_counts, labels=genre_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Movie Genres')
plt.axis('equal')  
plt.show()


movies['genres'] = movies['genres'].apply(lambda x: x.split('|'))


#Using MultiLabelBinarizer to encode genres
mlb = MultiLabelBinarizer()
genres_encoded = mlb.fit_transform(movies['genres'])
movies_encoded = pd.concat([movies, pd.DataFrame(genres_encoded, columns=mlb.classes_)], axis=1)

# Calculate the percentage of movies with no genre listed
percentage_no_genre = genres_encoded.count("(no genres listed)") / len(genres_encoded) * 100

# Print the percentage
print(f"Percentage of movies with no genre listed: {percentage_no_genre:.2f}%")
# Creating a list to store all pairs of genres
genre_pairs = []

# Iterating over the list of genres for each movie
for genres in movies['genres']:
    pairs = list(combinations(genres, 2))
    genre_pairs.extend(pairs)

# Counting the frequency of each pair
pair_counts = Counter(genre_pairs)

# Converting to a dataframe
pair_counts_df = pd.DataFrame(pair_counts.items(), columns=['Pair', 'Count'])

# Sorting the dataframe 
pair_counts_df = pair_counts_df.sort_values('Count', ascending=False)

# Plotting the most common genre pairs
plt.figure(figsize=(10, 6))
sns.barplot(x='Count', y='Pair', data=pair_counts_df.head(10), palette='coolwarm')
plt.title('Most Common Genre Pairs')
plt.xlabel('Count')
plt.ylabel('Genre Pair')
plt.show()

# Create a KNN model 
knn_data = movies_encoded.iloc[:, 3:]
knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
knn_model.fit(knn_data)


def get_genre_recommendations(movie_title, n_recommendations=5):
    movie_index = movies_encoded[movies_encoded['title'] == movie_title].index[0]
    _, indices_genres = knn_model.kneighbors([knn_data.iloc[movie_index]])
    recommendations = [movies_encoded.iloc[idx]['title'] for idx in indices_genres.flatten() if idx != movie_index][:n_recommendations]
    return recommendations


@app.route('/')
def home():
    movie_titles = movies['title'].tolist()
    return render_template("index.html", movie_titles=movie_titles)

@app.route('/recommend', methods=['POST'])
def recommend():
   # try:
        selected_movie = request.form['movie']
        selected_movie_genre = movies[movies['title'] == selected_movie]['genres'].values[0]
        recommendations = get_genre_recommendations(selected_movie)
        recommendation_genres = [movies[movies['title'] == movie]['genres'].values[0] for movie in recommendations]
        return render_template("recommendations.html", movie=selected_movie, genre=selected_movie_genre, recommendations=list(zip(recommendations, recommendation_genres)))

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
