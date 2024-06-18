# Genre-Based Movie Recommender System using K-Nearest Neighbors Algorithm

## Overview

This repository contains the implementation of a genre-based movie recommender system using the K-Nearest Neighbors (KNN) algorithm. The system focuses on genre-based similarity rather than collaborative filtering techniques to ensure privacy preservation while providing personalized movie recommendations.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Future Work](#future-work)
- [Acknowledgements](#acknowledgements)

## Introduction

Recommendation systems are crucial in today's digital age due to the overwhelming amount of information available and the diverse preferences of users. This project involves content-based filtering, where the system applies a K-Nearest Neighbors (KNN) algorithm based on genre similarity to offer personalized movie recommendations.

## Features

- **Genre-based movie recommendations**
- **Privacy-preserving approach**
- **User-friendly interface built with Flask**
- **Exploratory Data Analysis (EDA) with visualizations**

## Installation

To get started with this project, follow these steps:

1. **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/movie-recommender-system.git
    cd movie-recommender-system
    ```

2. **Create and activate a virtual environment:**
    ```bash
    python3 -m venv env
    source env/bin/activate
    ```

3. **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the Flask application:**
    ```bash
    flask run
    ```

## Usage

1. **Start the Flask server:**
    ```bash
    flask run
    ```

2. **Open your web browser and navigate to:**
    ```
    http://127.0.0.1:5000/
    ```

3. **Select a movie:**
    - Choose a movie from the dropdown list to get genre-based recommendations.
    
4. **Get recommendations:**
    - Click the "Get Recommendations" button to see movies with similar genres.

## Dataset

The dataset used in this analysis comprises movie information sourced from MovieLens, focusing on movie data with details on 9,742 movies, including a unique movie ID, the movie’s title, and its associated genres.

### Data Collection and Preprocessing

- **Source:** MovieLens dataset
- **Contents:** 9,742 movies with IDs, titles, and genres
- **Preprocessing:** Checked for anomalies, ensured data integrity, no missing values or duplicates

## Methodology

### Algorithm Description

- **K-Nearest Neighbors (KNN):**
  - Utilizes cosine similarity to measure genre-based similarity
  - Brute-force approach for finding nearest neighbors

### Cosine Similarity

Cosine similarity between two vectors A and B is calculated using the formula:

cosine similarity (A, B) = (A · B) / (||A|| * ||B||)

Where:
- A · B: Dot product of vectors A and B
- ||A|| and ||B||: Euclidean norms of vectors A and B

### Exploratory Data Analysis (EDA)

- **Pie Chart:** Distribution of movie genres
- **Word Cloud:** Visual representation of genre frequency
- **Bar Chart:** Common genre combinations

### Visualization

- **Homepage:** Displays movie selection interface
- **Recommendation Page:** Shows recommended movies based on selected genre

## Future Work

- **Advanced Algorithms:** Explore transformers, reinforcement learning, and GANs
- **Integration:** Combine with NLP and other computer vision techniques
- **Scalability:** Utilize cloud platforms for larger datasets
- **Applications:** Implement in chatbots, virtual assistants, and recommendation systems
- **User Experience:** Develop intuitive web or mobile applications

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

Special thanks to the open-source community and the authors of the pre-trained ResNet model. This project was made possible by the contributions and support of various developers and researchers in the field of deep learning.

