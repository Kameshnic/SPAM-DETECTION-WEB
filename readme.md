# Spam Detection Web App

## Overview

The Spam Detection Web App is an intelligent system designed to identify and classify messages as either **spam** or **ham** (non-spam). Utilizing advanced machine learning techniques, specifically a Logistic Regression model paired with TF-IDF vectorization, this web application provides users with a seamless interface to check the nature of their messages. This project is particularly beneficial for individuals and organizations seeking to filter spam messages effectively.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Training the Model](#training-the-model)
- [Running the Flask App](#running-the-flask-app)
- [Usage](#usage)
- [Directory Setup](#directory-setup)
- [Screenshots](#screenshots)
- [License](#license)

## Features

- **Machine Learning**: Detects whether a message is spam or ham using a Logistic Regression model.
- **User-Friendly Interface**: Clean and intuitive web interface for easy interaction.
- **Real-Time Feedback**: Displays classification results immediately after message submission.
- **Robust Performance**: High accuracy achieved through thorough training on an SMS spam dataset.

## Project Structure

```plaintext
/spam-detection
│
├── app.py                         # Main Flask application file
├── spam_model.pkl                  # Trained Logistic Regression model
├── tfidf_vectorizer.pkl            # Saved TF-IDF Vectorizer
├── spam.csv                        # Dataset containing Spam/Ham messages
├── static
│   ├── style.css                   # CSS for styling the frontend
│   └── app.js                      # JavaScript for client-side logic
├── templates
│   └── index.html                  # HTML template for the main page
├── train_model.py                  # Script to train the model and save it as a .pkl file
└── README.md                       # Project documentation file
