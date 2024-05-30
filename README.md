# NBA Shot Prediction System

[STILL IN DEVELOPMENT] ❗❗❗

## Introduction

Welcome to the NBA Shot Prediction System project, a machine learning engineering project which focuses on 3 key aspects:
- **Model development**
- **Model deployment**
- **MLOps infrastructure**

## Project Overview

This project aims to predict whether an NBA shot will be made or missed using the NBA shot logs dataset. The project is divided into two main parts: Model Development and Application.

- **Model Development**: This directory follows the cookiecutter data science template and is used for data exploration, analysis, cleaning, training machine learning models, and evaluating them offline.
  
- **Application**: This directory is set up locally using Docker Compose and consists of three services:
  - **Flask Server**: A server for making predictions based on the trained model.
  - **PostgreSQL Database**: A database for storing data.
  - **Airflow**: A workflow management service. Airflow runs daily to check for at least 100 new inputs, retrains the model if the condition is met, stores the updated model in an S3 bucket, and updates the prediction model.

## Services

### Flask Server

- **Purpose**: Handles incoming prediction requests and returns the prediction results.
- **Port**: 5000
- **Endpoint**: `/predict` (Accepts POST requests with features for prediction)

### PostgreSQL Database

- **Purpose**: Stores prediction data and features.
- **Port**: 5432
- **Initial Setup**: Database is seeded with initial data from a CSV file.

### Airflow

- **Purpose**: Manages the workflow for model retraining.
- **Port**: 8080
- **Functionality**: Checks daily if there are at least 100 new inputs. If so, it retrains the model, uploads it to S3, and updates the model used by the Flask server.
