# NBA Shot Prediction System

[STILL IN DEVELOPMENT] ❗❗❗

## Table of Contents
- [Introduction](#introduction)
- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Application](#application)
- [Model Development](#model-development)
- [How to Use](#how-to-use)
- [License](#license)

## Introduction

Welcome to the NBA Shot Prediction System project, a machine learning engineering project which focuses on 3 key aspects:
- **Model development**
- **Model deployment**
- **MLOps infrastructure**

## Project Overview

This project aims to predict whether an NBA shot will be made or missed using the NBA shot logs dataset. The project is divided into two main directories: ```model-development``` and ```application```.

- **model-development**: This directory follows the cookiecutter data science template and is used for data exploration, analysis, cleaning, training machine learning models, and evaluating them offline.
  
- **application**: This directory contains the MLOps infrastructure as a Docker Compose consisting of three services:
  - **Flask Server**: A server for making predictions based on the trained model.
  - **PostgreSQL Database**: A database for storing data.
  - **Airflow**: A workflow management service. Airflow runs daily to check for at least 100 new inputs, retrains the model if the condition is met, stores the updated model in an S3 bucket, and updates the prediction model.

## Application

### Flask Server

- **Purpose**: Handles incoming prediction requests and returns the prediction results.
- **Port**: 5000
- **Endpoint**: `/predict` (Accepts POST requests with features for prediction)

### PostgreSQL Database

- **Purpose**: Stores data features and labels.
- **Port**: 5432
- **Initial Setup**: Database is seeded with initial data from a CSV file.

### Airflow

- **Purpose**: Manages the workflow for model retraining.
- **Port**: 8080
- **Functionality**: Checks daily if there are at least 100 new inputs. If so, it retrains the model and uploads it to S3 to be used by the Flask server predict endpoint.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
