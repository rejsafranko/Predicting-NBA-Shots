# Predicting NBA Shots

This project revolves around predicting whether a shot was successfully made during an NBA match. The predictive models showcased here are trained on a processed NBA Shot Logs dataset from the 2014-2015 season.

## Key Elements
Dataset: NBA Shot Logs, Season 14-15: The core of this analysis lies in the data captured during the 2014-2015 NBA season, providing the foundation for training and evaluating the predictive models.

Models: A diverse array of models have been employed to tackle the shot classification challenge: Logistic Regression, Support Vector Machine (SVM), Tabular Transformer, k-Nearest Neighbors (kNN) Classifier, Stochastic Gradient Descent (SGD) Classifier, Fully Connected Neural Networks (FCNN), XGBoost, Voting Classifier, AdaBoost, LightGBM, Random Forest, Naive Bayes.

## Project Structure
src/: Explore the training scripts in this directory, specifically ```train_[model]_[dataset]_[task].py```, to check the model training process. Build the train and test datasets with ```make_dataset.py```.

notebooks/: Gain deeper insights into model performance through the ```model_evaluation.ipynb``` notebook. Check the data exploration and data preprocessing steps in the ```data_preprocess.ipynb``` notebook.

Run ```python script.py -h``` for help.
