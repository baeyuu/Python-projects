# MY PYTHON PROJECTS

### Table of Content


- [Project 1](Project-1)

## PROJECT 1
Machine Learning Project: Building Insurance Prediction

### Project Overview
This project aim is to predict whether a building will require an insurance claim based on various features. Here's a breakdown of the approach

### Data Source
train_data: The Primary dataset used for this analysis is the "train_data.csv" which which was gotten from kaggle.

### Tools
- Programming Language: Python a popular choice for machine learning projects with libraries like pandas for data manipulation, scikit-learn for various machine learning algorithms, and TensorFlowwere used.

### Data Cleaning and Preparation

1. Data Cleaning: Handled missing values (e.g., impute numerical values, remove rows with too many missing values). Address inconsistencies in data formats (e.g., standardize date formats).
2. Feature Engineering:
  - Categorical Features: Encodeed categorical features like Residential, Building_Painted, Building_Fenced, and Garden using techniques like one-hot encoding or label encoding.
  - Date Features: Extracted useful information from Year Of Observation and Date_of_Occupancy like age of the building or occupancy duration.
  - Geographical Features: Geo_Code were translated into geographical coordinates (latitude, longitude), consider creating new features like population density or proximity to natural disaster zones.
  - Feature Scaling: Standardized numerical features like Building Dimension and Number Of Windows to a common scale for better model performance.


### Exploratory Data Analysis

The EDA involved exploring the dataset to answer key questions such as:
- The relationship beteween claim and Building type, year of obseration and other categorical data.
- The relationship beteween claim and numerical data such as insured period, Building dimensions, etc
- Overview of the train and test datasets

### Model Selection and Training:

1. Split Data: Divide your data into training, validation, and test sets. The training set will be used to train the model, the validation set for hyperparameter tuning, and the test set for final evaluation.
2. Choose Models: Here are some potential models to consider:
- Decision tree: A classic model for binary classification problems like claim prediction.
- Random Forest: A robust ensemble method that can handle various data types and non-linear relationships.
- K Nearest Neighbours (KNN): Another powerful ensemble method known for handling complex relationships and high dimensionality.
3. Train and Evaluate: Train each model using the training set and fine-tune hyperparameters using the validation set. Evaluate the models on the unseen test set using metrics like accuracy, precision, recall, and F1-score.



Includes some interesting code features

``` Python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set(style = 'whitegrid', color_codes = True)

from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix 
from sklearn.neighbors import KNeighborsClassifier
```
### Result and Findings

The Analysis results are summarized thus:
- The mean accuracy of Decision tree classifier is "0.68" which is very low
- The mean accuracy of random forest is "0.77" which is the highest gotten
- The mean accuracy of K nearest neighbour classifier is "0.759" which is relatively low


### Recommendation
Based on the analysis the following recommendations should be considered by the company
The larger the training data set the more and immproved the accuracy will become. it is however important to get a prediction of 99.5 that is statisticall recommended there is a need to improve the training data set.

### Additional Considerations:

- Imbalanced Data: The dataset has a significant imbalance between Test and training data, claims and no claims data, undersampling the majority class was hereby inevitable.
- Feature Selection: Exploring feature selection techniques like LASSO regression or feature importance to identify the most relevant features and potentially improve model performance and interpretability will be recommended for future work.

### Refrences

1. [Kaggle](https://www.kaggle.com/datasets/kunalgupta2616/hackerearth-customer-segmentation-hackathon)
2. [Chatgpt](https://chat.openai.com/)
