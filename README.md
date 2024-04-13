# MY PYTHON PROJECTS

### Table of Content

- [Project 1](Project1)
- [Project 2](Project2)

## PROJECT 1
Machine Learning Project: Building Insurance Prediction

### Project Overview
This Project aims to predict whether a building will require an insurance claim based on various features. Here's a breakdown of the approach

### Data Source
train_data: The Primary dataset used for this analysis is the "train_data.csv, " which was obtained from kaggle.

### Tools
- Programming Language: Python a popular choice for machine learning projects with libraries like pandas for data manipulation, Matplotlib, seaborn, sci-kit-learn for various machine learning algorithms, and TensorFlow were used.

### Data Cleaning and Preparation

1. Data Cleaning: Handled missing values (e.g., input numerical values, remove rows with too many missing values). Address inconsistencies in data formats (e.g., standardised date formats).
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





## Project 2

EDA (Exploratory Data Analysis) of an Automobile Dataset.

### Project Overview

This project aims to perform comprehensive data analysis on automobile data to gain insights and understand the relationships between various features and automobile prices.

### Data Source
Automobile_data: The Primary dataset used for this analysis is the "Automobile_data.csv, " which was obtained from kaggle.

### Data Columns:

- symboling: Categorical - Relative cost/prestige of the car.
- normalized-losses: Numeric - Insurance risk rating.
- make: Categorical - Car manufacturer.
- fuel-type: Categorical - Type of fuel used (gas, diesel, etc.).
- aspiration: Categorical - Engine aspiration (standard, turbo, etc.).
- num-of-doors: Categorical - Number of doors.
- body-style: Categorical - Car's body style (sedan, hatchback, etc.).
- drive-wheels: Categorical - Drivetrain (front-wheel drive, rear-wheel drive, etc.).
- engine-location: Categorical - Location of the engine (front, rear).
- wheel-base: Numeric - Distance between the front and rear axles.
- engine-size: Numeric - Engine displacement in cubic centimetres.
- fuel-system: Categorical - Fuel delivery system (multi-point injection, carburettor, etc.).
- borestroke: Categorical - Engine bore and stroke measurements.
- compression-ratio: Numeric - Engine's compression ratio.
- horsepower: Numeric - Maximum engine horsepower.
- peak-rpm: Numeric - Engine's revolutions per minute at peak horsepower.
- city-mpg: Numeric - Car's fuel efficiency in miles per gallon (city driving).
- highway-mpg: Numeric - Car's fuel efficiency in miles per gallon (highway driving).
- price: Numeric - Car's price in US dollars.

### Tools
- Python 3.x
- Jupyter Notebook
- Pandas
- Matplotlib
- Seaborn
- Numpy

### Data Cleaning
- Identified and handled missing values in the dataset by replacing numerical values with mean and categorical values with mode.
- Checked for outliers that could skew the analysis.
- Standardize and clean data formats to ensure consistency across all columns.

### Exploratory Data Analysis

The EDA involved exploring the dataset to carry out the following activities such as:
- Performed univariate analysis to understand the distribution of individual variables.
- Conducted bivariate analysis to identify relationships between pairs of variables.
- Analyzing correlations between numerical variables using correlation matrices

### Data Visualization

This was done to visually represent the data using graphs and charts to facilitate a better understanding of the patterns and trends by doing the following.

- Created histograms, bar charts, and pie charts to visualize the distributions and proportions of categorical variables.
- Generated scatter plots and line graphs to visualize relationships between numerical variables.
- Designed heatmaps and correlation matrices to visualize correlations between variables.

Includes some interesting code features

``` Python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize']=(23,10)
ax = sns.boxplot(x="make", y="price", data=df_automobile)

sns.catplot(data=df_automobile, y="normalized-losses", x="symboling" , hue="body-style" ,kind="point")
```
### Result and Findings

The Analysis results are summarized thus:
- Most of the cars company produces cars in the range below 25000
- Curb-size, engine size, and horsepower are positively correlated while city-mpg, highway-mpg are negatively correlated also city-mpg is negatively correlated with price as increased horsepower reduces the mileage
- More than 70 % of the vehicles have Ohc type of Engine, 57% of the cars have 4 doors, Gas is preferred by 85 % of the vehicles, Most produced vehicles are of body style sedans around 48% followed by hatchbacks 32%
- Most of the car has a Curb Weight is in range 1900 to 3100 

### Recommendation
Based on the analysis the following recommendations should be considered by the company
Vehicles within a price range lesser than 2500, OHC type engines, and gas-powered engine to maximize profits


### Refrences

1. [Kaggle](https://www.kaggle.com/datasets/kunalgupta2616/hackerearth-customer-segmentation-hackathon)
2. [Chatgpt](https://chat.openai.com/)
