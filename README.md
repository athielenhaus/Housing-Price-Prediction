# Housing Price Prediction

## Introduction
In this project, we look at housing price data from approx. 21,400 properties from King County in the state of Washington, USA. The data describes properties which were sold in a one-month period in 2015. The dataset is from [Kaggle](https://www.kaggle.com/code/madislemsalu/predicting-housing-prices-in-king-county-usa).

#### Defining the problem
In this imaginary scenario, the client is a real estate agency in the Seattle area. The agency wishes to predict housing prices with a high level of accuracy. The agency is convinced that there is a linear relationship between the features in the dataset and the housing prices, therefore we will only concentrate on finding a Linear Model. 

#### Tools
- The primary libraries used include:
    - for Exploratory Data Analysis (EDA): Fast ML, Sweetviz, Dataprep and Matplotlib
    - for ML: Scikit-Learn, Fast ML
    
#### Steps

The following steps were undertaken:

##### Exploratory Data Analysis
After importing the data, EDA was conducted using EDA libraries including Sweetviz and Dataprep as well as by plotting data with Matplotlib and on a map in Tableau.
This revealed statistical as well as geographic outliers.
![alt text](Screenshot%20Tableau%20Map.jpg)

##### Creation of a baseline model
##### Feature Engineering and Selection
    - Outlier removal
    - Encoding / transformation of geospatial information (zip codes and latitude and longitude)
    - Feature removal
    - Binning
    - 
