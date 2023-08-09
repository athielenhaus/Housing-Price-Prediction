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

The following steps were undertaken to find the model:
- Exploratory Data Analysis (EDA)
- Creation of a baseline model
- Feature Engineering and Selection
- Gridsearch

Below we provide some additional information about the individual steps.

##### Exploratory Data Analysis
After importing the data, EDA was conducted using EDA libraries including Sweetviz and Dataprep as well as by plotting data with Matplotlib and on a map in Tableau.
The dataset is composed of 21 columns. Two columns, 'id' and 'date', were quickly dropped as their lack of relevance to estimating the price was clear.
The other learnings from EDA included:
- no missing data.
-the dataset includes statistical as well as geographic outliers.
![alt text](Screenshot%20Tableau%20Map.jpg)






<img src="Screenshot%20Tableau%20Map.jpg" alt="drawing" width="200"/>

##### Creation of a baseline model
##### Feature Engineering and Selection
    - Outlier removal
    - Encoding / transformation of geospatial information (zip codes and latitude and longitude)
    - Feature removal
    - Binning
    - 
