# Housing Price Prediction
<img src=https://www.gpsmycity.com/img/ctw/165.jpg />  


## Introduction
In this project, we look at housing price data from approx. 21,400 properties from King County in the state of Washington, USA. The data describes properties which were sold in a one-month period in 2015. The dataset is from [Kaggle](https://www.kaggle.com/code/madislemsalu/predicting-housing-prices-in-king-county-usa).

#### Defining the problem
In this imaginary scenario, the client is a real estate agency in the Seattle area. The agency wishes to predict housing prices with a high level of accuracy. The agency is convinced that there is a linear relationship between the features in the dataset and the housing prices, therefore we will only concentrate on finding a Linear Model. 

#### Tools
The primary libraries used include:
- for Exploratory Data Analysis (EDA): Fast ML, Sweetviz, Dataprep and Matplotlib
- for ML: Scikit-Learn, Fast ML
    
#### Steps

The following steps were undertaken to find the model:
- Exploratory Data Analysis (EDA)
- Creation of a baseline model
- Feature Engineering and Selection
- Gridsearch

Below, we provide some additional information about the individual steps.

#### Exploratory Data Analysis
After importing the data, EDA was conducted using EDA libraries including Sweetviz and Dataprep as well as by plotting data with Matplotlib and on a map in Tableau.
The dataset is composed of 21 columns. Two columns, 'id' and 'date', were quickly dropped as their lack of relevance to estimating the price was clear.
Some other learnings from EDA included:
- no missing data.
- the column 'yr_renovated' either contains zeros or the year in which the house was renovated and would likely benefit from binning
- the dataset includes statistical as well as geographic outliers.
<img src="Screenshot%20Tableau%20Map.jpg" alt="drawing" width="600"/>

- some features such as 'grade' and 'view' are ordinal values - they can thus be treated as either numerical or categorical
- some features such as 'sqr. ft living' and 'sqr. ft above' are highly correlated with each other, it is likely one can be dropped. 
- the various square footage features and 'grade' appear to be most highly correlated with price
<img src="sc_correlation_matrix.jpg" alt="drawing" width="600"/>

To obtain some industry knowledge, a brief internet research was conducted which revealed that the following can impact real estate price:  
- Prices of comparable properties
- Age and condition
- Property size / usable space
- Neighborhood / location (crime rate, schools, view, parking availability...)
- Upgrades


### Creation of a baseline model
A baseline vanilla Linear Regression model was created with Scikit Learn. In addition, some data preparation and testing functions were created for facilitating testing, including for the purposes of cross validation and for testing on validation and test sets.
The initial tests on the raw data (minus the 'id' and 'date' columns indicated an R2 of approx. 0.7 with an MAE of over $120,000 

### Feature Engineering and Selection
The Feature Engineering and Selection steps included:
    - Outlier removal
    - Encoding / transformation of geospatial information (zip codes and latitude and longitude)
    - Feature removal
    - Binning

__Outlier Removal__
Since the model should be able to forecast the price for a variety of properties but at the same time maintain good accuracy, it was deemed a good idea to remove at least some outliers. Since there were no outliers towards the bottom, the question was how to set the caps. 3 methods were used for determining caps, including:
- 3 standard deviations above the mean
- 95th percentile
- 99th percentile
Using the 99th percentile as a cap yielded the smallest number of outliers, but nonetheless yielded an immediate improvement in MAE of around 10%.

In addition, some geographic outliers were located using the Tableau map. It was decided to drop properties East of -121.7 degrees longitude, as these properties were located in geographically remote areas and in very different circumstances as the main data set, with a median value below the dataset median. It was also considered whether or not to remove approx. 115 properties located on two islands (Vashon and Maury Island) only accessible by ferry. However, since the median value of these properties was slightly above the dataset median, they would presumably of interest to our client, so it was decided to keep them in the dataset. 
<img src="geo_outliers.jpg" width="600"/>

### Sources:
- Insights regarding housing prices: https://www.opendoor.com/articles/factors-that-influence-home-value
- more insights regarding housing prices: https://www.experian.com/blogs/ask-experian/factors-that-affect-home-value/
- photo of houses: https://www.gpsmycity.com/discovery/queen-anne-sightseeing-walk-165.html    
