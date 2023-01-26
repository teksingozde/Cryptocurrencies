# Cryptocurrencies
## Project Overview
A client requested a review of cryptocurrencies in the market. At the same time, it is necessary to classify these cryptocurrencies. It is clear that the most accurate analysis for classification is in clusters. Analyzes should be easy to evaluate and the results should be explained in an explanatory way.

### Resources:
#### Dataset: 
- crypto_data.csv

#### Software:
- Python 3.9.12
- Jupyter Notebook 6.4.8
- Pandas 1.3.5 
- scikit-learn 1.0.2
- hvplot 0.8.2
- plotly 5.12.0

## Overview of Analysis
### 1.1. Principal Component Analysis
In data science studies, it may be necessary to work with a large number of variables. This situation; Excess training time brings with it various problems such as over-learning and multicollinearity. The prepared models will need to work in optimum time and with optimum performance. In addition, multicollinearity problem in statistical algorithms such as logistic regression and linear regression can lead to skewed and misleading results.

Variable selection and size reduction methods can be used to overcome these problems. In variable selection, the variable in the data set is preserved or completely removed. In size reduction, the number of variables is reduced by creating new variables that are a combination of existing variables. Thus, all the features in the dataset are somehow still present, but the number of variables is reduced.
Principal Components Analysis; It is a multivariate statistical analysis technique that transforms a multivariate system consisting of a large number of variables that are related to each other into a system consisting of fewer and unrelated new variables as linear functions of these variables and at the same time that can explain the total change of the previous system as much as possible. . Each new variable formed as a result of the analysis is called the principal component.

In Principal Components Analysis, each of the p principal components obtained against the p number of initial variables is a linear combination of the original variables. Therefore, each basic component contains a certain amount of information from all variables. Thanks to this feature, Principal Components Analysis can provide dimension reduction by using the first m important principal components instead of a p-dimensional dataset. If the first m principal components explain most of the total variance, the remaining p-m principal components can be neglected. When compared with classical variable selection techniques, information loss will be minimized with this method.

While applying Principal Components Analysis, some issues should be taken care of. Before applying Principal Component Analysis to the data, standardization must be done. Data at different scales will cause misleading components. In addition, the analysis is heavily influenced by outliers. Before analysis, data should be separated from outlier observations or alternative methods such as Randomized PCA, Sparse PCA should be used.

The algorithm can be used on its own, or it can serve as a data cleaning or data preprocessing technique used before another machine learning algorithm.
On its own, PCA is used across a variety of use cases:
- Visualize multidimensional data. Data visualizations are a great tool for communicating multidimensional data as 2- or 3-dimensional plots.
- Compress information. Principal Component Analysis is used to compress information to store and transmit data more efficiently. For example, it can be used to compress images without losing too much quality, or in signal processing. The technique has successfully been applied across a wide range of compression problems in pattern recognition (specifically face recognition), image recognition, and more.
- Simplify complex business decisions. PCA has been employed to simplify traditionally complex business decisions. For example, traders use over 300 financial instruments to manage portfolios. The algorithm has proven successful in the risk management of interest rate derivative portfolios, lowering the number of financial instruments from more than 300 to just 3-4 principal components.
- Clarify convoluted scientific processes. The algorithm has been applied extensively  in the understanding of convoluted and multidirectional factors, which increase the probability of neural ensembles to trigger action potentials.

Principal Components Analysis;

Size reduction,
- De-correlation of data
- Visualization of high-dimensional data
- Noise filtering
It is very useful for such work.

Countless high-dimensional datasets can be used to try out PCA in practice. Among the best ones are:
- Preprocess images of x-rays and feed the data to other machine learning algorithms to predict if a patient has pneumonia.
- Cut through the noise of irrelevant features to create a better training dataset for predicting outcomes of soccer matches.
- Predict Bitcoin prices. Use the original Bitcoin dataset to compute the usual trading metrics, then apply PCA to improve your predictive algorithm’s performance. 

Principal Component Analysis results are as follows.

#### Table 1. Crypto DataFrame
<img width="500" alt="1" src="https://user-images.githubusercontent.com/26927158/214963667-b75ad20b-b0da-49a5-91ad-1c8119d5da0a.png">

Crypto DataFrame table is as above. The columns in the table are Algorithm, Proof type, Total coins mined and total coin supply. Only the first 10 rows are shown in the table. Although the coin names are not included in the table, the column on the left shows the abbreviations of the coins.

#### Table 2. Principal Components
<img width="335" alt="2" src="https://user-images.githubusercontent.com/26927158/214963775-b6eea7a9-50f7-402a-935f-2ff4cbf51faa.png">

We assigned the number of dimensions we wanted to reduce as the n_components value, then we finished reducing the size of our dataset with the fit and transform operations.
Since we were asked to create a dataframe with three principal components for the study, three principal components were created in Table 2 and the eigenvectors of these principal components were given.
As can be seen, the common variance values of the second principal component are higher than the common variance values of the third principal component, while the common variance values of the third principal component are higher than the common variance values of the first principal component.

### 1.2. K-Means Clustering Algorithm
Clustering is one of the algorithms used. The aim is to find which cluster a group of data with feature extraction belongs to according to more than one cluster feature.
The mathematical method used is to place new clusters according to the distance from the center determined point (this is also the amount of error) for each class.
The algorithm basically consists of 4 stages:

- Determination of cluster centers
- Classification of samples outside the center according to their distance
- Determination of new centers according to the classification made (or shifting of old centers to the new center)
- Repeating steps 2 and 3 until stable state.

#### 1.2.1 Clustering Methods
Depending on how clusters are created from data, there may be different clustering methods. Let's take a look at the most popular clustering techniques used heavily by institutions. These types are:
- Partitioning Methods
Partition-based clustering methods cluster the given objects on an n-dimensional plane by measuring their distance from some random or specific objects. Therefore, these methods are also known as distance-based methods.

- Hierarchical Methods
Hierarchical clustering methods are different from partitioning methods. They divide the data points into levels/hierarchies based on their similarity. These levels together form a tree-like structure (dendrogram).

- Density-Based Methods
Rather than considering the distance of the data points, in density-based clustering methods, a neighborhood is considered to form clusters. Neighborhood refers to the number of data points there that must be located in a region of interest (typically another data point) to form a cluster from the given data.

#### 1.2.2.	Where Can I Practice K-Means?
- Document Classification
Cluster documents into multiple categories based on tags, topics, and document content. This is a very standard classification problem and the k-means tool is a very suitable algorithm for this purpose.

- Identification of Crime Locations
The crime-related data available in specific areas of a city, crime category, crime area, and the relationship between the two can provide quality information on crime-prone areas in a city or region.

- Customer Segmentation
Clustering helps marketers build their customer base, work in target areas, and segment customers based on purchase history, interests, or activity tracking. Classification helps the company target specific customer sets for specific campaigns.

- Player Analysis
Analyzing player statistics has always been a critical element of the sports world, and with increasing competition, machine learning has a critical role to play here.

- Fraud Detection
Machine learning plays an important role in fraud detection and has numerous applications in auto, health and insurance fraud detection. Using historical data on fraudulent claims, it is possible to isolate new claims based on their proximity to clusters indicating fraudulent patterns.

- Call Record Detail Analysis
A call detail record (CDR) is information obtained by telecom companies during a customer's call, SMS and internet activity. This information, when used with customer demographics, provides more insight into the customer's needs. It is used to understand customer segments based on their hourly usage.

- Automatic Clustering of IT Alerts
Large enterprise IT infrastructure technology components such as network, storage or database generate large volumes of alert messages. Because warning messages indicate potentially operational issues, they must be manually scanned for prioritization for further actions. Clustering of data can provide information on alert categories and assist with mean time to repair and failure estimates.

- K-Means algorithm can be used in many applications (such as Image Recognition) that are not listed here.

K-Means Algorithm results are as follows.

#### Plot 1. Elbow Curve
<img width="800" alt="3" src="https://user-images.githubusercontent.com/26927158/214964750-a192324d-6858-4088-a835-f0f32a19d67c.png">

Here the elbow method comes in handy when we are confused about how to need sets. Our graph looks like an elbow and we have to determine this elbow point.
Here the bend point is around 4 and this is our optimal number of clusters for the above data we need to choose.
When we continue to increase the number of clusters, if we look carefully after 4, there is no big change in WSS and remains constant.

## Results

#### Table 3. Clustered DataFrame
<img width="790" alt="4" src="https://user-images.githubusercontent.com/26927158/214964998-f1ba969e-023f-433f-a42f-18582df298fd.png">

Table 3 shows the names and algorithms of the coins and the common variance values of the basic components. Only the first 10 rows are shown in this table. In general, as a result of clustering analysis, it is seen that the classification is in the 0th and 1st clusters.

#### Table 4. Hvplot Table

Hvplot offers us a high level API data exploration and visualization. Hvplot renders support the matplotlib, bokeh, and plotly libraries.

<img width="790" alt="5" src="https://user-images.githubusercontent.com/26927158/214965303-da10aadc-ef29-4982-8f78-c4b44a73475f.png">

It can be clearly seen that the hvplot table above is different from the DataFrame obtained with the pandas library. Unlike Table 3, basic components have been omitted from the table.

#### Table 5. Total Coin Supply-Total Coins Mined with Class
<img width="500" alt="6" src="https://user-images.githubusercontent.com/26927158/214965485-a9fd397f-0ab9-4b03-9696-8e2597acf126.png">

The relationship between the total mined coins and the total coin supply and the table of the classes in which the coins are included are shown above. In the table, it is seen that there is a high supply in all coins, with the exception of ETH (Ethereum) and XMR (Monero) coins.
The coin with the highest supply is the Elite Coin, which is 1337.
If we are to identify the coins with high supply in order,
EliteCoin > 404Coin > LiteCoin > Ethereum Classic > Dash > ZCash=Bitcoin > 42 Coin.
The most issued coin is EliteCoin, while 404Coin is in the second place. However, the issued coins do not meet the supplied coins.


#### Plot 2. 3D Scatter PCA Plot and Class 
<img width="800" alt="9" src="https://user-images.githubusercontent.com/26927158/214965741-ae992848-73f1-4800-b190-65c2700c6d4e.png">

As stated before, we determined that there were 4 clusters with elbow curve. Thanks to this 3D Scatter plot, we can see the areas where the clusters are concentrated on the basic components. 

When the 2nd clustering class is hovered on the graphic, the image will be as below.

<img width="815" alt="Screen Shot 2023-01-26 at 4 40 44 PM" src="https://user-images.githubusercontent.com/26927158/214967040-f1e07028-ec8b-4d42-8d8e-4ff4ffe6f35e.png">

This means that only BitTorrent coin is in the 2nd clustering class and includes the basic component values and algorithm. As can be seen, the 1st core component value of this coin is quite high.

#### Plot 3. Total Coins Mined and Total Coin Supply Class Plot
<img width="800" alt="7" src="https://user-images.githubusercontent.com/26927158/214966136-537eabd1-cc48-4cfa-8016-a1afec0fbc25.png">
In Plot 3, the color of each nesting class is shown differently. The coin belonging to the clustering class that stands out the most in the chart is the 2nd. We've shown this in the 3D scatter plot graph. The name of this coin is BitTorrent. In general, it is clear that coins are stacked in clusters 0 and 1th Coins in the 3rd stack are LitecoinCash, Poa Network and Acute Angle Cloud. In general, the coins offered in the 3rd clustering class and mined coins are equal to each other.

The information of the 2nd clustering class is shown below.

<img width="800" alt="8" src="https://user-images.githubusercontent.com/26927158/214966343-5fbab095-a0b7-4cef-810f-5a5004d3973c.png">

BitTorrent coin is mined more than offered and its clustering class is 2.













































































































