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
### Principal Component Analysis
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

## Results

