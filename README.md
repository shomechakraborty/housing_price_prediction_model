# housing_price_prediction_model
A model for predicting home prices in low-income areas in the US.

Description: This program is a neural network model designed to predict housing prices for homes
in low-income areas in the US based on 5 pieces of information about a given house: 
its square feet, number of bedrooms, number of bathrooms, type of neighborhood, and age. 
The model is trained by a large amount of housing data about previously sold houses in the 
US consisting of their seeling price and the specific housing information used by the model. 
The model uses this data to train its parameters which it later uses to predict housing prices. 
The parameters for this model consists of weights for each input for a given house, bias, and 
a weight for the bias. A user is prompted to enter information about a given house, and the model 
calculates its prediction for the fair market price of the house. The model uses data from 50,000
homes, the first 70% of which (35,000 homes) is used for data with the later 30% (15,000 homes)
used for testing.

Functionality: This model consists of 9 layers in a 256 X 128 X 64 X 32 X 16 X 8 X 4 X 2 X 1
neuron layout. The model utilizes forward propagation for processing data through its multi-layer 
structure for both training and prediction. The model uses the Rectified Linear Unit Activation 
Function when processing weighted input values in each neuron. Gradient descent, using Calculus-based 
weight derivatives, is used to adjust the model's parameters as it processes training data.

Dataset: Housing Price Prediction Data -
https://www.kaggle.com/datasets/muhammadbinimran/housing-price-prediction-data

Model Testing Information: This model is tested on both the training and testing datasets, predicting the average price for homes in both the training and testing data. Information related for the model’s predicted average price, true average price, and the associated percent error rate of the Mean Absolute Error (MAE) relative to the true average price, for homes in both the training and testing data, are presented. The model’s training error and testing error rates are at around 3% and differ by around 0.1%, indicating the model’s strong predictive performance and ability to generalize to data beyond the training data.
