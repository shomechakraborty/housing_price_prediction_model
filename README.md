# housing_price_prediction_model
A model for predicting home prices in low-income areas in the US.

Detailed Description: This program is a neural network model designed to predict housing prices for homes
in low-income areas in the US based on 5 pieces of information about a given house: 
its square feet, number of bedrooms, number of bathrooms, type of neighborhood, and age. 
This neural network consists of 9 layers in a 256 X 128 X 64 X 32 X 16 X 8 X 4 X 2 X 1 neuron 
layout.The model is trained by a large amount of housing data about previously sold houses in the 
US consisting of their seeling price and the specific housing information used by the model. 
The model uses this data to train its parameters which it later uses to predict housing prices. 
The parameters for this model consists of weights for each input for a given house, bias, and 
a weight for the bias. A user is prompted to enter information about a given house, and the model 
calculates its prediction for the fair market price of the house. The model uses data from 50,000
homes, the first 70% of which (35,000 homes) is used for data with the later 30% (15,000 homes)
used for testing.
