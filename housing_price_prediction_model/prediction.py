import math, random, csv

from datetime import datetime

"""This program is a neural network model designed to predict housing prices for homes
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
NOTE: It may take some time for the model to be trained when ran due to its size.
Dataset Link: https://www.kaggle.com/datasets/muhammadbinimran/housing-price-prediction-data"""

def main():
    """This function gathers trained parameters for the model, determines the accuracy
    of the model using the testing data file, and uses the model to calculate the fair market
    price of a house after prompting users for information related to the house."""
    
    """Assembles training and testing data"""
    training_data_file = "housing_price_training_data.csv"
    testing_data_file = "housing_price_testing_data.csv"
    housing_price_training_data = assemble_housing_market_data_set(training_data_file)
    housing_price_testing_data = assemble_housing_market_data_set(testing_data_file)
    
    """Initializes model parameters"""
    parameters = initialize_parameters(housing_price_training_data)
    
    """Trains model"""
    training_outputs = train_layers(housing_price_training_data, parameters)
    
    """"Model Training Error:"""
    true_training_total_price = 0.0
    true_training_average_price = 0.0
    predicted_training_total_price = 0.0
    predicted_training_average_price = 0.0
    training_percent_error = 0.0
    for i in range(len(housing_price_training_data)):
        true_training_total_price += housing_price_training_data[i][-1]
        predicted_training_total_price += run_layers(housing_price_training_data[i][0:len(housing_price_training_data[i]) - 1], parameters)
    true_training_average_price = round(true_training_total_price / len(housing_price_training_data), 2)
    predicted_training_average_price = round(predicted_training_total_price / len(housing_price_training_data), 2)
    training_percent_error = round(abs(((true_training_average_price - predicted_training_average_price) / true_training_average_price) * 100), 2)
    
    """"Model Testing Error:"""
    true_testing_total_price = 0.0
    true_testing_average_price = 0.0
    predicted_testing_total_price = 0.0
    predicted_testing_average_price = 0.0
    testing_percent_error = 0.0
    for i in range(len(housing_price_testing_data)):
        true_testing_total_price += housing_price_testing_data[i][-1]
        predicted_testing_total_price += run_layers(housing_price_testing_data[i][0:len(housing_price_testing_data[i]) - 1], parameters)
    true_testing_average_price = round(true_testing_total_price / len(housing_price_testing_data), 2)
    predicted_testing_average_price = round(predicted_testing_total_price / len(housing_price_testing_data), 2)
    testing_percent_error = round(abs(((true_testing_average_price - predicted_testing_average_price) / true_testing_average_price) * 100), 2) 
    
    """Error Summary Functionality"""
    print("***** Model Performance Summary *****")
    print("The true average price of homes in the training data is $" + str(true_training_average_price) + ".")
    print("The predicted average price of homes in the training data is $" + str(predicted_training_average_price) + ".")
    print("Training Percent Error: " + str(training_percent_error) + "%.")
    print("The true average price of homes in the testing data is $" + str(true_testing_average_price) + ".")
    print("The predicted average price of homes in the testing data is $" + str(predicted_testing_average_price) + ".")
    print("Testing Percent Error: " + str(testing_percent_error) + "%.")
    
    """User Prompt Functionality:"""
    choice = int(input("This model is designed to help predict housing prices. This can be useful for if you are a prospective seller or buyer of a house and want to come up with a fair market price for it. Enter '1' to continue or '0' to quit. "))
    while choice != 0 and choice != 1:
        choice = int(input("Please enter '1' to continue or '0' to quit. "))
    while choice == 1:
        print("In order to determine a fair market price, there is some information the model need from you")
        squarefeet = input("How many squarefeet large is the house? Enter a number: ")
        while not squarefeet.isdigit():
            squarefeet = input("Please enter a number: ")
        squarefeet = float(squarefeet)
        bed_count = input("How many bedrooms does the house consist of? Enter a number: ")
        while not bed_count.isdigit():
            bed_count = input("Please enter a number: ")
        bed_count = float(bed_count)
        bath_count = input("How many bathrooms does the house consist of? Enter a number: ")
        while not bath_count.isdigit():
            bath_count = input("Please enter a number: ")
        bath_count = float(bath_count)
        neighborhood = input("Is this house located in a rural, suburban, or urban neighborhood? Enter 'Rural', 'Surburan', or 'Urban': ")
        while neighborhood != "Rural" and neighborhood != "Suburban" and neighborhood != "Urban":
            neighborhood = input("Please enter 'Rural', 'Surburan', or 'Urban': ")
        if neighborhood == "Rural":
            neighborhood_score = 1.0
        elif neighborhood == "Suburban":
            neighborhood_score = 2.0
        elif neighborhood == "Urban":
            neighborhood_score = 3.0
        age = input("How many years old is the house? Enter a number: ")
        while not age.isdigit():
            age = input("Please enter a number: ")
        age = float(age)
        inputs = [squarefeet, bed_count, bath_count, neighborhood_score, age]
        prediction = round(run_layers(inputs, parameters), 2)
        print("The model predicts that the fair market price for this house is around $" + str(prediction) + ".") 
        choice = int(input("Please enter '1' to continue or '0' to quit. "))
        while choice != 0 and choice != 1:
            choice = int(input("Please enter '1' to continue or '0' to quit. "))

def assemble_housing_market_data_set(file):
    """This function assembles the housing market training data used by the model to adjust its 
    parameters while it is being trained."""
    housing_market_data = []
    file = open(file, "r")
    csv_reader = csv.reader(file)
    for line in csv_reader:
        square_feet = float(line[0])
        bed_count = float(line[1])
        bath_count = float(line[2])
        if line[3] == "Rural":
            neighborhood_score = 1.0
        elif line[3] == "Suburb":
            neighborhood_score = 2.0
        elif line[3] == "Urban":
            neighborhood_score = 3.0
        age = float(2023.0 - float(line[4]))
        target_price = float(line[5])
        house_data = [square_feet, bed_count, bath_count, neighborhood_score, age, target_price]
        housing_market_data.append(house_data)
    return housing_market_data

def initialize_parameters(housing_market_training_data):
    """This function initializes the parameters of the model at random numbers between 0 and 0.1
    which become adjusted as the model is trained."""
    parameters = []
    for i in range(9):
        parameters.append([])
        for j in range(int(math.pow(2, 8 - i))):
            parameters[i].append([])
            for k in range(3):
                parameters[i][j].append([])
            if i == 0:
                for k in range(len(housing_market_training_data[0]) - 1):
                    parameters[i][j][0].append(random.uniform(0, 0.1))
            else:
                for k in range(int(math.pow(2, 9 - i))):
                    parameters[i][j][0].append(random.uniform(0, 0.1))
            parameters[i][j][1].append(random.uniform(0, 0.1))
            parameters[i][j][2].append(random.uniform(0, 0.1))
    return parameters

def run_layers(inputs, parameters):
    """This function runs the layers of the neural network in order to calcluate a prediction
    for the fair market value of a house using the given inputs and neural network parameters.
    The indidvidual predictions generated by each neuron in each layer serves as the inputs for
    the next layer of the neural network."""
    outputs = []
    for i in range(9):
        outputs.append([])
        for j in range(int(math.pow(2, 8 - i))):
            if i == 0:
                outputs[i].append(run_neuron(inputs, parameters[i][j][0], parameters[i][j][1], parameters[i][j][2]))
            else:
                outputs[i].append(run_neuron(outputs[i - 1], parameters[i][j][0], parameters[i][j][1], parameters[i][j][2]))    
    return outputs[8][0]

def run_neuron(inputs, input_weights, bias, bias_weight):
    """This functions runs each of the individual neurons of the neural network 
    which caculates individual predictions for a house's fair market value."""
    sum = compute_sum(inputs, input_weights, bias, bias_weight)
    output = compute_ReLU_activation(sum)
    return output 

def train_layers(housing_price_training_data, parameters):
    """This function uses the housing market training data to train parameters for each layer
    of the neural network. It comes with a returned output list consisting of all the housing
    price values it predicted and adjusted its weights based on from the training data."""
    outputs = []
    for i in range(len(housing_price_training_data)):
        initial_inputs = housing_price_training_data[i][0:len(housing_price_training_data[i]) - 1]
        target = housing_price_training_data[i][-1]
        outputs.append([])
        for j in range(9):
            outputs[i].append([])
            for k in range(int(math.pow(2, 8 - j))):
                if j == 0:
                    outputs[i][j].append(train_neuron(initial_inputs, parameters[j][k][0], parameters[j][k][1], parameters[j][k][2], target))
                else:
                    outputs[i][j].append(train_neuron(outputs[i][j - 1], parameters[j][k][0], parameters[j][k][1], parameters[j][k][2], target))         
    return outputs

def train_neuron(inputs, input_weights, bias, bias_weight, target):
    """This function trains the parameters for each individual neuron for each layer of the model
    using inputs and targets from the training data. Each neuron in this function calculates a 
    housing price prediction using inputs from the training data parameters and adjusts the model 
    parameters for each individual neuron if the prediction it generates does not equal the target 
    under the training data."""
    sum = compute_sum(inputs, input_weights, bias, bias_weight)
    output = compute_ReLU_activation(sum)
    if output != target:
        adjust_parameters(inputs, input_weights, bias, bias_weight, output, target)
    return output 
    
def compute_sum(inputs, input_weights, bias, bias_weight):
    """This function calculates a sum value using housing price inputs and parameters 
    for each individual neuron in calculating a housing price prediction"""
    sum = 0
    for i in range(len(inputs)):
        sum += (inputs[i] * input_weights[i])
    sum += (bias[0] * bias_weight[0])
    return sum

def compute_ReLU_activation(sum):
    """This function represents the ReLU activation function and processes the sum values
    generated from housing price inputs and parameters used for each individual neuron 
    in the model in order to generate a final housing price prediction."""
    output = 0
    if sum > 0:
        output = sum
    else:
        output = 0
    return output

def adjust_parameters(inputs, input_weights, bias, bias_weight, output, target):
    """This function adjusts the parameters of the model while using gradient descent with a learning
    rate of _X_ for each individual neuron in the model when the neuron calculates a housing price 
    prediction using inputs from the training data and parameters not equal to the target price under 
    the training data while the model is being trained."""
    loss = target - output
    learning_rate = 0.0000000000000008
    cost = math.pow(loss, 2)
    cost_derivative_with_respect_to_loss = 2 * loss
    loss_derivative_with_respect_to_output = int(-1)
    if output > 0:
        output_ReLU_activation_derivative_with_respect_to_sum = int(1)
    else:
        output_ReLU_activation_derivative_with_respect_to_sum = int(0)
    for i in range(len(input_weights)):
        sum_derivative_with_respect_to_weight = inputs[i]
        cost_derivative_with_respect_to_weight = learning_rate * cost_derivative_with_respect_to_loss * loss_derivative_with_respect_to_output * output_ReLU_activation_derivative_with_respect_to_sum * sum_derivative_with_respect_to_weight
        input_weights[i] -= cost_derivative_with_respect_to_weight
    sum_derivative_with_respect_to_bias = int(1)
    cost_derivative_with_respect_to_bias = learning_rate * cost_derivative_with_respect_to_loss * loss_derivative_with_respect_to_output * output_ReLU_activation_derivative_with_respect_to_sum * sum_derivative_with_respect_to_bias
    bias[0] -= cost_derivative_with_respect_to_bias
    sum_derivative_with_respect_to_bias_weight = bias[0]
    cost_derivative_with_respect_to_bias_weight = learning_rate * cost_derivative_with_respect_to_loss * loss_derivative_with_respect_to_output * output_ReLU_activation_derivative_with_respect_to_sum * sum_derivative_with_respect_to_bias_weight
    bias_weight[0] -= cost_derivative_with_respect_to_bias_weight
    
main()
        
        
