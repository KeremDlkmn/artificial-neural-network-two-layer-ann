"""
author: Kerem Delikmen
date: 23/10/2020
desc: 2-Layer Neural Network
"""


import numpy as np


## Intialize Weight and Bias ##
def init_parameters_and_layer_size_NN(x_train, y_train):
    parameters = {
                    'weight_1' : np.random.randn(3, x_train.shape[0]) * 0.1,
                    'bias_1'   : np.zeros((3, 1)),
                    'weight_2' : np.random.randn(y_train.shape[0], 3) * 0.1,
                    'bias_2'   : np.zeros((y_train.shape[0], 1)) }
    return parameters


## Sigmoid Function ##
def sigmoid(z):
    y_head = 1/(1+np.exp(-z))
    return y_head


## Forward Propagation ##
def forward_propagation(x_train, parameters):
    # Input - Hidden Layer
    z_1 = np.dot(parameters['weight_1'], x_train) + parameters['bias_1']
    A_1 = np.tanh(z_1)

    # Hidden - Output Layer
    z_2 = np.dot(parameters['weight_2'], A_1) + parameters['bias_2']
    A_2 = sigmoid(z_2)

    # Store z and A values
    cache = {
                'z_1' : z_1,
                'A_1' : A_1,
                'z_2' : z_2,
                'A_2' : A_2 }
    
    return A_2, cache


## Loss And Cost Function ##
def compute_cost_NN(A_2, Y, parameters):
    # Loss Value
    loss = np.multiply(np.log(A_2), Y)
    # Cost Value
    cost = -np.sum(loss)/Y.shape[1]

    return cost


## Backward Propagation ##
def backward_propagation(parameters, cache, X, Y):
    
    Dz_2 = cache['A_2'] - Y
    Dw_2 = np.dot(Dz_2, cache['A_1'].T) / X.shape[1]
    Db_2 = np.sum(Dz_2, axis=1, keepdims=True) / X.shape[1]

    Dz_1 = np.dot(parameters['weight_2'].T, Dz_2) * (1 - np.power(cache['A_1'], 2))
    Dw_1 = np.dot(Dz_1, X.T) / X.shape[1]
    Db_1 = np.sum(Dz_1, axis=1, keepdims=True) / X.shape[1]

    grads = {
                'Dweight_1' : Dw_1,
                'Dbias_1'   : Db_1,
                'Dweight_2' : Dw_2,
                'Dbias_2'   : Db_2 }
    
    return grads


## Update Parameters ##
def update_parameters_NN(parameters, grads, learning_rate = 0.01):
    parameters = {
        'weight_1' : parameters['weight_1'] - learning_rate*grads['Dweight_1'],
        'bias_1'   : parameters['bias_1'] - learning_rate*grads['Dbias_1'],
        'weight_2' : parameters['weight_2'] - learning_rate*grads['Dweight_2'],
        'bias_2'   : parameters['bias_2'] - learning_rate*grads['Dbias_2'] }
    
    return parameters


## Prediction ##
def prediction_NN(parameters, x_test):
    # Forward Propagation
    A_2, cache = forward_propagation(x_test, parameters)

    # Prediction Matrix
    y_prediction = np.zeros((1, x_test.shape[1]))

    for i in range(A_2.shape[1]):
        if A_2[0, i] <= 0.5:
            y_prediction[0, i] = 0
        else:
            y_prediction[0, i] = 1

    return y_prediction


### LOGISTIC REGRESSION ###
def two_neural_network(x_train, y_train, x_test, y_test, number_of_iteration):
    cost_list = []
    index_list = []

    # initialize
    parameters = init_parameters_and_layer_size_NN(x_train, y_train)

    for i in range(0, number_of_iteration):

        # forward propagation
        A_2, cache = forward_propagation(x_train, parameters)

        # compute cost
        cost = compute_cost_NN(A_2, y_train, parameters)
        
        # backward propagation
        grads = backward_propagation(parameters, cache, x_train, y_train)

        # update parameters model fitting
        parameters = update_parameters_NN(parameters, grads)

        if i % 100:
            cost_list.append(cost)
            index_list.append(i)
            print(f'Cost after iteration: {i}. {cost}')
    
    # predict
    y_pred_train = prediction_NN(parameters, x_train)
    y_pred_test  = prediction_NN(parameters, x_test)

    print(f'Train Accuaracy: {(100 - np.mean(np.abs(y_pred_train - y_train) * 100 ))}')
    print(f'Test  Accuaracy: {(100 - np.mean(np.abs(y_pred_test - y_test) * 100 ))}')
