"""
author: Kerem Delikmen
date: 23/10/2020
desc: 2-Layer Neural Network
"""


import pandas as pd
import numpy as np
import model


# Read a dataset
df_x = np.load('dataset/X.npy')
df_y = np.load('dataset/Y.npy')

# Image Size
image_size = 64

# Create a Images and Labels 
X = np.concatenate((df_x[204:409], df_x[822:1027]), axis=0)                         # 3D
Y = np.concatenate((np.zeros(205), np.ones(205)), axis=0).reshape(X.shape[0], 1)    # 2D

# Train Test
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=42)

number_of_train = X_train.shape[0]
number_of_test = X_test.shape[0]

# 3D convert to 2D
X_train_flatten = X_train.reshape(number_of_train, X_train.shape[1]*X_train.shape[1])
X_test_flatten = X_test.reshape(number_of_test, X_test.shape[1]*X_test.shape[1])

# Transpose
x_train = X_train_flatten.T
x_test = X_test_flatten.T
y_train = Y_train.T
y_test = Y_test.T

model.two_neural_network(x_train, y_train, x_test, y_test, 2500)