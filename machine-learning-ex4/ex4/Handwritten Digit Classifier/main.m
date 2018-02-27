%% Program Initialization and Data Load
% clear the console and workspace
clc; clear;
% Load the Images
load ex4data1.mat;

%% The Neural Net
input_layer_size  = 400;  % 20x20 Input Images of Digits
num_labels = 10;          % 10 labels, from 1 to 10   

hidden_layer_size = 25;   % 25 hidden units

%% Set up Initial Paramters
load ex4weights.mat; % Load the weights for testing
nn_weights = [Theta1 Theta2]; % Theta1 are the weights between Layer 1 and Layer 2
                              % Theta 2 are the weights between Layer 2 and
                              % 3

%% Forward Propogation, Calculate Cost and Apply Backpropgation
[J, gradient] = propogate(X, y, nn_weights, input_layer_size, ...
                        hidden_layer_size, ...
                        num_labels);




