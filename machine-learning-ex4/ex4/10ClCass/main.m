%% Neural Network for 4 Layered [Input Layer, 2 Hidden Layer, 1 Output Layer] 10 - Class Classification
clc; clear; close all;

% Load Data
load ex4data1.mat;

%% Setup Neural Net
input_neurons = size(X,2);
num_hidden_layers = 1;
hidden_layer_neurons_1 = 200;
hidden_layer_neurons_2 = 50;
output_layer = 10;

%% Display a random Image
dispRandImg(X);

%% Add Bias
X_with_bias = [ones(length(X),1) X];
input_neurons = input_neurons+1;
hidden_layer_neurons_1 = hidden_layer_neurons_1 + 1;
hidden_layer_neurons_2 = hidden_layer_neurons_2 + 1;

%% Set up Weights
epsilon_init = 0.12;

Weights_1 = rand([input_neurons, hidden_layer_neurons_1]) * 2 * epsilon_init - epsilon_init;
Weights_2 = rand([hidden_layer_neurons_1, hidden_layer_neurons_1]) * 2 * epsilon_init - epsilon_init;
Weights_3 = rand([hidden_layer_neurons_1, output_layer]) * 2 * epsilon_init - epsilon_init;
% Weights_1 = randn([input_neurons, hidden_layer_neurons_1]);
% Weights_2 = randn([hidden_layer_neurons_1, output_layer]);

% Weights_3 = rand([hidden_layer_neurons_2,output_layer]);

%% FeedForward and BackPropogation
fprintf('\nFeedforward Using Neural Network ...\n');

% This cost function does not include Regularization 
[J grad1 grad2 grad3] = nnCostFunction(Weights_1, Weights_2, Weights_3, ...
                   output_layer, X_with_bias, y);
               
%% Train The Neural Net

fprintf('\nTraining Neural Network... \n')
learning_rate = 0.0001;
num_of_epochs = 10000;
tic
for i = 1:num_of_epochs
    [J grad1 grad2 grad3] = nnCostFunction(Weights_1, Weights_2, Weights_3, ...
                            output_layer, X_with_bias, y);
    disp(['Cost: ', num2str(J)]);
    Weights_1 = Weights_1 - learning_rate*grad1;
    Weights_2 = Weights_2 - learning_rate*grad2;
    Weights_3 = Weights_3 - learning_rate*grad3;
end
toc
%% Accuracy Test
pred = predict(Weights_1, Weights_2, Weights_3, X_with_bias);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);