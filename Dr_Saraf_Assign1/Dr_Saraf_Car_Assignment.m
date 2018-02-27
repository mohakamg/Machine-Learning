%% ==================== Part 1: Initialization ====================

% Clear the workspace and terminal
% clc; clear;

% Load the Data Set
% load polydata.mat;
load carbig.mat;

% Split the Data
% indices = randperm(length(Acceleration), round(length(Acceleration)*0.8));
% X_train = Acceleration(indices);
% y_train = Horsepower(indices);

% Get the X and y arrays
X = Weight;
y = Horsepower;

% Create Dataset
% x = 0:0.025:1;
% noise = randn(1,length(x));
% X = x+noise;
% X = X';
% y = sin(X);

% Number of examples in the dataset
m = size(y,1);

% Clean the Data for Nans
nanVals = find(isnan(y));
y(nanVals) = 0;
y(nanVals) = mean(y);
nanVals = find(isnan(X));
X(nanVals) = 0;
X(nanVals) = mean(X);

% Feature Scaling
X_orig = X; % Make copies
y_orig = y;
% X = (X - mean(X))/(std(X));
% y = (y - mean(y))/(std(y));

% Display the data
plot(X,y,'rx','MarkerSize',10);
xlabel('Weight'); 
ylabel('Horsepower');

% Add a row of zeros to the X matrix for the intercept
X = [ones(m,1), X];
X_orig = [ones(m,1), X_orig];
fprintf('Program paused. Press enter to continue.\n');
pause;

%% ==================== Part 2: Hyperparamters Setup ====================

% Weights values for slope and intercept of line - Initial Guess (0,0)
[~, weightSize] = size(X);
weightGD = zeros(weightSize,1);

% Gradient Decent Settings
%iterations = input('Enter the number of epochs: '); % Number of Epochs
%alpha = input('Enter the alpha: '); % Learning Rate
iterations = 3e3;
alpha = 1e-7;

%% ==================== Part 3: Cost and Gradient Calculation ====================
% Testing Initial Cost at weightGD (0,0)
fprintf('\nTesting the cost function ...\n')
J = costFunction(X, y, weightGD);
fprintf('With weightGD = [0 ; 0]\nCost computed = %f\n', J);


% Testing Initial Cost at weightGD (-1,2)
fprintf('\nTesting the cost function ...\n')
J = costFunction(X, y, randn(1,weightSize)');
fprintf('With weightGD = [-1 ; 2]\nCost computed = %f\n', J);

fprintf('Program paused. Press enter to continue.\n');
pause;

fprintf('\nRunning Gradient Descent ...\n')
% run gradient descent
tic;
[weightGD, J_hist]= gradDecent(X, y, weightGD, alpha, iterations);
time_elapsed = toc;
%% ==================== Part 4: Display the Best Fit Line ====================
% print weightGD to screen
fprintf('weightGD found by gradient descent:\n');
fprintf('%f\n', weightGD);
% Plot the linear fit
hold on; % keep previous plot visible
plot(X(:,2), X*weightGD, '-')
legend('Training data', 'Linear regression')
hold off % don't overlay any more plots on this figure

%% ==================== Part 5: Closed Form Implimentation ====================
weights = pinv(X'*X)*X'*y;
fprintf('Weights found by Closed Form:\n');
fprintf('%f\n', weights);
% Plot the linear fit using closed form
hold on; % keep previous plot visible
t = X*weights;
plot(X(:,2), t);
legend('Training data', 'Linear regression', 'Closed Form')
hold off % don't overlay any more plots on this figure

%% ==================== Part 6: Predictions ====================
% testVal = input('Enter the Value for which you wanna predict the answer: ');
% prediction = ( weightGD(2)*testVal + weightGD(1) ) / (std(X_orig(:,2))*10);
% fprintf('Predicted Answer using Gradient Decent = %f\n', prediction);
% prediction = ( weights(2)*testVal + weights(1) )  / (std(X_orig(:,2))*10);
% fprintf('Predicted Answer using Closed Form = %f\n', prediction);

%% ==================== Part 7: Accuracy ====================
% pred = X*weights;
% fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
