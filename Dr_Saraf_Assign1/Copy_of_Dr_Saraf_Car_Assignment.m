function time_elapsed = Copy_of_Dr_Saraf_Car_Assignment(totalsulfurdioxide, freesulfurdioxide)

    %% ==================== Part 1: Initialization ====================

    % Clear the workspace and terminal
    % clc; clear;

    % Load the Data Set
    % load polydata.mat;
    % load carbig.mat;

    % Split the Data
    % indices = randperm(length(Acceleration), round(length(Acceleration)*0.8));
    % X_train = Acceleration(indices);
    % y_train = Horsepower(indices);

    % Get the X and y arrays
    X = totalsulfurdioxide;
    y = freesulfurdioxide;

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
    X = (X - mean(X))/(std(X));
    y = (y - mean(y))/(std(y));

    % Display the data
    plot(X,y,'rx','MarkerSize',10);
    xlabel('Total Sulfur Dioxide'); 
    ylabel('Free Sulfur Dioxide');

    % Add a row of zeros to the X matrix for the intercept
    X = [ones(m,1), X];
    X_orig = [ones(m,1), X_orig];
    %fprintf('Program paused. Press enter to continue.\n');
    %pause;

    %% ==================== Part 2: Hyperparamters Setup ====================

    % Weights values for slope and intercept of line - Initial Guess (0,0)
    [~, weightSize] = size(X);
    weightGD = zeros(weightSize,1);

    % Gradient Decent Settings
    %iterations = input('Enter the number of epochs: '); % Number of Epochs
    %alpha = input('Enter the alpha: '); % Learning Rate
    iterations = 3e6;
    alpha = 1e-5;

    %% ==================== Part 3: Cost and Gradient Calculation ====================
    % Testing Initial Cost at weightGD (0,0)
    %fprintf('\nTesting the cost function ...\n')
    J = costFunction(X, y, weightGD);
    %fprintf('With weightGD = [0 ; 0]\nCost computed = %f\n', J);


    % Testing Initial Cost at weightGD (-1,2)
    %fprintf('\nTesting the cost function ...\n')
    J = costFunction(X, y, randn(1,weightSize)');
    %fprintf('With weightGD = [-1 ; 2]\nCost computed = %f\n', J);

    %fprintf('Program paused. Press enter to continue.\n');
    %pause;

    %fprintf('\nRunning Gradient Descent ...\n')
    % run gradient descent
    tic
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
    plot(X(:,2), t, 'p-');
    legend('Training data', 'Linear regression', 'Closed Form')
    hold off % don't overlay any more plots on this figure

    
end