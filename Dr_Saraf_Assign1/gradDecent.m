function [weight, J_history] = gradDecent(X, y, weight, alpha, num_iters)
    % Initializtion Values
    [m,n] = size(X);
    J_history = zeros(num_iters, 1);
    
    dJ = zeros(n,1);

    
    % Perform Gradient Decent for the specified Number of Iterations
    for iter=1:num_iters
        hypothesis = X*weight;
        % Initialize gradient Vector of Cost

        for i=1:n
            dJ(i) = sum((hypothesis - y).*X(:,i));
        end
        % Get the new weight's
        weight = weight - (alpha/m)*dJ;
    end
    
    % Save the cost J in every iteration  
    J = costFunction(X, y, weight);
    fprintf('Cost Now = %f\n', J);
    J_history(iter) = J;
   
end