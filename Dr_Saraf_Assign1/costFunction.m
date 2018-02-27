function J = costFunction(X,y,weight)

    % Number of training datapoints
    m = length(y);
    
    % Initial Cost = 0
    J = 0;

    % Our hypothesis is a linear line
    hypothesis = X*weight;
    % Compute the Cost at this weight
    J = (1/(2*m))*sum((hypothesis - y).^2);
  
end