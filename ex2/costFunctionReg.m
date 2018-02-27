function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

%% COST
normalCost = 0;

g = X*theta;
h = sigmoid(g);

for i=1:m
   newCost = ( -1*y(i)*log(h(i)) - (1-y(i))*log(1-h(i)) ) ;
   normalCost = normalCost+newCost;
end
normalCost = normalCost/m;

% regularization = 0;
% for i=1:length(theta)
%     regularization = regularization+(theta(i)^2);
% end
regularization = (lambda*(sum(theta.^2) - theta(1)^2))/(2*m);

J = normalCost+regularization;

%% Gradient
for j=1:length(theta)
    for i=1:m
       diffCost = (h(i) - y(i)) * X(i,j);
       grad(j) = grad(j)+diffCost; 
    end
    grad(j) = grad(j)/m;
    if(j>1)
        grad(j) = grad(j)+lambda*theta(j)/m;
    end
end
% =============================================================

end
