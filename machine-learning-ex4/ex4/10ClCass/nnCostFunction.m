function [J grad1 grad2 grad3] = nnCostFunction(Weights_1, ...
                    Weights_2, Weights_3, ...
                    output_layer, X, y)
               
    % Set initial cost to zero           
    J = 0;
    
    num_of_ex = size(X,1);
    
    % Feed Forward
    a1 = X;
        % Second Layer
    z2 = a1*Weights_1;
    a2 = sigmoid(z2);
        % Third Layer
    z3 = a2*Weights_2;
    a3 = sigmoid(z3);
        % Output Layer
    z4 = a3*Weights_3;
    a4 = sigmoid(z4);
    
    % Calculate the Cross Entropy Log Loss Function
    for k = 1:output_layer
       y_k = y == k;
       a4_k = a4(:,k);
       %Cross Entropy
       J_k = (1/num_of_ex) * sum( -y_k.*log(a4_k) - (1-y_k).*log(1-a4_k) );
       % Least Squares
%        J_k = 0.5*(1/num_of_ex) * sum((y_k - a4_k).^2);
       J = J + J_k;
    end
    
    delta_4 = zeros(5000,10);
    % Compute the Delta and the gradient
   for k = 1:output_layer
      y_k = y == k;
      a4_k = a4(:,k);
      delta_4(:,k) = a4_k - y_k; 
   end
   delta_3 = delta_4*(Weights_3)' .* sigmoidGradient(z3);
   delta_2 = delta_3*(Weights_2)' .* sigmoidGradient(z2);
   
   grad3 = (a3)'*delta_4;
   grad2 = (a2)'*delta_3;
   grad1 = (a1)'*delta_2;
end