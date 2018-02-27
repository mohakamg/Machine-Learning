function p = predict(Theta1, Theta2, Theta3, X)
    m = size(X, 1);
    num_labels = size(Theta3, 1);

    p = zeros(m, 1);

    h1 = sigmoid(X * Theta1);
    h2 = sigmoid(h1 * Theta2);
    h3 = sigmoid(h2 * Theta3);
    [~, p] = max(h3, [], 2);



end
