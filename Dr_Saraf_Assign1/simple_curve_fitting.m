clc; clear;

x = linspace(0,1,10);
%noise = randn(1,length(x));
noise = wgn(1,length(x),1);
X_train = x';
y_train = sin(2*pi*X_train)+noise';

x2 = linspace(0,1,10);
noise = wgn(1,length(x),1);
X_test = x2';
y_test = sin(2*pi*X_test)+noise';

m = length(X_train);
X_train_orig = [ones(m,1) X_train];
X_test_orig = [ones(m,1) X_test];

max_degree = 9;
% tolerence = 0.12;

error_diff = zeros(1,max_degree);
J_train = zeros(max_degree,1);
J_test = zeros(max_degree,1);
for j = 1:max_degree

    for i = 1:j+1
       if i==1
          X_train = X_train_orig; 
          X_test = X_test_orig; 
       else
           X_train = [X_train X_train(:,2).^i];
           X_test = [X_test X_train(:,2).^i];
       end
    end

    lambda = 0;
    w = pinv( X_train'*X_train + lambda*eye(size(X_train'*X_train)) )*X_train'*y_train;

    figure;
    ylim([-7.5,7.5]);
    hold on;
    plot(x,sin(x),'b--');
    plot(X_train(:,2),y_train,'rx');
    plot(X_test(:,2),y_test,'yo');


    syms a b;
    f(a,b) = w(1) + w(2)*a;
    disp(['Degree:', num2str(j)]);
    if(j>=2)
        for i=2:j+1
            f(a,b) = f(a,b) + w(i+1)*a^(i);
        end
    end
    ezplot(b == f(a,b),[-7.5,7.5]);


    J_train(i) = (1/(2*m))*sum((y_train - X_train*w).^2);
    J_test(i) = (1/(2*m))*sum((y_test - X_test*w).^2);

    disp(['Training Error: ', num2str(J_train(i))]);
    disp(['Test Error: ', num2str(J_test(i))]);
    err_diff_for_this_complexity = abs(J_train(i)-J_test(i));
    disp(['Error Diff: ', num2str(err_diff_for_this_complexity)]);
    
    error_diff(j) = abs(J_train(i)-J_test(i));
    
    
end
[min_err_diff, bestComplexity] = min(error_diff);
disp(['Best Complexity without Regularization: ', num2str(bestComplexity)]);

