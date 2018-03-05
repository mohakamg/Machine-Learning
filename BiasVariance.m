clc; clear;

L = 100; % Number of Da
Dx = [];
Dy = [];
phi_all =[];
N = 25;


s = 0.1;
% ran_lmbd = (rand(3,1)-1)*5;
ran_lmbd = linspace(-5,5,20);
lambda = exp(ran_lmbd);

figure;
hold on;
    
for j = 1:L

    X  = rand(N,1);
    X = sort(X);
    E = randn(N,1)*0.3;
    t = sin(2*pi*X) + E;

    Dx = [Dx;X'];
    Dy = [Dy;t'];
%     scatter(X,t);

    phi = designMatrix(X,s);
    phi_all(:,:,j) = phi;

    for k = 1:3
        subplot(3,2,k*2-1);
        ylim([-2,2]);
        title(['ln lambda = ',num2str(ran_lmbd(k))]);
            
        hold on;
        w = pinv( phi_all(:,:,j)'*phi_all(:,:,j) + lambda(k)*eye( size(phi,2)) )*phi_all(:,:,j)'*t;
        x = linspace(0,1,N)';

        if rem(j,5) == 0
            plot(x,phi_all(:,:,j)*w,'r-');
        end
    
    end
    
end

hx = [];
h_inter = [];
means = [];
for k = 1:length(lambda)
    for j = 1:L
        w = pinv( phi_all(:,:,j)'*phi_all(:,:,j) + lambda(k)*eye( size(phi,2)) )*phi_all(:,:,j)'*t;
        h_inter =  phi_all(:,:,j)*w;
        hx = [hx;h_inter'];
    end
    means = [means;mean(hx)];
end
for k = 1:3
   subplot(3,2,k*2);
   plot(x,means(k,:),'r-');
   hold on;
   plot(x,mean(Dy),'g-'); 
end

bias_sq = (1/N)*sum((means - mean(Dy)).^2,2);
% variance = mean(mean(Dy(1:100,:) - means(1)).^2);
% for i=2:length(lambda)
%     var = mean(mean(Dy(100*(i-1):100*i,:) - means(i)).^2);
%     variance = [variance var];
% end
function z = designMatrix(X,s)
    z = ones(1,length(X))';
%     z = [];
    for i = 1:length(X)-1
        inter = exp(-(X - X(i)).^2 / (2*s^2));
        z = [z inter];
    end
end
