img = imread('turkevich.jpg');
[m,n] = size(img);
figure
hold on;
for i = 1:100
    for j=1:100
       plot3(i,j,img(i,j)); 
    end
end