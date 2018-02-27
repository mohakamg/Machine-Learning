function dispRandImg(X)
    rand_img_index = round(rand(1)*5000);
    rand_img_unfolded = X(rand_img_index,:);
    rand_img_folded = reshape(rand_img_unfolded,[20,20]);

    imshow(rand_img_folded);
    fg = get(groot,'CurrentFigure');
    fg.Units = 'Normalized';
    fg.OuterPosition = [0 0 1 1];
end