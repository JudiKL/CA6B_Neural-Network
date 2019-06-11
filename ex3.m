%% Setup the parameters
input_layer_size  = 2;  % 2 electrodes
num_labels = 2;          % 2 labels, left or right
                          % (note that left is mapped 1 and right 2                        
%% =========== Part 1: Loading and Visualizing Data =============

[X,y] = extract_data('01cr.fdt');
[X] = data_redux(X);
[rows, column] = size(X);
y = y';

left = find(labels==1); right = find(labels == 2);
% Plot Examples

plot(data(left, 1), data(left, 2), 'k+','LineWidth', 2, ...
'MarkerSize', 7);
hold on
plot(data(right, 1), data(right, 2), 'ko', 'MarkerFaceColor', 'y', ...
'MarkerSize', 7);

xlabel('mean voltage per trial in electrode L-HEOG')
ylabel('mean voltage per trial in electrode R-HEOG')
title('distribution of ocular electrodes mean voltage as a function of electrode channel')

legend('left saccadic eye movement','right saccadic eye movement')


fprintf('Program paused. Press enter to continue.\n');
pause;


%% ============ Part 2: One-vs-All Training ============
fprintf('\nTraining One-vs-All Logistic Regression...\n')

lambda = 0.1;
[all_theta] = oneVsAll(data, labels, num_labels, lambda);

fprintf('Program paused. Press enter to continue.\n');
pause;



%% ================ Part 3: Predict for One-Vs-All ================

pred = predictOneVsAll(all_theta, data);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == labels)) * 100);

