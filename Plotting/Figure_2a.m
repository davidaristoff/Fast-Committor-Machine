% Figure 2a
% Set plotting preferences
close all; 
set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');
set(groot,'defaultTextInterpreter','latex');
set(groot,'DefaultAxesFontSize',18);
set(groot,'DefaultTextFontSize',18);

% Toy max info FCM
% FFM shape (iterations [fix], [MSE, runtime], sample_size, repeat #)
load('your_FCM_data.mat')
sample_iteration = 5;       % Max iteration
num_repeats = 10;           % For error bars
num_samples = 10;           % Number of dataset sizes
toy_fmc_mse = zeros(10,2);
for i=1:num_samples
    mse_dat = FFM_data(sample_iteration,1,i,:);
    mse_dat = reshape(mse_dat,[10,1]);
    toy_fmc_mse(i,1) = mean(mse_dat);
    toy_fmc_mse(i,2) = std(mse_dat)/sqrt(num_repeats);
end

% Toy NN data set
% Assumes the file contains 
  % avg_mses:   Average mean square error for num_repeats trials of neural net
      % The shape is num_samples x 2, first column are means, second column are sample standard deviations
  % data_sizes: The size of each data set (total of num_samples)
load('your_NN_data.mat')

figure('position',[20 20 350 300]); hold on
set(gca, 'XScale', 'log', 'YScale', 'log');
axis tight; 
xlabel('sample size');
ylabel('mean squared error');
sz = 5;
box on

% Plot means with error bars
scatter(data_sizes, toy_fmc_mse(:,1),sz,'o', 'b', 'filled') 
hErrorbar = errorbar(data_sizes, toy_fmc_mse(:,1), toy_fmc_mse(:,2), 'o', 'LineStyle', 'none', ...
    'MarkerSize', sz, 'MarkerEdgeColor', 'b',  'LineWidth', 1.5, 'Color', 'b');
scatter(data_sizes, avg_mses(:,1), sz, 'x', 'r', 'filled')
hErrorbar2 = errorbar(data_sizes, avg_mses(:,1), avg_mses(:,2)/sqrt(num_repeats), 'x', 'LineStyle', 'none', ...
    'MarkerSize', sz, 'MarkerEdgeColor', 'red', 'LineWidth', 1.5, 'Color', 'r');

% Dummy scatter plots to match in legend
hDummy1 = scatter(NaN, NaN, 'o', 'MarkerEdgeColor', 'b', 'LineWidth', 1.5);
hDummy2 = scatter(NaN, NaN, 'x', 'MarkerEdgeColor', 'r', 'LineWidth', 1.5);

% Add legend
legend([hDummy1, hDummy2], {'FCM', 'NN'});
