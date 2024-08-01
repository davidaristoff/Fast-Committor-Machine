%% Figure 2a/5a ===================================================================================

% Set plotting preferences
close all; 
set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');
set(groot,'defaultTextInterpreter','latex');
set(groot,'DefaultAxesFontSize',18);
set(groot,'DefaultTextFontSize',18);

% Data from FCM as 'FFM.mat'
% FFM shape (iterations [fix], [MSE, runtime], sample_size, repeat #)
load('your_FCM_data.mat')
sample_iteration = 5;       % Max iteration
num_repeats = 10;           % For error bars
num_samples = 10;           % Number of dataset sizes
fmc_mse = zeros(10,2);
for i=1:num_samples
    mse_dat = FFM_data(sample_iteration,1,i,:);
    mse_dat = reshape(mse_dat,[10,1]);
    fmc_mse(i,1) = mean(mse_dat);
    fmc_mse(i,2) = std(mse_dat)/sqrt(num_repeats);
end

% NN data set
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
scatter(data_sizes, fmc_mse(:,1),sz,'o', 'b', 'filled') 
hErrorbar = errorbar(data_sizes, fmc_mse(:,1), fmc_mse(:,2), 'o', 'LineStyle', 'none', ...
    'MarkerSize', sz, 'MarkerEdgeColor', 'b',  'LineWidth', 1.5, 'Color', 'b');
scatter(data_sizes, avg_mses(:,1), sz, 'x', 'r', 'filled')
hErrorbar2 = errorbar(data_sizes, avg_mses(:,1), avg_mses(:,2)/sqrt(num_repeats), 'x', 'LineStyle', 'none', ...
    'MarkerSize', sz, 'MarkerEdgeColor', 'red', 'LineWidth', 1.5, 'Color', 'r');

% Dummy scatter plots to match in legend
hDummy1 = scatter(NaN, NaN, 'o', 'MarkerEdgeColor', 'b', 'LineWidth', 1.5);
hDummy2 = scatter(NaN, NaN, 'x', 'MarkerEdgeColor', 'r', 'LineWidth', 1.5);

% Add legend
legend([hDummy1, hDummy2], {'FCM', 'NN'});

%% Figure 2b/5b ===================================================================================

% Data from FCM as 'FFM.mat'
% FFM shape (iterations [fix], [MSE, runtime], sample_size, repeat #)
load('your_FCM_data.mat')
sample_iteration = 5;       % Max iteration
num_repeats = 10;           % For error bars
num_samples = 10;           % Number of dataset sizes
fmc_speed = zeros(10,2);
for i=1:num_samples
    speed_dat = FFM_data(sample_iteration,2,i,:);
    speed_dat = sample_iteration*reshape(speed_dat,[10,1]);
    fmc_speed(i,1) = mean(speed_dat);
    fmc_speed(i,2) = std(speed_dat)/sqrt(num_repeats);
end

% NN data set
% Assumes the file contains 
  % avg_mses:   Average mean square error for num_repeats trials of neural net
      % The shape is num_samples x 2, first column are means, second column are sample standard deviations
  % data_sizes: The size of each data set (total of num_samples)
load('your_NN_data.mat')

figure('position',[20 20 350 300]); hold on
set(gca, 'XScale', 'log', 'YScale', 'log');
axis tight; 
%ylim([1e-1 1e4])
xlabel('sample size');
ylabel('wall clock time');
box on

sz = 5;
scatter(data_sizes, fmc_speed(:,1),sz,'o', 'b', 'filled') 
hErrorbar = errorbar(data_sizes, fmc_speed(:,1), fmc_speed(:,2), 'o', 'LineStyle', 'none', ...
    'MarkerSize', sz, 'MarkerEdgeColor', 'b',  'LineWidth', 1.5, 'Color', 'b');
scatter(data_sizes, nn_speed(:,1), sz, 'x', 'r', 'filled')
hErrorbar2 = errorbar(data_sizes, nn_speed(:,1), nn_speed(:,2)/sqrt(num_repeats), 'x', 'LineStyle', 'none', ...
    'MarkerSize', sz, 'MarkerEdgeColor', 'red', 'LineWidth', 1.5, 'Color', 'r');

% Dummy scatter plots to match in legend
hDummy1 = scatter(NaN, NaN, 'o', 'MarkerEdgeColor', 'b', 'LineWidth', 1.5);
hDummy2 = scatter(NaN, NaN, 'x', 'MarkerEdgeColor', 'r', 'LineWidth', 1.5);

% Add legend
legend([hDummy1, hDummy2], {'FCM', 'NN'},'Location','best');

%% Figure 3a ===================================================================================

figure('position',[20 20 350 300]); hold on
axis tight; 
xlabel('iteration');
ylabel('mean squared error');
box on

% X_perf (size, iterations, [MSE, STDerror])
toy_perf = zeros(5, 5, 2); 

% Loop over the first 5 files reading in results and storing
output_folder = 'Outputs/data_for_toy';
sizes = [200,400,600,800,1000];
% Below assuming files are named 'FCM_alanine_Ns10_rSIZE_gam1e-06_h1_iters5_repeats10.mat'
% where SIZE indicates # columns
for i = 1:5
    filename = fullfile(output_folder, ['FCM_alanine_Ns10_r', num2str(sizes(i)), '_gam1e-06_h1_iters5_repeats10.mat']);
    %toy
    % Load the data from the file
    load(filename)
    data = squeeze(FFM_data(:,1,i,:));

    % Store mean and standard deviation
    toy_perf(i,:,1) = mean(data,2);
    toy_perf(i,:,2) = std(data,0,2)/sqrt(10);
end

sz = 5;
hErrorbar = gobjects(5, 1);

for i=1:5
    scatter(1:5, toy_perf(i,:,1),sz)
    hErrorbar(i) = errorbar(1:5, toy_perf(i,:,1), toy_perf(i,:,2), 'linewidth',1.5);
end
names = string(sizes) + ' columns';
legend(hErrorbar, names ,'Location','northeast');

%% Figure 3b ===================================================================================
figure('position',[20 20 350 300]); hold on
set(gca, 'YScale', 'log'); %, 'XScale', 'log'
axis tight; 
ylim([1e-5 1e-1])
xlabel('epochs');
ylabel('mean squared error');
box on
hold on;

% Load NN data and assign c1, c2, etc
load('your_NN_data.csv')

% citraj are MSE errors from NN for particular learning rates
c1traj = c1(c1>0);
c2traj = c2(c2>0);
c3traj = c3(c3>0);
c4traj = c4(c4>0);

plot(c2traj, 'LineWidth', 1.5, 'linestyle', '--');
plot(c3traj, 'LineWidth', 1.5, 'linestyle', '-.');
plot(c4traj, 'LineWidth', 1.5, 'linestyle', '-');

legend('$lr = 10^{-2}$', '$lr = 10^{-3}$', '$lr = 10^{-4}$', 'Location','northeast')
