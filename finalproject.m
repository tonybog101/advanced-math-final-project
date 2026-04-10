% MATLAB SCRIPT for Final Project
% ECE 3309-002 Fall 2023

%% Initialize

close all;
clear all;
clc;

addpath(genpath(pwd))

%% Load dataset

% Replace with your directory path and file name
data_dir = 'C:\Users\gateway\Downloads';

% Training Data
fname_train = 'mnist_train.csv';

M = readmatrix(fullfile(data_dir,fname_train));
X = M(:,2:end);
Y = M(:,1);
N = length(Y);

% Testing Data
fname_test = 'mnist_test.csv';

M_test = readmatrix(fullfile(data_dir,fname_test));
X_test = M_test(:,2:end);
Y_test = M_test(:,1);
N_test = length(Y_test);

% Normalize data
X = X./255;
X_test = X_test./255;


%% Display dataset

% Display Training data samples

rand_disp = randperm(N,20);

    figure(10)

    for si = 1:20
        subplot(4,5,si)
        imagesc(reshape(X(rand_disp(si),:),28,28)')
        str = ['Digit: ',num2str(Y(rand_disp(si)))];
        title(str)
        colormap gray
        axis('square')
        xticks([])
        yticks([])
    end
drawnow;


%% Specify number of hidden units and initialize weights

Ni = size(X,2); % Number of input features. (28x28: one per pixel)
No = 10;        % Number of output classes (10 digits)

% Modify this to alter Neural Network performance
Nh = 64;       % Number of hidden layer nodes (increased from 32 for better accuracy)

% Initialize weights and biases
% wh size => [Nh, Ni]
% wo size => [No, Nh]
% bh size => [Nh, 1]
% bo size => [No, 1]

[wh, wo, bh, bo] = init_weights(Ni, Nh, No);


%% Neural Network Training

% You may run this section multiple times to keep on training and fine tuning your network.

BZ = 64;        % Minibatch size

EPH = 2;      % Epochs

lr = 0.08;      % Learning rate

save_every = 25; % Save model every N epochs

NB = floor(N/BZ);

Err = NaN.*zeros(EPH*NB,2);

cnt = 1;
for i = 1:EPH
    i_rand1 = randperm(N);
    XX = X(i_rand1,:);
    YYY = Y(i_rand1);

    for bz = 1:NB
        ss = (bz-1)*BZ + 1;
        ee = bz*BZ;
        T = one_hot(YYY(ss:ee), No);

        % Forward propagation
        [y_hat, H] = forward_pass(XX(ss:ee,:), wh, bh, wo, bo);

        Err(cnt,1) = sum(sqrt(sum((T-y_hat).^2,2)))./BZ;

        Py = inv_oneHot(T);
        Pt = inv_oneHot(y_hat);

        Err(cnt,2) = sum(Py == Pt)./BZ;

        % Backpropagation
        [g_wh, g_bh, g_wo, g_bo] = backprop(XX(ss:ee,:), y_hat, T, H, wh, bh, wo, bo, BZ);

        % Gradient descent update
        wh = wh - lr.*g_wh;
        wo = wo - lr.*g_wo;
        bh = bh - lr.*g_bh;
        bo = bo - lr.*g_bo;

        figure(1)
        yyaxis left
        plot(Err(:,2),'b-')

        hold on
        yyaxis right
        plot(Err(:,1),'r-');

        legend({'Accuracy','Error'})
        hold off
        drawnow;
        cnt = cnt+1;

    end

    % Save model parameters periodically
    if mod(cnt, 100) == 0
    figure(1)
    yyaxis left
    plot(Err(:,2),'b-')
    hold on
    yyaxis right
    plot(Err(:,1),'r-');
    legend({'Accuracy','Error'})
    hold off
    drawnow;
end

end

% Save final model
save('model_final.mat', 'wh', 'wo', 'bh', 'bo');
fprintf('Final model saved.\n');


%% Testing

[yy,~] = forward_pass(X_test, wh, bh, wo, bo);
Y_prd = inv_oneHot(yy);
ACC = sum(Y_prd == Y_test)./N_test*100;
str = ['Testing Accuracy = ', num2str(ACC), ' %'];
disp(str)

    rand_disp = randperm(N_test,20);

    figure(2)

    for si = 1:20
        subplot(4,5,si)
        imagesc(reshape(X_test(rand_disp(si),:),28,28)')
        str = ['T:',num2str(Y_test(rand_disp(si))),', P:',num2str(Y_prd(rand_disp(si)))];
        title(str)
        colormap gray
        axis('square')
        xticks([])
        yticks([])
    end


%% Export final weights to Excel

filename = 'model_weights_groupXX.xlsx';
writematrix(wh, filename, 'Sheet', 'wh');
writematrix(wo, filename, 'Sheet', 'wo');
writematrix(bh, filename, 'Sheet', 'bh');
writematrix(bo, filename, 'Sheet', 'bo');
fprintf('Weights exported to %s\n', filename);


%% FUNCTIONS

function [wh, wo, bh, bo] = init_weights(NF, NH, NO)
    % Initialize weights using uniform distribution in [-0.5, 0.5]
    % This ensures a mix of positive and negative values with moderate magnitude.
    %
    % wh: [NH x NF]  - hidden layer weights
    % wo: [NO x NH]  - output layer weights
    % bh: [NH x 1]   - hidden layer bias
    % bo: [NO x 1]   - output layer bias

    wh = (rand(NH, NF) - 0.5);   % uniform in [-0.5, 0.5]
    wo = (rand(NO, NH) - 0.5);   % uniform in [-0.5, 0.5]
    bh = (rand(NH,  1) - 0.5);   % uniform in [-0.5, 0.5]
    bo = (rand(NO,  1) - 0.5);   % uniform in [-0.5, 0.5]
end


function [y_hat, H] = forward_pass(X, wh, bh, wo, bo)
    % Forward propagation through the two-layer neural network.
    %
    % Hidden layer:
    %   H = sigmoid( X * wh^T + bh^T )
    %   X  : [NB x NI]
    %   wh : [NH x NI]  => wh^T : [NI x NH]
    %   bh : [NH x  1]  => bh^T : [ 1 x NH]  (broadcast across batch)
    %   H  : [NB x NH]
    %
    % Output layer:
    %   y_hat = softmax( H * wo^T + bo^T )
    %   wo : [NO x NH]  => wo^T : [NH x NO]
    %   bo : [NO x  1]  => bo^T : [ 1 x NO]
    %   y_hat : [NB x NO]

    % Hidden layer linear transformation + sigmoid activation
    S_h = X * wh' + bh';        % [NB x NH]
    H   = sigmaa(S_h);          % [NB x NH]

    % Output layer linear transformation + softmax activation
    S_o   = H * wo' + bo';      % [NB x NO]
    y_hat = softmax(S_o);       % [NB x NO]
end


function [g_wh, g_bh, g_wo, g_bo] = backprop(X, y_hat, T, H, wh, bh, wo, bo, BZ)
    % Backpropagation: compute gradients for all parameters.
    %
    % Inputs:
    %   X     : minibatch input        [NB x NI]
    %   y_hat : predicted output       [NB x NO]
    %   T     : one-hot targets        [NB x NO]
    %   H     : hidden layer output    [NB x NH]
    %   wh,bh : hidden layer params
    %   wo,bo : output layer params
    %   BZ    : minibatch size
    %
    % Error at output layer
    %   Delta = (y_hat - T) / N        [NB x NO]
    %
    % Error at hidden layer
    %   delta = (Delta * wo) .* H .* (1 - H)   [NB x NH]
    %
    % Gradients:
    %   dE/dwo = Delta^T * H           [NO x NH]
    %   dE/dbo = sum(Delta^T, 2)       [NO x  1]
    %   dE/dwh = delta^T * X           [NH x NI]
    %   dE/dbh = sum(delta^T, 2)       [NH x  1]

    % Output layer error
    Delta = (y_hat - T) ./ BZ;         % [NB x NO]

    % Hidden layer error (backprop through sigmoid)
    delta = (Delta * wo) .* H .* (1 - H);  % [NB x NH]

    % Gradients for output layer
    g_wo = Delta' * H;                  % [NO x NH]
    g_bo = sum(Delta', 2);              % [NO x  1]

    % Gradients for hidden layer
    g_wh = delta' * X;                  % [NH x NI]
    g_bh = sum(delta', 2);              % [NH x  1]
end


% ---- Provided utility functions (do not modify) ----

function z = sigmaa(s)
    z = 1./(1+exp(-s));
end

function ss = softmax(z)
    ES = sum(exp(z),2);
    ss = exp(z)./ES;
end

function oh = one_hot(y, P)
    N = length(y);
    oh = zeros(N, P);
    for i = 1:N
        oh(i, y(i)+1) = 1;
    end
end

function P = inv_oneHot(y)
    [~,P] = max(y,[],2);
    P = P-1;
end