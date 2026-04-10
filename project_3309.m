% MATLAB SCRIPT for Final Project
% ECE 3309-002 Fall 2023

%% Initialize

close all;
clear all;
clc;

addpath(genpath(pwd))

%% Load dataset

% Replace with your directory path and file name
data_dir = 'C:\Users\Harshit Parmar\OneDrive - Texas Tech University\Spring2023\Projects\mnist';

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
Nh = 32;        % Number of hidden layer nodes

% YOUR CODE to initialize weights and bias
% Make sure the matrix size are proper.
% wh size => [Nh,Ni]
% wo size => [No,Nh]exam1
% bh size => [Nh,1]
% bo size => [No,1]

% Uncomment and modify code below.

%[wh,wo,bh,bo] = init_weights(Ni,Nh,No);

% You may create a function or add code lines directly here.
% DO NOT CHANGE VARIABLE NAMES

%% Neural Network Training

% You may run this section multiple times to keep on training and fine tuning your network.
% Usually 3 (hyper)parameters are modified with each fine-tuning run
% You may play around with different values for the 3 variables below.

BZ = 16;        % Minibatch size
% (Larger number will result in slower training and smaller number will result in poor training)

EPH = 500;      % Epochs (How many times to train over entire dataset.)

lr = 0.01;     % Learning rate. (Start with 0.01 and then tweak it to make perfect.)
% Too large learning rates may result in divering error and will blow up to a large number
% Too small learning rate may result in very very slow training.
% You may reduce this number gradually as the network trains to make accuracy better and better.



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
        T = one_hot(YYY(ss:ee),No);

        % Your function or code to do forward propagation
        % Forward pass requires minibatch, weights and bias.
        % It should output prediction and hidden layer output
        % Do not change the variable name otherwise the code might not work
        % Uncomment the line below and put your own code or your function name

        %[y_hat,H] = forward_pass(XX(ss:ee,:),wh,bh,wo,bo);
             % y_hat : Predicted output
             % H     : Output of hidden layer
             % XX(s:e,:) : Minibatch input
             % wh,wo : Weight for hidden and output layer
             % bh,bo : Bias for hidden and output layer


        Err(cnt,1) = sum(sqrt(sum((T-y_hat).^2,2)))./BZ;

        Py = inv_oneHot(T);
        Pt = inv_oneHot(y_hat);

        Err(cnt,2) = sum(Py == Pt)./BZ;

        % Your function or code to do backpropagation
        % Backpropagation should output gradients corresponding to weights and bias
        % The input and output variable names are for reference.
        % Do not change the variable names otherwise the code might not work.
        % Uncomment the line below and put your own code or your function name

        %[g_wh,g_bh,g_wo,g_bo] = backprop(X(ss:ee,:),y_hat,T,H,wh,bh,wo,bo,BZ);
             % g_wh, g_wo : Gradient for hidden and output weights
             % g_wh, g_wo : Gradient for hidden and output bias

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


end

%% Testing

[yy,~] = forward_pass(X_test,wh,bh,wo,bo);
Y_prd =  inv_oneHot(yy);
ACC = sum(Y_prd == Y_test)./N_test*100;
str = ['Testing Accuracy = ',num2str(ACC),' %'];
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




%% FUNCTION

% You may add or remove functions according to the need of you code.

% Uncomment and modify to use a function if you need them


%function [wh,wo,bh,bo] = init_weights(NF,NH,NO)
% Code to initialize weights and biases
% Return all the required matrices and outputs
%end

%function [y_hat,H] = forward_pass(X,wh,bh,wo,bo)
% Code to initialize weights and biases
% Return all the required matrices and outputs
%end

%function [g_wh,g_bh,g_wo,g_bo] = backprop(X,y_hat,T,H,wh,bh,wo,bo,BZ)
% Code to initialize weights and biases
% Return all the required matrices and outputs
%end


    
% Other useful functions

function z = sigmaa(s)
z = 1./(1+exp(-s));
end

function ss = softmax(z)
ES = sum(exp(z),2);
ss = exp(z)./ES;
end

function oh = one_hot(y,P)
N = length(y);
oh = zeros(N,P);

for i = 1:N
    oh(i,y(i)+1) = 1;
end

end

function P = inv_oneHot(y)
[~,P] = max(y,[],2);
P = P-1;
end

