% single time series prediction

clc
clear all
close all

%% Initialize Deep Network Parameters
inputSize = 1;
sparsityParam = 0.1;   % desired average activation of the hidden units.    

inputZeroMaskedFraction   = 0.0;  % denoising ratio
dropoutFraction  = 0.0;          % dropout ratio

% the parameter need be optimized
% hiddenSize = 40;    % Layer 1 Hidden Size
% lambda = 3e-5;         % weight decay parameter  
% beta = 0.01;              % weight of sparsity penalty term
% order = 3;  % order of HFCM
% C = 1e-4; % parameter for ridge regression

%% Load data

load('sp500.csv');
data = sp500;

% minmaxnorm
mindata = min(min(data));
maxdata = max(max(data));
data = ((data-mindata(1))/(maxdata(1)-mindata(1)));

%% STEP 2: Train sparse autoencoder
lambda1V = [10,1,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9,1e-10];
lambdaV = [1,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8];
betaV = [0.01,0.1,0.5,1:15];
HiddensizeV = [10:5:100];
orderV = 2:1:15;
RMSE = [];
% [1e-1,1e-3,1e-5,1e-7,1e-10,1e-13,1e-15,1e]
for m = 3:3%length(orderV) % L
for i = 5:5%length(HiddensizeV) % k5
for j = 5:5 % tao7
for l = 13:13 % alpha
for g = 2:2%length(lambdaV) % beta
for k = 8:8 %lambda
C = (10^(-k-2));
CC(k) = C;
% % the parameter need be optimized
hiddenSize = HiddensizeV(i);
lambda = lambdaV(g);
beta = betaV(l);
lambda1 = lambda1V(j);
order = orderV(m);
% Randomly initialize the parameters


numratio = 163;
trainData = data(1:numratio,:)';
trainLabels = trainData((order+1:end))';
% 
testData = data(numratio-order:end,:)';
testLabels = testData((order+1:end))';

% theta = initializeParameters(hiddenSize, inputSize);
seed = 1;
theta = initializeParameters_nonneg(hiddenSize, inputSize, seed);

%  Use minFunc to minimize the function
addpath minFunc/
options.Method = 'lbfgs';
options.maxIter = 400;
options.display = 'on';

[opttheta, cost] = minFunc( @(p) sparseAutoencoderCost_nonneg(p, ...
                                   inputSize, hiddenSize, ...
                                   lambda, inputZeroMaskedFraction,...
                                   dropoutFraction, sparsityParam, ...
                                   beta, trainData), ...
                                   theta, options);
                                                 

[saeFeatures] = feedForwardAutoencoder(opttheta, hiddenSize, ...
                                        inputSize, dropoutFraction, trainData);

%% Construct high-order FCMs

% % Randomly initialize the parameters
WFCM = rand(hiddenSize*order,hiddenSize)*0.1;
WFCM = [WFCM;zeros(1,hiddenSize)];

%% Establish the model of output

% the output of HFCM
W2 = WFCM(1:hiddenSize,:);
b2 = WFCM(end,:);
Wx = WFCM(hiddenSize+1:end-1,:);

[a2,a2t] = DataforHFCM(saeFeatures,order); % output of training data and target

a3 = sigmoid(W2'*a2 + repmat(b2',1,size(a2,2)) + Wx'*a2t); % output of HFCM
% 
atemp = [a2', a3'];
% the weight matrix of output
W3 = ridge(trainLabels,atemp,C);
%% Fine-tuning AE,HFCM

W2 = WFCM(1:hiddenSize,:);
b2 = WFCM(end,:);
Wx = WFCM(hiddenSize+1:end-1,:);

% W2
thetaW2 = [W2(:) ; Wx(:) ; b2(:)];
options.Method = 'lbfgs';
options.maxIter = 400;	  
options.display = 'on';

dbstop if error
[OptThetaW2,cost1] = minFunc( @(p) HFCMCostW2x(p, saeFeatures, hiddenSize, ...
                                         lambda1, trainData, order, ...
                                 W3),thetaW2, options);


WFCM(1:hiddenSize,:) = reshape(OptThetaW2(1:hiddenSize*hiddenSize),...
                                                  hiddenSize, hiddenSize);

WFCM(end,:) = OptThetaW2(hiddenSize*hiddenSize*order+1:end)';

WFCM(hiddenSize+1:end-1,:) = reshape(OptThetaW2(hiddenSize*hiddenSize+1:hiddenSize*hiddenSize*order),...
    hiddenSize*(order-1), hiddenSize);
                                                             

%% the optimized W3
% % the output of HFCM
% W2 = WFCM(1:hiddenSize,:);
% b2 = WFCM(end,:);
% Wx = WFCM(hiddenSize+1:end-1,:);
% 
% % [a2,a2t] = DataforHFCM(saeFeatures,order); % output of training data and target
% a3 = sigmf(W2'*a2 + repmat(b2',1,size(a2,2)) + Wx'*a2t, [1 0]); % output of HFCM
% atemp = [a3',a2'];

%% Test

% [R(1,k)] = AEHFCM_predict(opttheta, hiddenSize, inputSize, ...
%             dropoutFraction, trainData, WFCM, trainLabels, W3, order, [mindata,maxdata]);
[R(1,j)] = AEHFCM_predict(opttheta, hiddenSize, inputSize, ...
            dropoutFraction, testData, WFCM, testLabels, W3, order, [mindata,maxdata]);

end
% RMSE = [RMSE ; R];
end
end
end
end
end


function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end