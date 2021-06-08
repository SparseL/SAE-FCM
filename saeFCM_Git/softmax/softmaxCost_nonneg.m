function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);

numCases = size(data, 2);

groundTruth = full(sparse(labels, 1:numCases, 1));
cost = 0;

thetagrad = zeros(numClasses, inputSize);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.

% tmp = theta*data;

prob = exp(theta*data);

[r,c] = find(isinf(prob));
prob(r,c) = exp(709);  % avoid Inf in prob matrix

prob_norm = prob./repmat(sum(prob),numClasses,1);

[r,c] = find(prob_norm == 0);
prob_norm(r,c) = eps;

theta_neg = zeros(size(theta,1), size(theta,2));

theta_neg(find(theta<0)) = theta(find(theta<0));

theta_neg_abs = theta_neg;
theta_neg_abs(theta_neg_abs~=0)=1;

weight_neg_decay = sum(sum(theta_neg.^2)) ;


cost = -sum(sum(groundTruth.*log(prob_norm)))/numCases + lambda/2*weight_neg_decay;

cost_acc = -sum(sum(groundTruth.*log(prob_norm)))/numCases
if isnan(cost_acc)
    error()
end

% cost = -sum(sum(groundTruth.*log(prob_norm)))/numCases + lambda/2*weight_neg_decay - 0.5*lambda*(sum(theta_neg(:)));

thetagrad = -1/numCases * (data*(groundTruth-prob_norm)') ;

thetagrad = thetagrad' + lambda*theta_neg;

% thetagrad = thetagrad' + lambda*theta_neg - 0.5*lambda*theta_neg_abs;



% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end

