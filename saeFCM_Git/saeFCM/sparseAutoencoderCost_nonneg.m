function [cost,grad, objhistory] = sparseAutoencoderCost_nonneg(theta, visibleSize, hiddenSize, ...
                                             lambda, inputZeroMaskedFraction, dropoutFraction, sparsityParam, beta, data)


objhistory = [];

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 
cost = 0;
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));

%% computing delta's in output and hidden layers


y = data;
a1 = data;
if (inputZeroMaskedFraction>0)
    a1 = a1.*(rand(size(a1))>inputZeroMaskedFraction);
end

z2 = W1*a1 + repmat(b1,1,size(a1,2));
a2 = sigmoid(z2);

%dropout
if(dropoutFraction > 0)
    dropOutMask = (rand(size(a2))>dropoutFraction);
    a2 = a2.*dropOutMask;
end

z3 = W2*a2 + repmat(b2,1,size(a2,2));
a3 = sigmoid(z3);

yhat = a3;

delta3 = -(y - yhat) .* (a3.*(ones(visibleSize,size(y,2))-a3));


param = sum(a2,2)./size(y,2);
par = sparsityParam*ones(hiddenSize,1);
sparsity = beta*(-par./param + (ones(hiddenSize,1)-par)./(ones(hiddenSize,1)-param));
sparsity = repmat(sparsity,1,size(data,2));

delta2 = (W2'*delta3 + sparsity) .* (a2.*(ones(hiddenSize,size(y,2))-a2));

if(dropoutFraction > 0)
    delta2 = delta2.*dropOutMask;
end


kl = sum(sparsityParam*log(par./param) + (1-sparsityParam)*log((ones(hiddenSize,1)-par)./(ones(hiddenSize,1)-param)));


idx1 = find(W1 < 0);
idx2 = find(W1 <= -1);
idx3 = find(W1 >= 0);

idx4 = find(W2 < 0);
idx5 = find(W2 <= -1);
idx6 = find(W2 >= 0);

L2_regN = sum(sum(W1(idx1).^2)) + sum(sum(W2(idx4).^2));
L2_regP = sum(sum(W1(idx3).^2)) + sum(sum(W2(idx6).^2));
L1_reg = sum(abs(W1(:))) + sum(abs(W2(:)));


cost = 0.5*sum(sum((y-yhat).^2))./size(y,2) + beta*kl + lambda/2*L2_regN;

newobj = 0.5*sum(sum((y-yhat).^2))./size(y,2);
objhistory = [objhistory newobj];

one1 = ones(size(W1));
one2 = ones(size(W2));

W1grad = delta2*(a1')./(size(y,2));
W1grad(idx1) = W1grad(idx1) + lambda*W1(idx1);

W2grad = delta3*(a2')./(size(y,2));
W2grad(idx4) = W2grad(idx4) + lambda*W2(idx4);

b1grad = sum(delta2,2)./(size(y,2));

b2grad = sum(delta3,2)./(size(y,2));



grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

 

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end

