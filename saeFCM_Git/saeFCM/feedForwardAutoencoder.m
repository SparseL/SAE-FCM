function [activation] = feedForwardAutoencoder(theta, hiddenSize, visibleSize, dropoutFraction, data)



W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);


z2 = W1*data + repmat(b1,1,size(data,2));
activation = sigmoid(z2);

if(dropoutFraction > 0)
   activation = activation.*(1 - dropoutFraction);
end
 


end


function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
