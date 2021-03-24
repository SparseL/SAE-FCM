function [X,X_] = DataforHFCM(sae1Features,order)

X_ = [];
FCMData = sae1Features;
X = FCMData(:,order:end-1);
for i = 1:order-1
    X_ = [X_;FCMData(:,i:end-order+i-1)];
end
% Y_test = Y;
% [S,I] = find(Y == 1);
% for i = 1:length(S)
%     Y(S(i),I(i)) = 0.999999;
% end
% case 1, sigmoid
% Y = -log((1-Y)./Y)/k;
% case 2, tanh
% Y = log((Y+1)./(1-Y))/k;

end