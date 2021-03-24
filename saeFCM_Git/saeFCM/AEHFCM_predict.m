function [RMSE, SMAPE, NRMSE] = AEHFCM_predict(opttheta, hiddenSize, ...
                   inputSize, dropoutFraction, testData, WFCM, testLabels, W3, order, Mdata)
%% Test

[saeTest] = feedForwardAutoencoder(opttheta, hiddenSize, ...
                                        inputSize, dropoutFraction, testData);

% data for HFCM of test

% the output of HFCM
W2 = WFCM(1:hiddenSize,:);
b2 = WFCM(end,:);
Wx = WFCM(hiddenSize+1:end-1,:);

[a2,a2t] = DataforHFCM(saeTest,order); % output of training data and target
a3 = sigmf(W2'*a2 + repmat(b2',1,size(a2,2)) + Wx'*a2t, [1 0]); % output of HFCM

a3 = [a2; a3];

outf = W3'*a3; % output of fianl layer

% outfinal = outf';
outfinal = (Mdata(2)-Mdata(1))*outf' + Mdata(1);
testLabels = (Mdata(2)-Mdata(1))*testLabels + Mdata(1);
% performance measurement
% RMSE
RMSE = sqrt(sum((outfinal-testLabels).^2)/length(testLabels));
% RMSE = sqrt(sum((outfinal-testLabels).*(outfinal-testLabels))/length(testLabels));
% SMAPE
SMAPE = sum(abs(outfinal-testLabels))/sum(abs(outfinal)+abs(testLabels));
% NRMSE
outmean = mean(testLabels);
NRMSE = sqrt(sum((outfinal-testLabels).^2)/...
    sum((outfinal-outmean).^2));
% if RMSE < 0.002
figure
% subplot(2,1,1);
plot(outfinal,'r-*');
hold on
plot(testLabels,'b-.');
% legend('predicted data','target data');
% % xlabel('Time');
% axis tight
% ylabel('S&P 500');
% subplot(2,1,2);
% plot(testLabels-outfinal,'b-');
% xlabel('Time');
% ylabel('Absolute Error');
% axis tight
% end
% fprintf('After Finetuning Test Accuracy: %0.3f%%\n', acc_after(seed) * 100);

end