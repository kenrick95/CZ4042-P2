%autoencoderPreprocess('..\Images_Data_Clipped');

load 'dataTest.mat';
load 'dataTrain.mat';

%Take 100 samples for training and 20 for testing
% dataTestSubset = dataTest(1, 1:20);
% dataTrainSubset = dataTrain(1, 1:100);

hiddenSize1 = 100;
hiddenSize2 = 50;

maxEpochs = 200;                    % default: 10000,   best: 200
sparsityRegularization = 0.25;      % default: 1.00,    best: 1.50
sparsityProportion = 0.04;          % default: 0.05,    best: 0.04
encoderTransferFunction = 'logsig'; % default: logsig,  best: logsig
decoderTransferFunction = 'logsig'; % default: logsig,  best: logsig

randn('seed', 42);
s = RandStream('mcg16807','Seed', 42);
RandStream.setGlobalStream(s);

autoenc1 = trainAutoencoder(dataTrain, hiddenSize1, ...
    'MaxEpochs', maxEpochs, ...
    'SparsityRegularization', sparsityRegularization, ...
    'SparsityProportion', sparsityProportion, ...
    'EncoderTransferFunction', encoderTransferFunction, ...
    'DecoderTransferFunction', decoderTransferFunction);
save('autoenc1.mat', 'autoenc1');

figure(), plotWeights(autoenc1);
print('exp-f1','-dpng')


feat1 = encode(autoenc1, dataTrain);
autoenc2 = trainAutoencoder(feat1, hiddenSize2, ...
    'MaxEpochs', maxEpochs, ...
    'SparsityRegularization', sparsityRegularization, ...
    'SparsityProportion', sparsityProportion, ...
    'EncoderTransferFunction', encoderTransferFunction, ...
    'DecoderTransferFunction', decoderTransferFunction);
save('autoenc2.mat', 'autoenc2');
feat2 = encode(autoenc2, feat1);

figure(), plotWeights(autoenc2);
print('exp-f1-ly2','-dpng')

reconstructed = decode(autoenc1, decode(autoenc2, encode(autoenc2, encode(autoenc1, dataTest))));

mseError = 0;
for i = 1:numel(dataTest)
    mseError = mseError + mse(double(dataTest{1, i}) - reconstructed{1, i});
end

mseError = mseError/i;
fileID = fopen('exp.txt','w');
fprintf(fileID, '[enc2_layers: %d; enc2_layers: %d]; maxEpochs: %5d, sparsityRegularization: %5.3f, sparsityProportion: %5.3f, mseError: %5.10e, encoderTransferFunction: %s, decoderTransferFunction: %s\r\n', ...
    hiddenSize1, ...
    hiddenSize2, ...
    maxEpochs, ...
    sparsityRegularization, ...
    sparsityProportion, ...
    mseError, ...
    encoderTransferFunction, ...
    decoderTransferFunction);
fclose(fileID);
figure;
for i = 1:20
    subplot(4,5,i);
    imshow(dataTest{i});
end
print('exp-f2','-dpng')

figure;
for i = 1:20
    subplot(4,5,i);
    imshow(reconstructed{i});
end
print('exp-f3','-dpng')

