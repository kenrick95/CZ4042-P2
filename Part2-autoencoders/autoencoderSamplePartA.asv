%autoencoderPreprocess('..\Images_Data_Clipped');

load 'dataTest.mat';
load 'dataTrain.mat';

%Take 100 samples for training and 20 for testing
% dataTestSubset = dataTest(1, 1:20);
% dataTrainSubset = dataTrain(1, 1:100);

hiddenSize1 = 100;

maxEpochs = 200;
sparsityRegularization = 0.75;
sparsityProportion = 0.04;

randn('seed', 42);
s = RandStream('mcg16807','Seed', 42);
RandStream.setGlobalStream(s);

autoenc1 = trainAutoencoder(dataTrainSubset,hiddenSize1, ...
    'MaxEpochs', maxEpochs, ...
    'SparsityRegularization', sparsityRegularization, ...
    'SparsityProportion', sparsityProportion);

figure(), plotWeights(autoenc1);
print('exp-f1','-dpng')

reconstructed = predict(autoenc1, dataTestSubset);

mseError = 0;
for i = 1:numel(dataTestSubset)
    mseError = mseError + mse(double(dataTestSubset{1, i}) - reconstructed{1, i});
end

mseError = mseError/i;
fileID = fopen('exp.txt','w');
fprintf(fileID, 'maxEpochs: %5d, sparsityRegularization: %5.3f, sparsityProportion: %5.3f, mseError: %5.10e\r\n', ...
    maxEpochs, ...
    sparsityRegularization, ...
    sparsityProportion, ...
    mseError);
fclose(fileID);
figure;
for i = 1:20
    subplot(4,5,i);
    imshow(dataTestSubset{i});
end
print('exp-f2','-dpng')

figure;
for i = 1:20
    subplot(4,5,i);
    imshow(reconstructed{i});
end
print('exp-f3','-dpng')

