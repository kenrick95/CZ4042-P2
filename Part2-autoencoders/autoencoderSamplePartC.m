%autoencoderPreprocess('..\Images_Data_Clipped');

load 'dataTest.mat';
load 'dataTrain.mat';
load 'labelsTrain.mat';
load 'labelsTest.mat';

hiddenSize1 = 500;
hiddenSize2 = 500;

maxEpochs1 = 100;                   % max 200
maxEpochs2 = 150;                   % max 200
maxEpochs3 = 100;                   % max 200
sparsityRegularization1 = 1.00;      % default: 1.00; 1 .. 10
sparsityRegularization2 = 1.00;      % default: 1.00; 1 .. 10
sparsityProportion1 = 0.15;          % default: 0.05; 0.1 .. 0.3
sparsityProportion2 = 0.15;          % default: 0.05; 0.1 .. 0.3
encoderTransferFunction = 'logsig'; % default: logsig; use logsig
decoderTransferFunction = 'logsig'; % default: logsig; use logsig
useGpu = false;
backprop = true;

randn('seed', 42);
s = RandStream('mcg16807','Seed', 42);
RandStream.setGlobalStream(s);

autoenc1 = trainAutoencoder(dataTrain, hiddenSize1, ...
    'MaxEpochs', maxEpochs1, ...
    'SparsityRegularization', sparsityRegularization1, ...
    'SparsityProportion', sparsityProportion1, ...
    'EncoderTransferFunction', encoderTransferFunction, ...
    'DecoderTransferFunction', decoderTransferFunction, ...
    'UseGPU', useGpu);
save('autoenc1.mat', 'autoenc1');

figure(), plotWeights(autoenc1);
print('exp-f1','-dpng')


feat1 = encode(autoenc1, dataTrain);
autoenc2 = trainAutoencoder(feat1, hiddenSize2, ...
    'MaxEpochs', maxEpochs2, ...
    'SparsityRegularization', sparsityRegularization2, ...
    'SparsityProportion', sparsityProportion2, ...
    'EncoderTransferFunction', encoderTransferFunction, ...
    'DecoderTransferFunction', decoderTransferFunction, ...
    'UseGPU', useGpu);
save('autoenc2.mat', 'autoenc2');
feat2 = encode(autoenc2, feat1);

figure(), plotWeights(autoenc2);
print('exp-f1-ly2','-dpng')


softnet = trainSoftmaxLayer(feat2, labelsTrain, ...
    'MaxEpochs', maxEpochs3);
save('softnet.mat', 'softnet');


% Make it 784 * 1000 matrix instead of 1 * 1000 matrix
xTest = zeros(28 * 28, numel(dataTest));
for i = 1:numel(dataTest)
    xTest(:,i) = dataTest{i}(:);
end

deepnet = stack(autoenc1,autoenc2,softnet);
save('deepnet.mat', 'deepnet');

if backprop
    xTrain = zeros(28 * 28, numel(dataTrain));
    for i = 1:numel(dataTrain)
        xTrain(:,i) = dataTrain{i}(:);
    end
    deepnet = train(deepnet, xTrain, labelsTrain);
    save('deepnet_bp.mat', 'deepnet');
end


y = deepnet(xTest);
figure(), plotconfusion(labelsTest,y);
print('exp-f-confmtx','-dpng')

reconstructed = decode(autoenc1, decode(autoenc2, encode(autoenc2, encode(autoenc1, dataTest))));

mseError = 0;
for i = 1:numel(dataTest)
    mseError = mseError + mse(double(dataTest{1, i}) - reconstructed{1, i});
end


accuracy = 1.0 - confusion(labelsTest,y);


fileID = fopen('exp.txt','w');
fprintf(fileID, strcat('[enc2_layers: %d; enc2_layers: %d];', ...
    'maxEpochs: [%5d %5d %5d], ', ...
    'sparsityRegularization1: %5.3f, ', ...
    'sparsityRegularization2: %5.3f, ', ...
    'sparsityProportion1: %5.3f, ', ...
    'sparsityProportion2: %5.3f, ', ...
    'backpropagation: %i, ', ...
    'mseError: %5.10e, ', ...
    'accuracy: %5.10e, ', ...
    'encoderTransferFunction: %s, decoderTransferFunction: %s\r\n'), ...
    hiddenSize1, ...
    hiddenSize2, ...
    maxEpochs1, ...
    maxEpochs2, ...
    maxEpochs3, ...
    sparsityRegularization1, ...
    sparsityRegularization2, ...
    sparsityProportion1, ...
    sparsityProportion2, ...
    backprop, ...
    mseError, ...
    accuracy, ...
    encoderTransferFunction, ...
    decoderTransferFunction);
fclose(fileID);

