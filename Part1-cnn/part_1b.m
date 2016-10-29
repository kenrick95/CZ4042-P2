%cnnPreprocess('..\Images_Data_Clipped');

load 'dataTeststore.mat';
load 'dataTrainstore.mat';

imageDim = 28;

randn('seed', 42);
s = RandStream('mcg16807','Seed', 42);
RandStream.setGlobalStream(s);

layers = [imageInputLayer([imageDim imageDim]), ...
    convolution2dLayer([5, 5],30), ...
    averagePooling2dLayer(2), ...
    convolution2dLayer([5, 5],50), ...
    averagePooling2dLayer(2), ...
    fullyConnectedLayer(10), ...
    softmaxLayer(), ...
    classificationLayer()];

options = trainingOptions('sgdm', ... 
    'MaxEpochs', 20,...
    'InitialLearnRate', 3e-4, ...
    'MiniBatchSize', 500, ...
    'L2Regularization', 1e-4, ...
    'Momentum', 9e-1 ...
    );

convnet = trainNetwork(dataTrainstore,layers,options);

YTest = classify(convnet, dataTeststore);
TTest = dataTeststore.Labels;
accuracy = sum(YTest == TTest)/numel(YTest);

disp(accuracy);
