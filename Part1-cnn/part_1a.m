%cnnPreprocess('..\Images_Data_Clipped');

load 'dataTeststore.mat';
load 'dataTrainstore.mat';

dataTrainstoreSubset = imageDatastore('..\Images_Data_Clipped\Train\*','LabelSource','foldernames');
dataTeststoreSubset = imageDatastore('..\Images_Data_Clipped\Test\*','LabelSource','foldernames');

%Take 100 samples for training and 20 for testing
dataTrainstoreSubset.Files = dataTrainstore.Files(1:5000);
dataTeststoreSubset.Files = dataTeststore.Files(1:1000);

dataTrainstoreSubset.Labels = dataTrainstore.Labels(1:5000);
dataTeststoreSubset.Labels = dataTeststore.Labels(1:1000);

imageDim = 28;

layers = [imageInputLayer([imageDim imageDim]), ...
    convolution2dLayer([9, 9],20), ...
    averagePooling2dLayer(2), ...
    fullyConnectedLayer(10), ...
    softmaxLayer(), ...
    classificationLayer()];

options = trainingOptions('sgdm', ... 
    'MaxEpochs', 25,...
    'InitialLearnRate', 3e-4, ...
    'MiniBatchSize', 500, ...
    'L2Regularization', 1e-4, ...
    'Momentum', 9e-1 ...
    );

convnet = trainNetwork(dataTrainstoreSubset,layers,options);

YTest = classify(convnet, dataTeststoreSubset);
TTest = dataTeststoreSubset.Labels;
accuracy = sum(YTest == TTest)/numel(YTest);

disp(accuracy);
