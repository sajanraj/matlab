digitDatasetPath = fullfile('data\train');

trainData = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

digitDatasetPath = fullfile('data\val');

valData = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

layers = [
    imageInputLayer([256 256 1])

    convolution2dLayer(3,16,'Padding',1)
   % batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,32,'Padding',1)
    %batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,64,'Padding',1)
    %batchNormalizationLayer
    reluLayer

    fullyConnectedLayer(3)
    softmaxLayer
    classificationLayer];

options = trainingOptions('sgdm', ...
           'LearnRateSchedule', 'piecewise', ...
           'LearnRateDropFactor', 0.2, ... 
           'LearnRateDropPeriod', 5, ... 
           'ExecutionEnvironment','gpu',...
           'MaxEpochs', 20, ... 
           'InitialLearnRate',0.002, ...
           'MiniBatchSize', 32);

net = trainNetwork(trainData,layers,options);

predictedLabels = classify(net,valData);
valLabels = valData.Labels;

accuracy = sum(predictedLabels == valLabels)/numel(valLabels)
save('compltnet.mat','trainData','valData','net')
