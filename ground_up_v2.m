%% network creation

inputSize = [32 32 3];
imgLayer = imageInputLayer(inputSize)

filterSize = 3;
numFilters = 32;
conv = convolution2dLayer(filterSize,numFilters,'Padding',1);
relu = reluLayer();

poolSize = 2;
maxPoolDownsample2x = maxPooling2dLayer(poolSize,'Stride',2);

downsamplingLayers = [
    conv
    relu
    maxPoolDownsample2x
    conv
    relu
    maxPoolDownsample2x
    ]


filterSize = 4;
transposedConvUpsample2x = transposedConv2dLayer(4,numFilters,'Stride',2,'Cropping',1);


upsamplingLayers = [
    transposedConvUpsample2x
    relu
    transposedConvUpsample2x
    relu
    ]

numClasses = 3;
conv1x1 = convolution2dLayer(1,numClasses);

finalLayers = [
    conv1x1
    softmaxLayer()
    pixelClassificationLayer()
    ]

net = [
    imgLayer    
    downsamplingLayers
    upsamplingLayers
    finalLayers
    ]



my_gt = 'C:\Users\max\Documents\MATLAB\vipr\resize_labels\'
train_data = 'C:\Users\max\Documents\MATLAB\vipr\resize_train/'

labelDir = 'C:\Users\max\Documents\MATLAB\vipr\resize_labels\'
imageDir = 'C:\Users\max\Documents\MATLAB\vipr\resize_train/'

classNames = [ ...
    "Road", ...
    "Grass", ...
    "Vegetation", ...
    "Tree", ...
    "Sky", ...
    "Obstacle"]
    


labelIDs = [ ...

    [
    170 170 170; ... % "Road"
    ]
    
    % "Grass"mkdir gt_resized
    [
    0 255 0; ... % "Grass"
    ]
    % Vegetation
    [
    102 102 51; ... % "Vegetation"
    ]
    
    % Treemkdir gt_resized
    [
    0 60 0; ... % "Tree"
    ]
    
    % sky
    [
    0 120 255
    ]
    % Obstaclemkdir gt_resized
    [
    0 0 0; ... % "Obstacle"
    ]
];

imds = imageDatastore(train_data);
countEachLabel(imds)
%I = readimage(imds,10)
%B = labeloverlay(I,C,'Colormap',cmap,'Transparency',0.4);
%imshow(B)
%pixelLabelColorbar(cmap, classes);
%I = histeq(I);
%imshow(I)






numClasses = numel(classNames);
imageSize = [720 960 3];
labelDir = fullfile(my_gt);
pxds = pixelLabelDatastore(labelDir,classNames,labelIDs);


cds = combine(imds,pxds)



tbl = countEachLabel(pxds)


if(tbl.ImagePixelCount(3)==0)
   tbl.ImagePixelCount(3)=1;
end
if(tbl.PixelCount(3)==0)
    tbl.PixelCount(3)=1;
end


[imdsTrain, imdsVal, imdsTest, pxdsTrain, pxdsVal, pxdsTest] =parseID(imds,pxds);
numTrainingImages = numel(imdsTrain.Files)
numValImages = numel(imdsVal.Files)
numTestingImages = numel(imdsTest.Files)


% Specify the network image size. This is typically the same as the traing image sizes.


% Specify the number of classes.
numClasses = numel(classNames);

% Create DeepLab v3+.
lgraph = deeplabv3plusLayers(imageSize, numClasses, "resnet18");

imageFreq = tbl.PixelCount ./ tbl.ImagePixelCount;
classWeights = median(imageFreq) ./ imageFreq

pxLayer = pixelClassificationLayer('Name','labels','Classes',tbl.Name,'ClassWeights',classWeights);
lgraph = replaceLayer(lgraph,"classification",pxLayer);

% Define validation data.
pximdsVal = pixelLabelImageDatastore(imdsVal,pxdsVal);

augmenter = imageDataAugmenter('RandXReflection',true,...
    'RandXTranslation',[-10 10],'RandYTranslation',[-10 10]);

pximds = pixelLabelImageDatastore(imdsTrain,pxdsTrain, ...
    'DataAugmentation',augmenter);


%rcds = read(cds)


% Define training options. 
options = trainingOptions('sgdm', ...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropPeriod',10,...
    'LearnRateDropFactor',0.3,...
    'Momentum',0.7, ...
    'InitialLearnRate',1e-2, ...
    'L2Regularization',0.005, ...
    'ValidationData',pximdsVal,...
    'MaxEpochs',35, ...  
    'MiniBatchSize',8, ...
    'Shuffle','every-epoch', ...
    'CheckpointPath', tempdir, ...
    'VerboseFrequency',2,...
    'Plots','training-progress',...
    'ValidationPatience', 4, ...
    'ExecutionEnvironment','gpu');




%[net, info] = trainNetwork(pximds,lgraph,options);

%% Forest training
I = readimage(imdsTest,5);
C = semanticseg(I, net);
%se = offsetstrel('ball',5,5);
se = strel('rectangle',[40 30]);

cmap = camvidColorMap2;
B = labeloverlay(I,C,'Colormap',cmap,'Transparency',0.5);
%B = imerode(B,se);
%B = imerode(B,se);
%B = imdilate(B,se);
%B = imdilate(B,se);
%B_g = im2gray(B)
%B_g = wiener2(B_g,[5,5]);
%B = cat(3, B_g, B_g, B_g)
imshow(B)
%pixelLabelColorbar(cmap, classes);