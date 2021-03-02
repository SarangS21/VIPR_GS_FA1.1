%data load

dataDir = fullfile(toolboxdir('vision'),'visiondata');
imDir = fullfile(dataDir,'building');
pxDir = fullfile(dataDir,'buildingPixelLabels');

imds = imageDatastore(imDir);

I = readimage(imds,1);
figure
imshow(I)

classNames = ["sky" "grass" "building" "sidewalk"];

pixelLabelID = [1 2 3 4];

pxds = pixelLabelDatastore(pxDir,classNames,pixelLabelID);

C = readimage(pxds,1);
C(5,5)

B = labeloverlay(I,C);
figure
imshow(B)

buildingMask = C == 'building';

figure
imshowpair(I, buildingMask,'montage')

%%%%%%%%%%%%%%%%%%%%%%%

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
%%
%%training

dataSetDir = fullfile(toolboxdir('vision'),'visiondata','triangleImages');
imageDir = fullfile(dataSetDir,'trainingImages');
labelDir = fullfile(dataSetDir,'trainingLabels');

imds = imageDatastore(imageDir);

classNames = ["triangle","background"];
labelIDs   = [255 0];
pxds = pixelLabelDatastore(labelDir,classNames,labelIDs);

I = read(imds);
C = read(pxds);

I = imresize(I,5);
L = imresize(uint8(C{1}),5);
imshowpair(I,L,'montage')


numFilters = 64;
filterSize = 3;
numClasses = 2;
layers = [
    imageInputLayer([32 32 1])
    convolution2dLayer(filterSize,numFilters,'Padding',1)
    reluLayer()
    maxPooling2dLayer(2,'Stride',2)
    convolution2dLayer(filterSize,numFilters,'Padding',1)
    reluLayer()
    transposedConv2dLayer(4,numFilters,'Stride',2,'Cropping',1);
    convolution2dLayer(1,numClasses);
    softmaxLayer()
    pixelClassificationLayer()
    ]


opts = trainingOptions('sgdm', ...
    'InitialLearnRate',1e-3, ...
    'MaxEpochs',100, ...
    'MiniBatchSize',64);

trainingData = pixelLabelImageDatastore(imds,pxds);

net = trainNetwork(trainingData,layers,opts);

%% test image

testImage = imread('triangleTest.jpg');
imshow(testImage)

C = semanticseg(testImage,net);
B = labeloverlay(testImage,C);
imshow(B)


%% train improvements

tbl = countEachLabel(trainingData)

totalNumberOfPixels = sum(tbl.PixelCount);
frequency = tbl.PixelCount / totalNumberOfPixels;
classWeights = 1./frequency


layers(end) = pixelClassificationLayer('Classes',tbl.Name,'ClassWeights',classWeights);

net = trainNetwork(trainingData,layers,opts);

%% test image

C = semanticseg(testImage,net);
B = labeloverlay(testImage,C);
imshow(B)



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
my_gt = 'C:\Users\max\Documents\MATLAB\vipr\resize_labels\'
train_data = 'C:\Users\max\Documents\MATLAB\vipr\resize_train/'



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



%tbl = countEachLabel(pxds)


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




[net, info] = trainNetwork(pximds,lgraph,options);

%%
I = readimage(imdsTest,5);
C = semanticseg(I, net);
%se = offsetstrel('ball',5,5);
se = strel('rectangle',[40 30]);

cmap = camvidColorMap;
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

%%
weight = net.Layers(100,1)
weights = weight.ClassWeights

%Wtest=resnet18(weights,'none')

lgraph = resnet18('weights','none')

[netF, info] = trainNetwork(pximds,lgraph,options);