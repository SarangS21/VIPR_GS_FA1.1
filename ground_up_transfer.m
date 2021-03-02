
%%City Dataset

pretrainedURL = 'https://www.mathworks.com/supportfiles/vision/data/deeplabv3plusResnet18CamVid.mat';
pretrainedFolder = fullfile(tempdir,'pretrainedNetwork');
pretrainedNetwork = fullfile(pretrainedFolder,'deeplabv3plusResnet18CamVid.mat'); 
if ~exist(pretrainedNetwork,'file')
    mkdir(pretrainedFolder);
    disp('Downloading pretrained network (58 MB)...');
    websave(pretrainedNetwork,pretrainedURL);
end

imageURL = 'http://web4.cs.ucl.ac.uk/staff/g.brostow/MotionSegRecData/files/701_StillsRaw_full.zip';
labelURL = 'http://web4.cs.ucl.ac.uk/staff/g.brostow/MotionSegRecData/data/LabeledApproved_full.zip';
 
outputFolder = fullfile(tempdir,'CamVid'); 
labelsZip = fullfile(outputFolder,'labels.zip');
imagesZip = fullfile(outputFolder,'images.zip');

if ~exist(labelsZip, 'file') || ~exist(imagesZip,'file')   
    mkdir(outputFolder)
       
    disp('Downloading 16 MB CamVid dataset labels...'); 
    websave(labelsZip, labelURL);
    unzip(labelsZip, fullfile(outputFolder,'labels'));
    
    disp('Downloading 557 MB CamVid dataset images...');  
    websave(imagesZip, imageURL);       
    unzip(imagesZip, fullfile(outputFolder,'images'));    
end


imgDir = fullfile(outputFolder,'images','701_StillsRaw_full');
imds = imageDatastore(imgDir);
I = readimage(imds,559);
I = histeq(I);
imshow(I)



classes = [
    "Sky"
    "Building"
    "Pole"
    "Road"
    "Pavement"
    "Tree"
    "SignSymbol"
    "Fence"
    "Car"
    "Pedestrian"
    "Bicyclist"
    ];


params = load("C:\Users\max\Documents\MATLAB\vipr\params_2021_01_28__22_40_07.mat");
%%Network Creation

inputSize = [720 960 3];
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

numClasses = 11;
conv1x1 = convolution2dLayer(1,numClasses);

weight = [0.318184709354742;0.208197860785155;5.09236733293851;0.174381825257403;0.710338097812948;0.417518560687874;4.53707481548293;1.83864826191456;1;6.60587857315587;5.11333841605959]

finalLayers = [
    conv1x1
    relu
    %fullyConnectedLayer(11)
    crop2dLayer('centercrop')
    softmaxLayer()
    pixelClassificationLayer("Name","labels","ClassWeights",params.labels.ClassWeights,"Classes",params.labels.Classes)
    
    %pixelClassificationLayer()
]
    
net = [
    imgLayer    
    downsamplingLayers
    upsamplingLayers
    finalLayers
    ]

%net = connectLayers(net,"imageinput","crop/ref");


%%City Dataset

pretrainedURL = 'https://www.mathworks.com/supportfiles/vision/data/deeplabv3plusResnet18CamVid.mat';
pretrainedFolder = fullfile(tempdir,'pretrainedNetwork');
pretrainedNetwork = fullfile(pretrainedFolder,'deeplabv3plusResnet18CamVid.mat'); 
if ~exist(pretrainedNetwork,'file')
    mkdir(pretrainedFolder);
    disp('Downloading pretrained network (58 MB)...');
    websave(pretrainedNetwork,pretrainedURL);
end

imageURL = 'http://web4.cs.ucl.ac.uk/staff/g.brostow/MotionSegRecData/files/701_StillsRaw_full.zip';
labelURL = 'http://web4.cs.ucl.ac.uk/staff/g.brostow/MotionSegRecData/data/LabeledApproved_full.zip';
 
outputFolder = fullfile(tempdir,'CamVid'); 
labelsZip = fullfile(outputFolder,'labels.zip');
imagesZip = fullfile(outputFolder,'images.zip');

if ~exist(labelsZip, 'file') || ~exist(imagesZip,'file')   
    mkdir(outputFolder)
       
    disp('Downloading 16 MB CamVid dataset labels...'); 
    websave(labelsZip, labelURL);
    unzip(labelsZip, fullfile(outputFolder,'labels'));
    
    disp('Downloading 557 MB CamVid dataset images...');  
    websave(imagesZip, imageURL);       
    unzip(imagesZip, fullfile(outputFolder,'images'));    
end


imgDir = fullfile(outputFolder,'images','701_StillsRaw_full');
imds = imageDatastore(imgDir);
I = readimage(imds,559);
I = histeq(I);
imshow(I)



classes = [
    "Sky"
    "Building"
    "Pole"
    "Road"
    "Pavement"
    "Tree"
    "SignSymbol"
    "Fence"
    "Car"
    "Pedestrian"
    "Bicyclist"
    ];


labelIDs = camvidPixelLabelIDs();
labelDir = fullfile(outputFolder,'labels');
pxds = pixelLabelDatastore(labelDir,classes,labelIDs);

C = readimage(pxds,559);
cmap = camvidColorMap;
B = labeloverlay(I,C,'ColorMap',cmap);
imshow(B)
pixelLabelColorbar(cmap,classes);
tbl = countEachLabel(pxds)
frequency = tbl.PixelCount/sum(tbl.PixelCount);

%%
[imdsTrain, imdsVal, imdsTest, pxdsTrain, pxdsVal, pxdsTest] = partitionCamVidData(imds,pxds);
numTrainingImages = numel(imdsTrain.Files)

numValImages = numel(imdsVal.Files)
numTestingImages = numel(imdsTest.Files)
% Specify the network image size. This is typically the same as the traing image sizes.

imageSize = [720 960 3];
% Specify the number of classes.
numClasses = numel(classes);

%lgraph = deeplabv3plusLayers(imageSize, numClasses, "net");

imageFreq = tbl.PixelCount ./ tbl.ImagePixelCount;
classWeights = median(imageFreq) ./ imageFreq

pxLayer = pixelClassificationLayer('Name','labels','Classes',tbl.Name,'ClassWeights',classWeights);
%lgraph = replaceLayer(lgraph,"classification",pxLayer);

pximdsVal = pixelLabelImageDatastore(imdsVal,pxdsVal);


% Define training options. 
options = trainingOptions('sgdm', ...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropPeriod',50,...
    'LearnRateDropFactor',0.1,...
    'Momentum',0.7, ...
    'InitialLearnRate',1e-2, ...
    'L2Regularization',0.005, ...
    'ValidationData',pximdsVal,...
    'MaxEpochs',10, ...  
    'MiniBatchSize',2, ...
    'Shuffle','every-epoch', ...
    'CheckpointPath', tempdir, ...
    'VerboseFrequency',2,...
    'Plots','training-progress',...
    'ValidationPatience', 4, ...
    'ExecutionEnvironment','gpu');


augmenter = imageDataAugmenter('RandXReflection',true,...
    'RandXTranslation',[-10 10],'RandYTranslation',[-10 10]);

pximds = pixelLabelImageDatastore(imdsTrain,pxdsTrain, ...
    'DataAugmentation',augmenter);

 
[netK, info] = trainNetwork(pximds,lgraph_4,options);



%% Validation Images


I = readimage(imdsTest,10);
C = semanticseg(I, netK);
%se = offsetstrel('ball',5,5);
se = strel('rectangle',[40 30]);


B = labeloverlay(I,C,'Colormap',cmap,'Transparency',0.5);
%B = imerode(B,se);
%B = imerode(B,se);
%B = imdilate(B,se);
%B = imdilate(B,se);
%B_g = im2gray(B)
%B_g = wiener2(B_g,[5,5]);
%B = cat(3, B_g, B_g, B_g)
imshow(B)
pixelLabelColorbar(cmap, classes);


%% transfer weights
weight = netK.Layers(16,1)
weights = weight.ClassWeights
weights([2,3,5,7,8,9,10,11]) = []
weights([4,5]) = weights([3])
weights([6]) = mean(weights')
%% Forest Dataset and Learning

my_gt = 'C:\Users\max\Documents\MATLAB\vipr\resize_labels\'
train_data = 'C:\Users\max\Documents\MATLAB\vipr\resize_train/'

labelDir = 'C:\Users\max\Documents\MATLAB\vipr\resize_labels\'
imageDir = 'C:\Users\max\Documents\MATLAB\vipr\resize_train/'

classNames = [ ...
    "Road", ...
    "Grass", .../
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
%lgraph = deeplabv3plusLayers(imageSize, numClasses, "resnet18");

imageFreq = tbl.PixelCount ./ tbl.ImagePixelCount;
classWeights = weights

pxLayer = pixelClassificationLayer('Name','labels','Classes',tbl.Name,'ClassWeights',classWeights);
%lgraph = replaceLayer(lgraph,"classification",pxLayer);

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




[netF, info] = trainNetwork(pximds,lgraph_4,options);

%% Validation Images


I = readimage(imdsTest,5);
C = semanticseg(I, netF);
cmap2 = camvidColorMap2;
%se = offsetstrel('ball',5,5);
se = strel('rectangle',[40 30]);


B = labeloverlay(I,C,'Colormap',cmap2,'Transparency',0.5);
%B = imerode(B,se);
%B = imerode(B,se);
%B = imdilate(B,se);
%B = imdilate(B,se);
%B_g = im2gray(B)
%B_g = wiener2(B_g,[5,5]);
%B = cat(3, B_g, B_g, B_g)
imshow(B)
pixelLabelColorbar(cmap2, classNames);
%%
res = resnet18