
resnet18();

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

lgraph = deeplabv3plusLayers(imageSize, numClasses, "resnet18");

imageFreq = tbl.PixelCount ./ tbl.ImagePixelCount;
classWeights = median(imageFreq) ./ imageFreq

pxLayer = pixelClassificationLayer('Name','labels','Classes',tbl.Name,'ClassWeights',classWeights);
lgraph = replaceLayer(lgraph,"classification",pxLayer);

pximdsVal = pixelLabelImageDatastore(imdsVal,pxdsVal);


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


augmenter = imageDataAugmenter('RandXReflection',true,...
    'RandXTranslation',[-10 10],'RandYTranslation',[-10 10]);

pximds = pixelLabelImageDatastore(imdsTrain,pxdsTrain, ...
    'DataAugmentation',augmenter);


doTraining = false;
if doTraining    
    [net, info] = trainNetwork(pximds,lgraph,options);
else
    data = load(pretrainedNetwork); 
    netRes = data.net;
end

%%
I = readimage(imdsTest,15);
C = semanticseg(I, net);

B = labeloverlay(I,C,'Colormap',cmap,'Transparency',0.4);
imshow(B)
pixelLabelColorbar(cmap, classes);


%%

options = trainingOptions('sgdm', ...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropPeriod',10,...
    'LearnRateDropFactor',0.3,...
    'Momentum',0.7, ...
    'InitialLearnRate',1e-2, ...
    'L2Regularization',0.005, ...
    'ValidationData',pximdsVal,...
    'MaxEpochs',35, ...  
    'MiniBatchSize',10, ...
    'Shuffle','every-epoch', ...
    'CheckpointPath', tempdir, ...
    'VerboseFrequency',2,...
    'Plots','training-progress',...
    'ValidationPatience', 4, ...
    'ExecutionEnvironment','gpu');

classes = [
"Road"
"Grass"
"Vegetation"
"Tree"
"Sky"
"Obstacle"
];

    
labelIDs = { ...

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
};



my_gt = 'C:\Users\max\Documents\MATLAB\vipr\resize_labels\'
train_data = 'C:\Users\max\Documents\MATLAB\vipr\resize_train/'

labelIDs = pixID();


imds = imageDatastore(train_data);
countEachLabel(imds)
I = readimage(imds,10)
B = labeloverlay(I,C,'Colormap',cmap,'Transparency',0.4);
imshow(B)
pixelLabelColorbar(cmap, classes);
I = histeq(I);
imshow(I)






numClasses = numel(classes);
imageSize = [720 960 3];
labelDir = fullfile(my_gt);
pxds = pixelLabelDatastore(labelDir,classes,labelIDs);








cds = combine(imds,pxds)



C = readimage(pxds,10);
cmap = camvidColorMap;
B = labeloverlay(I,C,'ColorMap',cmap);
imshow(B)

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
numClasses = numel(classes);

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
%%

doTraining = true;
if doTraining    
    [net, info] = trainNetwork(pximds,lgraph,options);
else
    data = load(pretrainedNetwork); 
    net = data.net;
end



%%

I = readimage(imdsTest,10);
C = semanticseg(I, net);
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
%pixelLabelColorbar(cmap, classes);


