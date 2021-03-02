my_gt = 'C:\Users\max\Documents\MATLAB\Examples\R2020a\deeplearning_shared\TrainAndDeployFullyConvolutionalNetworksExample\freiburg_forest_annotated\train\GT_color\*.png'
train_data = 'C:\Users\max\Documents\MATLAB\Examples\R2020a\deeplearning_shared\TrainAndDeployFullyConvolutionalNetworksExample\freiburg_forest_annotated\train\rgb\*.jpg'
test_data = 'C:\Users\max\Documents\MATLAB\Examples\R2020a\deeplearning_shared\TrainAndDeployFullyConvolutionalNetworksExample\freiburg_forest_annotated\test\rgb\*.jpg'

out_label = 'C:\Users\max\Documents\MATLAB\Examples\R2020a\deeplearning_shared\TrainAndDeployFullyConvolutionalNetworksExample\resize_labels'
out_train = 'C:\Users\max\Documents\MATLAB\Examples\R2020a\deeplearning_shared\TrainAndDeployFullyConvolutionalNetworksExample\resize_train'
out_test = 'C:\Users\max\Documents\MATLAB\Examples\R2020a\deeplearning_shared\TrainAndDeployFullyConvolutionalNetworksExample\resize_test'


srcFiles = dir(my_gt);

for i = 1 : length(srcFiles)
filename = strcat('C:\Users\max\Documents\MATLAB\Examples\R2020a\deeplearning_shared\TrainAndDeployFullyConvolutionalNetworksExample\freiburg_forest_annotated\train\GT_color\',srcFiles(i).name);
im = imread(filename);
k=imresize(im,[720 960]);
newfilename=strcat('C:\Users\max\Documents\MATLAB\Examples\R2020a\deeplearning_shared\TrainAndDeployFullyConvolutionalNetworksExample\resize_labels\',srcFiles(i).name);
imwrite(k,newfilename,'jpg');
end