# Roundworm-Vitality
Uses transfer learning to train a deep network that can classify images of roundworms as either alive or dead. 

Are the Roundworms Alive or Dead?
This example uses transfer learning to train a deep network that can classify images of roundworms as either alive or dead. (Alive worms are round; dead ones are straight.)

Load AlexNet
anet = alexnet;
layers= anet.Layers;


Create Image DataStore
imloc = 'C:\Users\jabril.jacobs\Documents\MATLAB\deeplearning_course_files\Roundworms\WormImages';
labels = readtable("deeplearning_course_files\Roundworms\WormData.csv");
status = categorical(labels.Status);
imds = imageDatastore(imloc,"Labels",status);
[train,valid,test] = splitEachLabel(imds,0.5,0.35,"randomized");


Augment Image DataStores
% Determine required input size for AlexNet
insize = anet.Layers(1).InputSize;
atrain = augmentedImageDatastore([insize(1) insize(2)],train,"ColorPreprocessing","gray2rgb");
avalid = augmentedImageDatastore([insize(1) insize(2)],valid,"ColorPreprocessing","gray2rgb");
atest = augmentedImageDatastore([insize(1) insize(2)],test,"ColorPreprocessing","gray2rgb");


Augment Layers
fc = fullyConnectedLayer(2);
layers(23) = fc;
layers(end) = classificationLayer;

Train Network
opts = trainingOptions("sgdm","InitialLearnRate",0.001,"ValidationData",avalid,"Momentum",0.5);
[wormnet,info] = trainNetwork(atrain,layers,opts);

Evaluate Network
Find the true and predicted statuses of the test images.
preds = classify(wormnet,atest);
truetest = test.Labels;
numCorrect = preds == truetest;
fracCorrect = nnz(numCorrect)/numel(numCorrect)
confusionchart(truetest,preds);
