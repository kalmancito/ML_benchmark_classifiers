load('A0016255_knn.mat')
knnc=class;
load('A0016255_bay.mat')
bayc=class;
load('A0016255_mlp.mat')
mlpc=class;
load('A0016255_som.mat')
somc=class;
load('clase_comparar.mat')
kkc=tclass;

mlpe=(1-length(find(mlpc~=kkc'))/length(kkc))*100
knne=(1-length(find(kkc'~=knnc'))/length(kkc))*100
baye=(1-length(find(bayc'~=kkc'))/length(kkc))*100
some=(1-length(find(somc~=kkc'))/length(kkc))*100
% mlpe=(1-length(find(mlpc~=knnc'))/length(knnc'))*100
%%
trainimages = loadMNISTImages('train-images.idx3-ubyte');
trainlabels = loadMNISTLabels('train-labels.idx1-ubyte');