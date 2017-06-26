
clear all
close all
clc
%%
trainimages = loadMNISTImages('train-images.idx3-ubyte');
trainlabels = loadMNISTLabels('train-labels.idx1-ubyte');
testimages=loadMNISTImages('t10k-images.idx3-ubyte');
testlabels=loadMNISTLabels('t10k-labels.idx1-ubyte');
load('Test_numbers_HW1')
%% =========== Task 2: Classical Classifiers(K-nn)=============
 %
 %
 %
 %
 
 fprintf('Classical Classifiers(K-nn) ...\n')
 %
 k_choosed=3


 Ncomponents_PCA=50; %numero de dimensiones con las que vamos a quedarnos.

 
 %
 
  for k=1:length(trainlabels)% ejemplo
            % k

            digito=zeros(28,28);
            for i=1:28
                for j=1:28
                    digito(i,j)=trainimages((i-1)*28+j,k);
%                     digitot(i,j)=testimages((i-1)*28+j,k);
                end
            end
            
            
            X_Row = reshape(digito,1,[]);
            [X_norm, mu, sigma]=zscore(X_Row);
% X_norm=X_Row;
%             X_Rowt = reshape(digitot,1,[]);
%             [X_normt, mu, sigma]=zscore(X_Rowt);

           imagen_vector{k}=X_norm;
           imagen_label{k}=trainlabels(k);
           
%            imagen_vectort{k}=X_normt;
   
    end
%
%%

  for k=1:length(Test_numbers.image)% ejemplo
            % k

            digito=zeros(28,28);
            for i=1:28
                for j=1:28
%                     digito(i,j)=trainimages((i-1)*28+j,k);
                    digitot(i,j)=Test_numbers.image((i-1)*28+j,k);
                end
            end
            
            
%             X_Row = reshape(digito,1,[]);
%             [X_norm, mu, sigma]=zscore(X_Row);

            X_Rowt = reshape(digitot,1,[]);
            [X_normt, mu, sigma]=zscore(X_Rowt);
% X_normt=X_Rowt;
%            imagen_vector{k}=X_norm;
%            imagen_label{k}=trainlabels(k);
           
           imagen_vectort{k}=X_normt;
   
    end
%%

Inputffn=cell2mat(imagen_vector');
Inputffn=Inputffn';
Outputffn=cell2mat(imagen_label);
Outputffn=Outputffn;
%
Inputffnt=cell2mat(imagen_vectort');
Inputffnt=Inputffnt';
%  [X_normt, mu, sigma]=zscore(Inputffnt');

%%


%%
  X_norm=Inputffn'; 
  X_normt=Inputffnt';       
%%
            %  Run PCA   ---> [U, S] = pca(X_norm);
C_X=cov(X_norm);
C_Xt=cov(X_normt);

[U,S] = eig(C_X);
[Ut,St] = eig(C_Xt);

D=length(U);

% [residual, preconstructed]=pcares(t_normalized,Ncomponents_PCA);


for i=1:Ncomponents_PCA
    transf_matt(i,:)=Ut(:,D+1-i)';
    transf_mat(i,:)=U(:,D+1-i)';
end
%
size(transf_mat);
size(X_norm);
%
error=S(1,1);
reducedData =  transf_mat*X_norm';
reducedDatat =  transf_matt*X_normt';

% X_rec=reducedData'*transf_mat;
%%



pvalor=reducedData;
clase=Outputffn;
pvalor_pascual=reducedDatat;
[trainInd,~,testInd] = dividerand([pvalor;clase],0.9,0,0.1);
trainvalor=trainInd(1:end-1,:);
trainclase=trainInd(end,:);
testvalor=testInd(1:end-1,:);
testclase=testInd(end,:);
%% 

% Mdl = fitcknn(trainvalor',trainclase',...
%      'Distance','minkowski','Exponent',3,...
%     'NumNeighbors',k_choosed);

Mdl = fitcknn(trainvalor',trainclase',...
     'Distance','euclidean',...
    'NumNeighbors',k_choosed);



[tclass,score,cost] = predict(Mdl,testvalor');
[tclass_pacual,score,cost] = predict(Mdl,pvalor_pascual');
%% Comparación
errknn=(1-length(find(tclass~=testclase'))/length(testclase))*100
% 
% %
% fprintf('Program paused. Press enter to continue.\n');
% pause;
for k=1:length(testclase)
           trainclase_confu(k,:)=zeros(10,1);
           
           trainclase_confu(k,testclase(k)+1)=1;
           
           tclass_confu(k,:)=zeros(10,1);
           
           tclass_confu(k,tclass(k)+1)=1;
end
%%
% plotconfusion(trainclase_confu',tclass_confu')
figure;hist(tclass_pacual)
figure;hist(tclass)

k_choosed
% err_parak(k_choosed)=mean(errknn)
err_parak=mean(errknn)

%  %%
