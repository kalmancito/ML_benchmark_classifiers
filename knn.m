
clear all
close all
clc
load Trainnumbers.mat
load Test_numbers_HW1.mat

% trainimages = loadMNISTImages('train-images.idx3-ubyte');
% trainlabels = loadMNISTLabels('train-labels.idx1-ubyte');
% testimages=loadMNISTImages('t10k-images.idx3-ubyte');
% testlabels=loadMNISTLabels('t10k-labels.idx1-ubyte');
rng('shuffle')
%% =========== Task 2: Classical Classifiers(K-nn)=============
 %
 %
 %
 %
 
 fprintf('Classical Classifiers(K-nn) ...\n')
 %
%  for cont=1:2
 k_choosed=5;% 5
 Ncomponents_PCA=40;%50;%73 %numero de dimensiones con las que vamos a quedarnos.

 
 %
 
  for k=1:length(Trainnumbers.label)% ejemplo
            % k

            digito=zeros(28,28);
            for i=1:28
                for j=1:28
                   digito(i,j)=Trainnumbers.image((i-1)*28+j,k);
                  digitot(i,j)=Test_numbers.image((i-1)*28+j,k);
%                     digito(i,j)=trainimages((i-1)*28+j,k);
%                     digitot(i,j)=testimages((i-1)*28+j,k);
                end
            end
            
            
            X_Row = reshape(digito,1,[]);
            [X_norm, mu, sigma]=zscore(X_Row);
%             X_norm=X_Row;
            
            X_Rowt = reshape(digitot,1,[]);
            [X_normt, mu, sigma]=zscore(X_Rowt);
%             X_normt=X_Rowt;
            
           imagen_vector{k}=X_norm;
           imagen_label{k}=Trainnumbers.label(k);
%            imagen_label{k}=trainlabels(k);
           
           imagen_vectort{k}=X_normt;
   
    end
%


Inputffn=cell2mat(imagen_vector');
Inputffn=Inputffn';
Outputffn=cell2mat(imagen_label);
Outputffn=Outputffn;
%
Inputffnt=cell2mat(imagen_vectort');
Inputffnt=Inputffnt';
%  [X_normt, mu, sigma]=zscore(Inputffnt');
  X_norm=Inputffn'; 
%
  X_normt=Inputffnt';       

            %  Run PCA   ---> [U, S] = pca(X_norm);
C_X=cov(X_norm);
C_Xt=cov(X_normt);

[U,S] = eig(C_X);
[Ut,S] = eig(C_Xt);


D=length(U);
Dt=length(Ut);

% [residual, preconstructed]=pcares(t_normalized,Ncomponents_PCA);


for i=1:Ncomponents_PCA
    
    transf_mat(i,:)=U(:,D+1-i)';
    transf_matt(i,:)=Ut(:,Dt+1-i)';
end
%
size(transf_mat);
size(X_norm);
%
error=S(1,1);
reducedData =  transf_mat*X_norm';
reducedDatat =  transf_mat*X_normt';
size(reducedData);

%

X_rec=reducedData'*transf_mat;
%



pvalor=reducedData;

pvalort=reducedDatat;
clase=Outputffn;
  
% %
% [m,n]=size(Trainnumbers.image);
% ind_rand=randperm(n);
% trainvalor=reducedData(:,ind_rand(1:8000));
% trainclase=Trainnumbers.label(ind_rand(1:8000));
% testvalor=reducedData(:,ind_rand(8001:end));
% testclase=Trainnumbers.label(ind_rand(8001:end));
% 
% 
% 
% nnclass=knnclassify(testvalor', trainvalor', trainclase,k);
% 
% no_errors_nn=length(find(nnclass'~=testclase));
% 
% Acierto=100*(length(testclase)-no_errors_nn)/length(testclase)
% error('kk')
%

[trainInd,~,testInd] = dividerand([pvalor;clase],0.85,0,0.15);
trainvalor=trainInd(1:end-1,:);
trainclase=trainInd(end,:);
testvalor=testInd(1:end-1,:);
testclase=testInd(end,:);
%
% k-nn classifier

% Clasificamos t

% nnclass = knnclassify(testvalor', trainvalor', trainclase,k_choosed,'euclidean'); 
%%
chiSqrDist = @(x,Z,wt)sqrt((bsxfun(@minus,x,Z).^2)*wt);

w = ones(1,Ncomponents_PCA);

Mdl = fitcknn(trainvalor',trainclase',...
     'Distance','minkowski','Exponent',3,...
    'NumNeighbors',k_choosed);


Mdl2 =  fitcknn(trainvalor',trainclase',...
     'Distance','euclidean',...
    'NumNeighbors',k_choosed);

%%
[nnclass,score,cost] = predict(Mdl2,testvalor');
% Comparación
aciertos_knn= (1-length(find(nnclass'~=testclase))/length(testclase))*100

% [nnclass,score,cost] = predict(Mdl2,testvalor');
% aciertos_knn= (1-length(find(nnclass'~=testclase))/length(testclase))*100

%%
 [nnclasst,score,cost] = predict(Mdl2,pvalort(:,:)');

% 
% %
fprintf('Program paused. Press enter to continue.\n');
% pause;
%  end
%%
figure; hist(nnclass)
figure; hist(nnclasst)
mean(aciertos_knn)