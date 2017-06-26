 clear all
close all
clc
load Trainnumbers.mat
load Test_numbers_HW1.mat

trainimages = loadMNISTImages('train-images.idx3-ubyte');
trainlabels = loadMNISTLabels('train-labels.idx1-ubyte');
testimages=loadMNISTImages('t10k-images.idx3-ubyte');
testlabels=loadMNISTLabels('t10k-labels.idx1-ubyte');


rng('shuffle')
%% =========== Task 3: Classical Classifiers(Bayesian classifiers)=============
 %
 %
 %
 %
 
 fprintf('Classical Classifiers(Bayesian classifiers) ...\n')
 
Ncomponents_PCA=75; %numero de dimensiones con las que vamos a quedarnos.
  for k=1:length(Trainnumbers.label)% ejemplo
            % k

            digito=zeros(28,28);
            for i=1:28
                for j=1:28
%                     digito(i,j)=Trainnumbers.image((i-1)*28+j,k);
%                     digitot(i,j)=Test_numbers.image((i-1)*28+j,k);
                    digito(i,j)=trainimages((i-1)*28+j,k);
                    digitot(i,j)=testimages((i-1)*28+j,k);
                end
            end
            
            
         
            X_Row = reshape(digito,1,[]);
%            [X_norm, mu, sigma]=zscore(X_Row);
            X_norm=X_Row;
            
            X_Rowt = reshape(digitot,1,[]);
%             [X_normt, mu, sigma]=zscore(X_Rowt);
            X_normt=X_Rowt;
            
           imagen_vector{k}=X_norm;
%            imagen_label{k}=Trainnumbers.label(k);
         imagen_label{k}=trainlabels(k);
           
           imagen_vectort{k}=X_normt;
   
    end
%


Inputffn=cell2mat(imagen_vector');
Inputffn=Inputffn';
Outputffn=cell2mat(imagen_label);
Outputffn=Outputffn;
%
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


%
[trainInd,~,testInd] = dividerand([pvalor;clase],0.6,0,0.4);
trainvalor=trainInd(1:end-1,:);
trainclase=trainInd(end,:);
testvalor=testInd(1:end-1,:);
testclase=testInd(end,:);
%
[bayclassq, err_qua, posteriorq] = classify(testvalor',trainvalor', trainclase, 'quadratic','empirical');
% 
% Comparación
% err_qua 
acierto_bay_qua=(1-length(find(bayclassq'~=testclase))/length(testclase))*100;
% De los  datos clasificados, 6.48% están mal (con frontera cuadrática) 

%

[bayclassl, err_lin, posteriorl] = classify(testvalor',trainvalor', trainclase, 'linear','empirical');

% 
% Comparación
% err_lin 
acierto_bay_lin=100-length (find(bayclassl'~=testclase))/length(testclase)*100;
% De los  datos clasificados, 14% están mal (con frontera lineal)


% % % %% Proportional probability of every class
% % % % Auxiliary variables
% % % contp = zeros(1,10);
% % % 
% % % % Loop that runs along the vector p.clase for knowing which is the column of
% % % % each group and classifying the data
% % % for i=1:length(clase)
% % %     
% % %     if clase(i) == 0
% % %         contp(1) = contp(1) + 1;
% % %     elseif clase(i) == 1
% % %         contp(2) = contp(2) + 1;
% % %     elseif clase(i) == 2
% % %         contp(3) = contp(3) + 1;
% % %     elseif clase(i) == 3
% % %         contp(4) = contp(4) + 1;
% % % 	elseif clase(i) == 4
% % %         contp(5) = contp(5) + 1;
% % % 	 elseif clase(i) ==5
% % %         contp(6) = contp(6) + 1;
% % %     elseif clase(i) == 6
% % %         contp(7) = contp(7) + 1;
% % % 	elseif clase(i) == 7
% % %         contp(8) = contp(8) + 1;
% % % 	elseif clase(i) == 8
% % %         contp(9) = contp(9) + 1;
% % %     elseif clase(i) == 9
% % %         contp(10) = contp(10) + 1;
% % %     end
% % % end
% % % 
% % % % Probability
% % % for i=1:10
% % % prior(i) = (contp(i))/length(clase) ;
% % % end
% % % %%
% % % % Classification with p training data.
% % % [bayclasslp, errlp, posteriorl12] = classify(pvalor',pvalor', clase, 'linear',[prior]);
% % % [bayclassqp, errqp, posteriorq12] = classify(pvalor',pvalor', clase, 'quadratic',[prior]);
% % %  
% % % % Obtain erroneously classified items
% % % errlp          % 0.0833 (linear)
% % %    % [0.3615 0.6385] Therefore, data 1 is more likely to be of class 2
% % % errqp          % 0.0833 (quadratic)
% % %   % [0.2605 0.7395] Therefore, data 1 is more likely to be of class 2
% % % 
% % % num_errors_bay_lin_post=length (find(bayclasslp'~=clase))/length(clase)*100
% % % num_errors_bay_qua_post=length (find(bayclassqp'~=clase))/length(clase)*100
%%
close all
[bayclassqt, err_quat, posteriorqt] = classify(pvalort(:,1:10000)',trainvalor', trainclase, 'diagLinear');

[bayclassqt2, err_quat, posteriorqt] = classify(pvalort(:,1:10000)',trainvalor', trainclase,'quadratic','empirical');


figure;hist(bayclassq)
figure;hist(bayclassqt)  
figure;hist(bayclassqt2) 

acierto_bay_lin

acierto_bay_qua
%
% %%
%  fprintf('Program paused. Press enter to continue.\n');
% pause;
%   
   