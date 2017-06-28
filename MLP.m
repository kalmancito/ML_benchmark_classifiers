clear all
close all
clc
load Trainnumbers.mat
load Test_numbers_HW1.mat

 %% =========== Task 4a: Neural Classifiers(FFN)=============
 %
 %
 %
 %
 
 fprintf(' Neural Classifiers(FFN)(a) ...\n')
 %%
 
    for k=1:length(Trainnumbers.label)% ejemplo
            % k

            digito=zeros(28,28);
            for i=1:28
                for j=1:28
                    digito(i,j)=Trainnumbers.image((i-1)*28+j,k);
                end
            end
            
            
            X_Row = reshape(digito,1,[]);
            [X_norm, mu, sigma]=zscore(X_Row);


           imagen_vector{k}=X_norm;
           imagen_label{k}=zeros(10,1);
           imagen_label{k}(Trainnumbers.label(k)+1)=1;
    end
%


Inputffn=cell2mat(imagen_vector');
Inputffn=Inputffn';
Outputffn=cell2mat(imagen_label);
Outputffn=Outputffn*1;
%

%  [X_norm, mu, sigma]=zscore(Inputffn');
    X_norm=Inputffn';

            %  Run PCA   ---> [U, S] = pca(X_norm);
C_X=cov(X_norm);

[U,S] = eig(C_X);

D=length(U);

Ncomponents_PCA=100; %numero de dimensiones con las que vamos a quedarnos.
% [residual, preconstructed]=pcares(t_normalized,Ncomponents_PCA);


for i=1:Ncomponents_PCA
    
    transf_mat(i,:)=U(:,D+1-i)';
end
%
size(transf_mat);
size(X_norm);
%
error=S(1,1);
reducedData =  transf_mat*X_norm';
 %%
 %  
neurons=25;
inputs=reducedData;
targets=Outputffn;
%

netpatern = patternnet(neurons);
netpatern.trainParam.showWindow=1; %default is 1)
netpatern.trainFcn = 'trainlm';
netpatern.trainParam.epochs = 10;
netpatern.trainParam.goal  = 0.001;
netpatern.trainParam.show = 100;

% Set up Division of Data for Training, Validation, Testing
netpatern.divideParam.trainRatio = 70/100;
netpatern.divideParam.valRatio = 15/100;
netpatern.divideParam.testRatio = 15/100;
%%

% Train the Network
[netpatern,tr] = train(netpatern,inputs,targets);
%%
% Test the Network
outputs = netpatern(inputs);
errors = gsubtract(targets,outputs);
performance = perform(netpatern,targets,outputs)


% [netpatern,tr] = train(netpatern,inputs,targets);
y = netpatern(inputs);
%%
figure
plotconfusion(targets,y)

fprintf('Program paused. Press enter to continue.\n');
% pause;

 %% =========== Task 4b: Neural Classifiers(FFN)=============
 %
 %
 %
 %
 
 fprintf(' Neural Classifiers(FFN)(b) ...\n')
 %%
 
    for k=1:length(Trainnumbers.label)% ejemplo
            % k

            digito=zeros(28,28);
            for i=1:28
                for j=1:28
                    digito(i,j)=Trainnumbers.image((i-1)*28+j,k);
                    digitot(i,j)=Test_numbers.image((i-1)*28+j,k);
                end
            end
            
            
            X_Row = reshape(digito,1,[]);
            X_Rowt = reshape(digitot,1,[]);
            [X_norm, mu, sigma]=zscore(X_Row);
             [X_normt, mu, sigma]=zscore(X_Rowt);


           imagen_vector{k}=X_norm;
           imagen_label{k}=zeros(10,1);
           imagen_label{k}(Trainnumbers.label(k)+1)=1;
           
           imagen_vectort{k}=X_normt;
    end
%%


Inputffn=cell2mat(imagen_vector');
Inputffn=Inputffn';
Outputffn=cell2mat(imagen_label);
Outputffn=Outputffn*100;

Inputffnt=cell2mat(imagen_vectort');
Inputffnt=Inputffnt';
%

%  [X_norm, mu, sigma]=zscore(Inputffn');
   X_norm=Inputffn';   
   X_normt=Inputffnt';

            %  Run PCA   ---> [U, S] = pca(X_norm);
C_X=cov(X_norm);

[U,S] = eig(C_X);

D=length(U);

Ncomponents_PCA=100; %numero de dimensiones con las que vamos a quedarnos.
% [residual, preconstructed]=pcares(t_normalized,Ncomponents_PCA);


for i=1:Ncomponents_PCA
    
    transf_mat(i,:)=U(:,D+1-i)';
end
%
size(transf_mat);
size(X_norm);
%
error=S(1,1);
reducedData =  transf_mat*X_norm';
reducedDatat =  transf_mat*X_normt';
 %%
 %  
neurons=20; % 10
P=reducedData;
Pt=reducedDatat;
T=Outputffn;

netff = feedforwardnet(neurons);
%     {'tansig' 'purelin'},...
%     'trainlm',...
%     'learngd','mse');
netff.trainParam.showWindow=1; %default is 1)
netff.trainFcn = 'trainlm';
netff.trainParam.epochs = 1000;
netff.trainParam.goal  = 0.001;
netff.trainParam.show = 100;

% Set up Division of Data for Training, Validation, Testing
netff.divideParam.trainRatio = 70/100;
netff.divideParam.valRatio = 15/100;
netff.divideParam.testRatio = 15/100;

[netff,tr] = train(netff,P,T);
%%
% y = netff(P);
y=netpatern(P);
% yt = netff(Pt);
yt=netpatern(Pt);
[~,yt2]=max(yt);
hist(yt2)
figure
plotconfusion(T,y)

fprintf('Program paused. Press enter to continue.\n');
pause;
