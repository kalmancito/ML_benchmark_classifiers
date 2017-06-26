function [acierto_mlp, t_classi, t_train] =mlp_para_test(mlp_neu,Ncomponents_PCA,N_epochs)
clc
load Trainnumbers.mat


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
%             [X_norm, mu, sigma]=zscore(X_Row);
X_norm= X_Row;

           imagen_vector{k}=X_norm;
           imagen_label{k}=zeros(10,1);
           imagen_label{k}(Trainnumbers.label(k)+1)=1;
    end
%


Inputffn=cell2mat(imagen_vector');
Inputffn=Inputffn';
Outputffn=cell2mat(imagen_label);
Outputffn=Outputffn*100;
%

%  [X_norm, mu, sigma]=zscore(Inputffn');
    X_norm=Inputffn';

            %  Run PCA   ---> [U, S] = pca(X_norm);
C_X=cov(X_norm);

[U,S] = eig(C_X);

D=length(U);

% Ncomponents_PCA=100; %numero de dimensiones con las que vamos a quedarnos.
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
neurons=mlp_neu;
inputs=reducedData;
targets=Outputffn;
%

netpatern = patternnet(neurons);
netpatern.trainParam.showWindow=1; %default is 1)
netpatern.trainFcn = 'trainlm';
netpatern.trainParam.epochs = N_epochs;
netpatern.trainParam.goal  = 0.001;
netpatern.trainParam.show = 100;

% Set up Division of Data for Training, Validation, Testing
netpatern.divideParam.trainRatio = 70/100;
netpatern.divideParam.valRatio = 15/100;
netpatern.divideParam.testRatio = 15/100;
%%

% Train the Network
tic
[netpatern,tr] = train(netpatern,inputs,targets);
t_train=toc;
%%
% Test the Network
% outputs = netpatern(inputs);
% errors = gsubtract(targets,outputs);
% performance = perform(netpatern,targets,outputs)


tic
y = netpatern(inputs);
t_classi=toc;
%%
figure
plotconfusion(targets,y)
%%
for i=1:length(targets)
targets_c(:,i)=find(max(targets(:,i))==targets(:,i));
end

for i=1:length(y)
y_c(:,i)=find(max(y(:,i))==y(:,i));
end


acierto_mlp=(1-length(find(y_c~=targets_c))/length(targets_c))*100;
end
 