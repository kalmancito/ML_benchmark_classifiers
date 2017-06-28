clear all
close all
clc
load Trainnumbers.mat

 %% =========== Task 1: Dimensionality reduction(PCA)=============
 %
 %
 %
 %
 
 fprintf('Dimensionality reduction(PCA) ...\n')
 
 %
 
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
Outputffn=Outputffn*100;
%

 [X_norm, mu, sigma]=zscore(Inputffn');
            

            %  Run PCA   ---> [U, S] = pca(X_norm);
C_X=cov(X_norm);

[U,S] = eig(C_X);

D=length(U);

Ncomponents_PCA=25 %numero de dimensiones con las que vamos a quedarnos.
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
size(reducedData);

%

X_rec=reducedData'*transf_mat;
%

mseError_PCA = mse(X_norm-X_rec)


fprintf('Program paused. Press enter to continue.\n');
pause;
