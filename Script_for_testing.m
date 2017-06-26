
clc; clear all; close all;
%% Loading the data
load Trainnumbers.mat
 %% =========== Task : Script for testing and graphics =============

 
 fprintf('Script for testing and graphics ...\n')
 
 %
 cont=0;
Ncomponents_PCA=1:10:200;
nsamples=10000   
    cont=cont+1;
    
    for k=1:length(Trainnumbers.label(1:nsamples))% ejemplo
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
           imagen_label{k}=Trainnumbers.label(k);
   
    end
%


Inputffn=cell2mat(imagen_vector');
Inputffn=Inputffn';
Outputffn=cell2mat(imagen_label);
Outputffn=Outputffn;
%

 [X_norm, mu, sigma]=zscore(Inputffn');
            

            %  Run PCA   ---> [U, S] = pca(X_norm);
C_X=cov(X_norm);

[U,S] = eig(C_X);

D=length(U);

 %numero de dimensiones con las que vamos a quedarnos.
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

mseError_PCA = mse(X_norm-X_rec);





pvalor=reducedData;
clase=Outputffn;

%%

tic
[bayclassq, err_qua, posteriorq] = classify(pvalor',pvalor', clase, 'quadratic');
exe_time_q(cont)=toc;
% 
% Comparación
err_qua ;
num_errors_bay_qua(cont)=length (find(bayclassq'~=clase))/length(clase)*100;
% De los  datos clasificados, 6.48% están mal (con frontera cuadrática) 

%%
tic
[bayclassl, err_lin, posteriorl] = classify(pvalor',pvalor', clase, 'linear');
exe_time_l(cont)=toc;
% 
% Comparación
err_lin ;
num_errors_bay_lin(cont)=length (find(bayclassl'~=clase))/length(clase)*100;
% De los  datos clasificados, 14% están mal (con frontera lineal)

%%
for k=1:length(clase)
           clase_confu(k,:)=zeros(10,1);
           
           clase_confu(k,clase(k)+1)=1;
           
           bayclassq_confu(k,:)=zeros(10,1);
           
           bayclassq_confu(k,bayclassq(k)+1)=1;
end
%%
plotconfusion(clase_confu',bayclassq_confu')
%%
%%

plot([500:100:10000],exe_time_q,'b');hold on; grid on;
plot([500:100:10000],exe_time_l,'r')
xlabel('number of samples')
ylabel('execution time(s)')
title('execution time vs number of samples')
legend('quadratic','linear')
%%
plot([500:100:10000],num_errors_bay_qua,'b');hold on; grid on;
plot([500:100:10000],num_errors_bay_lin,'r')
xlabel('number of samples')
ylabel('% of the error')
title('% of error in classification vs number of samples')
legend('quadratic','linear')
%%
plot([1:10:200],num_errors_bay_qua,'b');hold on; grid on;
plot([1:10:200],num_errors_bay_lin,'r')
xlabel('PCA Components')
ylabel('% of the error')
title('% of error in classification vs PCA components used')
legend('quadratic','linear')
