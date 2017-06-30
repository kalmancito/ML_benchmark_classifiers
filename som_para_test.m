function [acierto_som, t_classi, t_train] =som_para_test(som_neu,Ncomponents_PCA,N_epochs)

% clear all
% close all
clc
load Trainnumbers.mat
% load Test_numbers_HW1.mat 

 %% =========== Task 4a: Neural Classifiers(FFN)=============
 %
 %
 %
 %
 %%
 
 
 %%
 
%  Ncomponents_PCA=100; %numero de dimensiones con las que vamos a quedarnos.
% som_neu=10;
 
 %%
 
 fprintf(' Neural Classifiers(SOM)(a) ...\n')
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

            X_norm=X_Row;
           imagen_vector{k}=X_norm;
           imagen_label{k}=Trainnumbers.label(k);

    end
%


Inputffn=cell2mat(imagen_vector');
Inputffn=Inputffn';
Outputffn=cell2mat(imagen_label);

%
    X_norm=Inputffn';
%

 
            %  Run PCA   ---> [U, S] = pca(X_norm);
C_X=cov(X_norm);


[U,S] = eig(C_X);


D=length(U);

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

% [residual, preconstructed]=pcares(t_normalized,Ncomponents_PCA);

%%

pvalor=reducedData;
clase=Outputffn;
%%




[trainInd,~,testInd] = dividerand([pvalor;clase],2/3,0,1/3);
trainvalor=trainInd(1:end-1,:);
trainclase=trainInd(end,:);
testvalor=testInd(1:end-1,:);
testclase=testInd(end,:);
%
% k-nn classifier

% Clasificamos t

% nnclass = knnclassify(testvalor', trainvalor', trainclase,knn); 

%%

% Creo y pinto SOM de dos neuronas
Topol = [som_neu som_neu];
net = selforgmap(Topol);
net = configure(net,trainvalor);
filas = Topol(1);
colum = Topol(2);
% Parametros
net.trainParam.epochs = N_epochs;
net.trainParam.showWindow=1;
%%


tic
net = train(net,trainvalor);
t_train=toc;
% plotsomhits(net,trainvalor)

%%


% load('som25x25.mat')
%%


% Posicion de las neuronas
NeurPosit = net.IW{:,:}';
% Asignacion de clases a cada neurona por numero de veces que se activa
% por cada clase de las muestras de entrenamiento
NoNeur = filas*colum;
NoClases = 10;
% Inicializo por eficiencia
Ranking(NoNeur,NoClases)=0;
NeurClas(NoNeur)=0;
% Neuronas que se activan para puntos de entrenamiento
ActivedNeurX = vec2ind(net(trainvalor));
% Ranking
for i=1:length(trainvalor)
neu = ActivedNeurX(i);
clas = trainclase(i)+1;
Ranking(neu,clas) = Ranking(neu,clas)+1;
end
%% Asigno a cada neuroan la clase que más veces la ha activado
 NoNeur = length(Ranking(:,1));
 
 for neu=1:NoNeur
     clase=find(Ranking(neu,:)==max(Ranking(neu,:)));
     if length(clase)>1
        NeurClas(neu) = datasample(clase,1)-1;
     else
     NeurClas(neu) = clase-1;
     end

 end

%  hist(NeurClas)
 
 
 %%
 
 % Calculo de las muestras de test cuantas clasifico bien y cuantas mal
 tic
ActivedNeurT = vec2ind(net(testvalor));
NoMuestrasTest = length(testvalor);
for i=1:NoMuestrasTest
ClasTnn(i) = NeurClas(ActivedNeurT(i));
end
t_classi=toc;
% Calculo cuantas han sido mal clasificadas
% NoErrores = sum(ClasTnn~=testclase
%%
% figure
% hist(ClasTnn)
acierto_som=(1-length(find(ClasTnn~=testclase))/length(testclase))*100;

end