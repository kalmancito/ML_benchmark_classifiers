 clc
num_epochs=100;
Ncomponents_PCA=100;

for som_neu=5:3:25
[acie_som(som_neu), e_time(som_neu), t_time(som_neu)]=som_para_test(som_neu,Ncomponents_PCA,num_epochs)
end
%%
% save('resultadossom_neuronas2.mat','acie_som', 'e_time','t_time')
%%
plot([5:3:25],acie_som)
xlabel('number of neurons grid nxn')
ylabel('% of right classification')
grid on
%%
plot([5:3:25],e_time)
xlabel('number of neurons grid nxn')
ylabel('time for classify')
grid on
%%
plot([5:3:25],t_time)
xlabel('number of neurons grid nxn')
ylabel('time for training')
grid on
%%
plot(acie_som)
som_neu=20;
cont=0;
for num_epochs=100:50:400
    cont=cont+1;
[acie_som_e(cont), e_time_e(cont), t_time_e(cont)]=som_para_test(som_neu,Ncomponents_PCA,num_epochs)
end

% save('resultadossom_epochs.mat','acie_som_e', 'e_time_e','t_time_e')
%%
plot([100:50:400],acie_som_e)
xlabel('number of epochs')
ylabel('% of right classification')
grid on
%%
plot([100:50:400],e_time_e)
xlabel('number of epochs')
ylabel('time for classify')
grid on
%%
plot([100:50:400],t_time_e)
xlabel('number of epochs')
ylabel('time for training')
grid on

%%

cont=0;
som_neu=20;
num_epochs=250;
for Ncomponents_PCA=[60,80,100,120,150]
    cont=cont+1;
[acie_som_pca(cont), e_time_pca(cont), t_time_pca(cont)]=som_para_test(som_neu,Ncomponents_PCA,num_epochs)
end
%%
% save('resultadossom_pca.mat','acie_som_pca', 'e_time_pca','t_time_pca')
%%
plot([60,80,100,120,150],acie_som_pca)
xlabel('PCA components')
ylabel('% of right classification')
grid on
%%
plot([60,80,100,120,150],e_time_pca)
xlabel('PCA components')
ylabel('time for classify')
grid on
%%
plot([60,80,100,120,150],t_time_pca)
xlabel('PCA components')
ylabel('time for trainining')
grid on

%%
cont=0;
num_epochs=100;
Ncomponents_PCA=50;
cont=0;
for mlp_neu=15:5:30
    cont=cont+1;
[acie_mlp(cont), e_time(cont), t_time(cont)]=mlp_para_test(mlp_neu,Ncomponents_PCA,num_epochs)
end

%%
% save('resultadossom_pca.mat','acie_som_pca', 'e_time_pca','t_time_pca')
%%
plot([15:5:30],acie_mlp)
xlabel('number of neurons ')
ylabel('% of right classification')
grid on
%%
plot([15:5:30],e_time)
xlabel('number of neurons ')
ylabel('time for classify')
grid on
%%
plot([15:5:30], t_time)
xlabel('number of neurons ')
ylabel('time for training')
grid on


%%





cont=0;
num_epochs=100;
mlp_neu=20;
cont=0;
for Ncomponents_PCA=[10,30,60,100,150,200]
    cont=cont+1;
[acie_mlp_pca(cont), e_time_pca(cont), t_time_pca(cont)]=mlp_para_test(mlp_neu,Ncomponents_PCA,num_epochs)
end
%%
% save('resultadosmlp_pca.mat','acie_mlp_pca', 'e_time_pca','t_time_pca')
%%
plot([10,30,60,100,150,200],acie_mlp_pca)
xlabel('PCA components')
ylabel('% of right classification')
grid on
%%
plot([10,30,60,100,150,200],e_time_pca)
xlabel('PCA components')
ylabel('time for classify')
grid on
%%
plot([10,30,60,100,150,200],t_time_pca)
xlabel('PCA components')
ylabel('time for training')
grid on
%%










cont=0;
% num_epochs=100;
mlp_neu=10;
cont=0;
Ncomponents_PCA=60;
for num_epochs=[50,100,200,300]
    cont=cont+1;
[acie_mlp_e(cont), e_time_e(cont), t_time_e(cont)]=mlp_para_test(mlp_neu,Ncomponents_PCA,num_epochs)
end
%%
save('resultadosmlp_epochs.mat','acie_mlp_e', 'e_time_e','t_time_e')
%%
plot([50,100,200,300],acie_mlp_e)
xlabel('number of epochs')
ylabel('% of right classification')
grid on
%%
plot([50,100,200,300],e_time_e)
xlabel('number of epochs')
ylabel('time for classify')
grid on
%%
plot([50,100,200,300],t_time_e)
xlabel('number of epochs')
ylabel('time for training')
grid on



%%



%%



cont=0;
% num_epochs=100;
mlp_neu=15;
cont=0;
Ncomponents_PCA=30;
num_epochs=50;
for kk=1:10
    cont=cont+1;
[acie_mlp_e(cont), e_time_e(cont), t_time_e(cont)]=mlp_para_test(mlp_neu,Ncomponents_PCA,num_epochs)
end
acie_mlp_e=mean(acie_mlp_e)
e_time_e=mean(e_time_e)
acie_mlp_e=mean(t_time_e)
%%
% save('resultadosmlp_train.mat','acie_mlp_e', 'e_time_e','t_time_e')
%%
plot([1,2,3,4,5],acie_mlp_e,'*-')
xlabel('hidden layers')
ylim([90 100])
ylabel('% of right classification')
grid on
%%
plot([1:5],e_time_e,'*-')
xlabel('hidden layers')
ylabel('time for classify')
grid on
%%
plot([1:5],t_time_e,'*-')
xlabel('hidden layers')
ylabel('time for training')
grid on
ylim([100 200])