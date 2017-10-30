function [right_ratio,time_train,time_test] = SRLSC_function( X_label,X_unlabel,X_test,param )
% Sparse regularization for semi-supervised classification
% Mingyu Fan, Nannan Gu, Hong Qiao, Bo Zhang,
% Pattern Recognition 44 (2011) 1777–1784
% Zhuang Zhao
% 2016-03-31
% Input:
% X_label          labled samples  each row is a sample
% X_unlabe         unlabelded samples
% X_test           testing samples
% param.numofclass  number of all label classes
% param.label_label label of labeled samples
% param.label_test  label of testing samples
% param.ra          coefficients of S-RLSC
% param.rl          coefficients of S-RLSC
% param.sigma       Gaussian kernel

% Output:
% right_ratio_self     recognition rate
% time_train           time of training stage
% time_test            time of testing stage

numofclass = param.numofclass;
label_label = param.label_label;
label_test = param.label_test;
ra = param.ra;
rl = param.rl;
sigma = param.sigma;
X_train = [X_label X_unlabel];
[M_train,N_train] = size(X_train);% M_train样本的维数
X_mean = zeros(M_train,numofclass);% 每一类标记样本的均值

X_label = X_label ./ repmat(sqrt(sum(X_label .^ 2)),[size(X_label,1) 1]);% 按列归一化
% X_unlabel = X_unlabel ./ repmat(sqrt(sum(X_unlabel .^ 2)) + eps,[size(X_unlabel,1) 1]);% 按列归一化
X_train = X_train ./ repmat(sqrt(sum(X_train .^ 2)) + eps,[size(X_train,1) 1]);% 按列归一化
X_test = X_test ./ repmat(sqrt(sum(X_test .^ 2)) + eps,[size(X_test,1) 1]);% 按列归一化

for i = 1:numofclass
    ind = find(label_label == i);
    X_temp = X_label(:,ind);
    X_mean(:,i) = mean(X_temp,2);
end
W = zeros(N_train);
param.lambda = 0.01; % not more than 20 non-zeros coefficients
param.numThreads = -1; % number of processors/cores to use; the default choice is -1 and uses all the cores of the machine
param.mode = 1;
tic
%% 使用同一类的样本对其进行重建
N_label = size(X_label,2);
for i = 1:N_label    
    x_i = X_train(:,i);
    ind = find(label_label ~= i);
    B = X_label(:,ind);
    alpha = mexLasso(x_i,B,param);
    alpha = full(alpha);
    alpha = alpha(:);
    W(i,ind) = alpha';
    %                 W(i,num1:num2) = alpha(:)';
end
% time_sparse = toc;
W = sparse(W);

%%
X2 = sum(X_train .^2 ,1);
distance = repmat(X2,N_train,1) + repmat(X2',1,N_train) - 2 * X_train' * X_train;
ind = distance < 0;
distance(ind) = 0;
K = exp(-(distance) / sigma);
clear distance% clear distance
%%
Ld_new = eye(N_train) - W - W' + W' * W;
ind = find(label_label > 0);
u = zeros(1,N_train);
u(ind) = 1;
U = diag(u);
I = eye(N_train);
I = sparse(I);
Y = zeros(N_train ,numofclass);
for i = 1:numofclass
    ind = find(label_label == i);
    Y(ind,i) = 1;
end
%% 
alpha_new = inv(U * K + rl * Ld_new * K + ra * I) *  Y;
clear K %clear K
time_train = toc;
    tic
    dis_K = exp(-(repmat(sum(X_train .* X_train,1),size(X_test,2),1) + repmat(sum(X_test .* X_test,1)',1,size(X_train,2)) ...
- 2*X_test' * X_train)/(sigma)); 
    temp = dis_K * alpha_new;
    [~,label_k1] = max(temp,[],2);
    df = label_k1(:) - label_test(:);
    ind_right = find(df == 0);
    right_ratio = numel(ind_right) / size(X_test,2);
    time_test = toc;

