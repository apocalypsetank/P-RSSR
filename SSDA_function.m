function eigvector_full = SSDA_function(X_train,numofclass,numoflabel,alpha,beta ,KK)
% Semi-supervised Discriminant Analysis?
% Deng Cai,Xiaofei He,Jiawei Han

% Zhuang Zhao
% 2015-06-15
% Input:
% X_train        training samples including labeled and unlabeled samples
% numofclass     number of all label classes
% numoflabel     number of samples per labeled class
% alpha,beta     coefficients of LapRLS
% KK             neighborhood size

% 
% Output
% eigvector_full  eigvector of SSDA
%% 
[N,M] = size(X_train);
N_train = N;
Dim = M;
X2 = sum(X_train .^2 ,2);
distance = repmat(X2,1,N) + repmat(X2',N,1) - 2 * X_train * X_train';
ind = distance < 0;
distance(ind) = 0;
[~,id] = sort(distance,2);
%% 
S = zeros(N_train);
for i = 1:N_train
    S(i,id(i,1:KK)) = 1;
end
L = diag(sum(S)) - S;

W = zeros(N_train);
for i = 1:numofclass
    num1 = (i - 1) * numoflabel + 1;
    num2 = i * numoflabel;
    W(num1:num2,num1:num2) = 1 / numoflabel;
end

I = zeros(1,N_train);
I(1:numofclass * numoflabel) = 1;
I = diag(I);
% alpha = 0.8;
% beta = 0.005;
LPrime = X_train' * W * X_train;
DPrime = X_train' * (I + alpha * L) * X_train + beta * eye(Dim);

[eigvector, eigvalue] = eig(LPrime,DPrime);
eigvalue = diag(eigvalue);
[~, index] = sort(eigvalue);
eigvector_full = eigvector(:,index);

