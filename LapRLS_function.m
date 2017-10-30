function [right_ratio_self,time_train,time_test] = LapRLS_function( X_label, X_unlabel,X_test,label_label,label_test, ...
numofclass,numoflabel,numofunlabel,ra,rl,sigma,KK)
% function [ output_args ] = SSDR( input_args )
% Manifold Regularization: A Geometric Framework for Learning from Examples
% Mikhail Belkin, Partha Niyogi, Vikas Sindhwani
% Zhuang Zhao
% 2015-07-07
% Input:
% X_label        labled samples  each row is a sample
% X_unlabe       unlabelded samples
% X_test         testing samples
% label_label    label of labeled samples
% label_unlabel  label of unlabeled samples
% label_test     label of testing samples
% numofclass     number of all label classes
% numperclass    number of smaples per class
% numoflabel     number of samples per labeled class
% rd,ra,rl       coefficients of LapRLS
% sigma          Gaussian kernel

% Output:
% right_ratio_self     recognition rate
% time_train           time of training stage
% time_test            time of testing stage



tic
N_label = numofclass * numoflabel;
X_train = [X_label;X_unlabel];
ind = find(label_label == 0);
label_label(ind) = [];

ind =find(label_test == 0);
label_test(ind) = [];





[N,M] = size(X_train);
X2 = sum(X_train .^2 ,2);
distance = repmat(X2,1,N) + repmat(X2',N,1) - 2 * X_train * X_train';
ind = distance < 0;
distance(ind) = 0;
K = exp(-(distance) / (sigma));

%% new algorithm
[~,idx] = sort(distance,2);
clear distance;
W_new = zeros(N);
for i = N: -1 :1%先处理未标记点
    id_temp = idx(i,:);
    W_new(i,id_temp(1:KK)) = 1;
end
clear idx;
%%
W_new = sparse(W_new);
Ww_new = W_new;
Dw_new = diag(sum(Ww_new,1));
Lw_new = Dw_new - Ww_new;
Ld_new = rl * Lw_new;
ind = find(label_label > 0);
u = zeros(1,N);
u(ind) = 1;
U = diag(u);
U = sparse(U);
I = eye(N);
I = sparse(I);
Y = zeros(N ,numofclass);
for i = 1:numofclass
    ind = find(label_label == i);
    Y(ind,i) = 1;
end

alpha_new = inv(U * K + Ld_new * K + ra * I) *  Y;
clear K;
clear U;
clear Ld_new;
clear I;
time_train = toc;
label_k = zeros(1,size(X_test,1));
debug = 1;
tic
 K = exp(-(repmat(sum(X_train .* X_train,2)',size(X_test,1),1) + repmat(sum(X_test .* X_test,2),1,size(X_train,1)) ...
        - 2*X_test * X_train')/(sigma)); 
temp = K * alpha_new;
[~,label_k] = max(temp,[],2);
df = label_k(:) - label_test(:);
ind_right = find(df == 0);
right_ratio1 = numel(ind_right) / size(X_test,1);
right_ratio_self = right_ratio1;
time_test = toc;
