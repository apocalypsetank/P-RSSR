function [right_ratio_ssdr,time_train,time_test] = SSDR_function( X_label, X_unlabel,X_test,label_label,label_unlabel,label_test, ...
numofclass,numperclass,numoflabel,numofunlabel,rd,ra,rl,sigma,KK)
% Classification by semi-supervised discriminative regularization
% Fei Wu,Wenhua Wang, Yi Yang, Yueting Zhuang, Feiping Nie
% Zhuang Zhao
% 2015-01-26


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
N_unlabel = numofclass * numofunlabel;
N_test = numofclass * (numperclass - numoflabel - numofunlabel);
N_train = numofclass * (numoflabel + numofunlabel);




X_train = [X_label;X_unlabel];
% [X_train,base] = pca(X_train,0.95);
% X_test = X_test * base;

ind = find(label_label == 0);
label_label(ind) = [];
ind_label = 1:numofclass * numoflabel;
X_label = X_train(ind_label,:);

ind = find(label_unlabel == 0);
label_unlabel(ind) = [];
ind_unlabel = numofclass * numoflabel + 1:N_train;
X_unlabel = X_train(ind_unlabel,:);
label_train = [label_label label_unlabel];

ind =find(label_test == 0);
label_test(ind) = [];





[N,M] = size(X_train);
X2 = sum(X_train .^2 ,2);
distance = repmat(X2,1,N) + repmat(X2',N,1) - 2 * X_train * X_train';
ind = distance < 0;
distance(ind) = 0;
[~,id] = sort(distance,2);
K = exp(-sqrt(distance) / sigma);

%% 
num_label = numel(label_label);
Ww = zeros(N,N);
Wb = zeros(N,N);
W = zeros(N,N);
U = zeros(N,N);
for i = 1:numofclass
    ind = find(label_label == i);
    Ww(ind,ind) = 1;
    ind1 = find(label_label ~= i);
    Wb(ind,ind1) = 1;
end
Ww = Ww / num_label;
Wb = Wb / num_label;
for i = 1:N
    id_temp = id(i,:);
    W(i,id_temp(1:KK)) = 1;
end

Dw = diag(sum(Ww,1));
Db = diag(sum(Wb,1));
D = diag(sum(W,1));
Lw = Dw - Ww;
Lb = Db - Wb;
L = D - W;
ind = find(label_label > 0);
u = zeros(1,N);
u(ind) = 1;
U = diag(u);

I = eye(N);
I1 = ones(N,1);

%%

Y = zeros(N ,numofclass);
for i = 1:numofclass
    ind = find(label_label == i);
    Y(ind,i) = 1;
end
Ld = rd * Lw - (1 - rd) * Lb + rl * L;
alpha = inv(U * K + Ld * K + ra * I) *  Y;
time_train = toc;
%% 
tic;
label_k = zeros(1,size(X_test,1));
debug = 1;
if debug
    parfor i = 1:size(X_test,1)
        k = X_test(i,:);
        k = repmat(k,[numofclass * (numoflabel + numofunlabel) 1]);
        dis_k = sum((k - X_train) .* (k - X_train),2);
        dis_k = exp(-sqrt(dis_k) / sigma);
        temp1 = dis_k' * alpha;
        [~,label_k(i)] = max(temp1);
    end
    ind_test = find(label_test == 0);
    label_test(ind_test) = [];
    df = label_k(:)' - label_test(:)';
    ind_right = find(df == 0);
    right_ratio1 = numel(ind_right) / size(X_test,1);
end
right_ratio_ssdr = right_ratio1;
time_test = toc;

