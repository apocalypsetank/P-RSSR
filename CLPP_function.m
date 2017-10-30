function eigvector_full = CLPP_function(X_train,label_train,K,sigma)
% SEMI-SUPERVISED DIMENSIONALITY REDUCTION USING PAIRWISE EQUIVALENCE CONSTRAINTS
% Hakan Cevikalp, Jakob Verbeek, Fred? eric Jurie, Alexander Kl? aser¨
% Zhuang Zhao
% 2015-02-02
% X_train        training samples including labeled and unlabeled samples
% label_train    label of training samples
% K              neighborhood size
% sigma          Gaussian kernel
% 
% Output
% eigvector_full  eigvector of CLPP

N = size(X_train,1);
X2 = sum(X_train .^2 ,2);
distance = repmat(X2,1,N) + repmat(X2',N,1) - 2 * X_train * X_train';
ind = distance < 0;
distance(ind) = 0;
[~,id] = sort(distance,2);
Ws = zeros(N,N);
Wd = zeros(N,N);
W = zeros(N,N);
for i = 1:N
    id_temp = id(i,:);
    W(i,id_temp(1:K+1)) = 1;
end
W = exp(-sqrt(distance) / sigma) .* W;
%% 原始程序
% for i = 1:N
%     label_a = label_train(i);
%     knn_a = id(i,2:1 + K);
%     index = 1:N;
%     index(i) = [];
%     for index1 = 1:N - 1
%         j = index(index1);
%         label_b = label_train(j);
%         knn_b = id(j,2:1 + K);
%         if label_a == label_b
%             ind = ismember(knn_a,knn_b);
%             ind = (knn_a(ind));
%             Ws(i,ind) = 1;
%         elseif label_a ~= label_b
%             ind = ismember(knn_a,knn_b);
%             ind = (knn_a(ind));
%             Wd(i,ind) = -1;
%         end
%     end
% end

%% 优化后
for i = 1:N
    knn_a = id(i,2:1 + K);
%     if label_a > 0
    ind_sameclass = find(label_train == label_train(i));
    ind_sameclass(ismember(ind_sameclass,i)) = [];
    idx = id(ind_sameclass,2:1 + K);
    ind_knn = ismember(knn_a,idx);
    Ws(i,knn_a(ind_knn)) = 1;

    ind_diffclass = find(label_train ~= label_train(i));
    ind_diffclass(ismember(ind_diffclass,i)) = [];
    idx = id(ind_diffclass,2:1 + K);
    ind_knn = ismember(knn_a,idx);
    Wd(i,knn_a(ind_knn)) = -1;
%     end
end
% time2 = toc;
% ratio = time1 / time2;
clear distance;
clear id;

Ws = max(Ws,Ws');
Wd = max(Wd,Wd');
W = max(W,W');
Ws = sparse(Ws);
Wd = sparse(Wd);
W = sparse(W);
Ds = diag(sum(Ws,1));
Dd = diag(sum(Wd,1));
D = diag(sum(W,1));
Ls = Ds - Ws;
Ld = Dd - Wd;
L = D - W;
L_finnal = L + Ls - Ld;
D_finnal = D + Ds - Dd;
DPrime = X_train' * D_finnal * X_train;
LPrime = X_train' * L_finnal * X_train;
clear Ws Wd W Ls Ld L;

[eigvector, eigvalue] = eig(LPrime,DPrime);
eigvalue = diag(eigvalue);
[~, index] = sort(eigvalue);
eigvector_full = eigvector(:,index);
eigvector_full = real(eigvector_full);
