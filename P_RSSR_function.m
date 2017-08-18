function [W_sparse,select_index,train_time,S] = P_RSSR_function( X_label,X_unlabel,param)
% Random Subspace Sparse Representation with select dimension with different probability
% get the sparse cofficients in each subsapce
%  
% 
% 2016-08-18

%%
label_label = param.label_label;
numofclass = param.numofclass;
T = param.T;% Number of base classifiers
P = param.P;% Dimensionality of random subspace
% select_unlabel = param.select_unlabel;
% num_select_unlabel = param.num_select_unlabel;
% select_label = param.select_label;
%%
X_train = [X_label X_unlabel];% column first
N_train = size(X_train,2);
N_label = size(X_label,2);
N_unlabel = size(X_unlabel,2);
Dim = size(X_label,1);% Dimensionality of original data
%% Randomly generate T binary indicative vectors rt
tic
%% use LASSO to sovle SR
param_LASSO.lambda = 0.001; % not more than 20 non-zeros coefficients
param_LASSO.numThreads = -1; % number of processors/cores to use; the default choice is -1 and uses all the cores of the machine
param_LASSO.mode = 1;

%% use OMP to solve SR
% param_OMP_SR.L = Dim;
% param_OMP.numThreads = -1;
% param_OMP.eps = 0;
%% 
Y = zeros(N_train,numofclass);
for i = 1:N_label
    y = zeros(1,numofclass);
    y(label_label(i)) = 1;
    Y(i,:) = y;
end

U = zeros(N_train);
for i = 1:N_label
    U(i,i) = 1;
end
select_index = zeros(T,P);
S = zeros(Dim * T,N_train);
% L_new = zeros(N_train,N_train,T);
W_sparse = zeros(N_train * T,N_train);
%% calculate the probability 
% prob = cal_local_entropy(X_label,X_unlabel,param);
out = cal_local_entropy(X_label,X_unlabel,param);
ind_0 = out.ind_0;% less than the mean value
ind_1 = out.ind_1;% less than the mean value
s_0 = out.s_0;
s_1 = out.s_1;
select_1 = ceil(s_1 / (s_0 + s_1) * P);% 
select_0 = P - select_1;
%% 



M = param.M;
N = param.N;
for i = 1:T
   a = rand(numel(ind_0),1);
   [~,index] = sort(a);
   index_0 = ind_0(index(1:select_0));
   a = rand(numel(ind_1),1);
   [~,index] = sort(a);
   index_1 = ind_1(index(1:select_1));
   index = [index_0(:);index_1(:)];
% 	index = randsrc(10 ,10,[1:M * N;prob(:)']);
    select_index(i,:) = index(:)';
    
    ind = zeros(Dim,1);
    ind(index) = 1;
    num1 = (i - 1) * Dim + 1;
    num2 = i * Dim;
    S(num1:num2,:) = repmat(ind(:),[1 N_train]);
    
    X_label_T = X_label(index,:);
    X_unlabel_T = X_unlabel(index,:);
    X_label_T = X_label_T ./ repmat(sqrt(sum(X_label_T .^ 2)) + eps,[size(X_label_T,1) 1]);% 
    X_unlabel_T = X_unlabel_T ./ repmat(sqrt(sum(X_unlabel_T .^ 2)) + eps,[size(X_unlabel_T,1) 1]);% 
    X_train_T = [X_label_T X_unlabel_T];
    %     X_train_T = X_train_T ./ repmat(sqrt(sum(X_train_T .^ 2)) + eps,[size(X_train_T,1) 1]);% 
    
    %% SR
    W = zeros(N_train);
    %     tic
    parfor index_i = 1:N_train
        
        x_i = X_train_T(:,index_i);
        if index_i <= N_label % labeled samples
%             select_unlabeled_sample = select_unlabel(label_label(index_i),1:num_select_unlabel(label_label(index_i)));
            select_unlabeled_sample = 1:N_unlabel;
            ind = find(label_label == label_label(index_i));
            index = ismember(ind,index_i);
            index = ~index;
            ind_s = ind(index);
            D = [X_label_T(:,ind_s) X_unlabel_T(:,select_unlabeled_sample)];
            %% LASSO
            ss = mexLasso(x_i,D,param_LASSO);
            %% OMP
%             ss = mexOMP(x_i,D,param_OMP);
            %%
            ss = full(ss);
            ss = ss(:);
            mod_i = mod(index_i,numel(ind));
            if mod_i == 1
                ss = [0;ss];
            elseif mod_i == 0
                ss = [ss;0];
            else
                s11 = ss(1:mod_i - 1);
                s12 = ss(mod_i:end);
                ss = [s11;0;s12];
            end
            s = zeros(N_train,1);
            index = [ind(:);N_label + select_unlabeled_sample(:)];
            s(index) = ss;
        else % unlabeled samples
            index_unlabel = index_i - N_label;
%             ind = find(label_label == select_label(index_unlabel));
            ind = 1:N_label;
            index = 1:N_unlabel;
            index(index_unlabel) = [];
            D = [X_unlabel_T(:,index) X_label_T(:,ind)];
            %% 
            ss = mexLasso(x_i,D,param_LASSO);
            %% 
%             ss = mexOMP(x_i,D,param_OMP);
            ss = full(ss);
            ss = ss(:);
            mod_i = mod(index_unlabel,N_unlabel);
            if mod_i == 1
                ss = [0;ss];
            elseif mod_i == 0
                ss = [ss;0];
            else
                s11 = ss(1:mod_i - 1);
                s12 = ss(mod_i:end);
                ss = [s11;0;s12];
            end
            s = zeros(N_train,1);
            index = 1:N_unlabel;
            index = [N_label + index(:);ind(:)];
            s(index) = ss;
        end
        W(index_i,:) = s(:)';
    end
    num1 = (i - 1) * N_train + 1;
    num2 = i * N_train;
    W_sparse(num1:num2,:) = W;
%     L_new(:,:,i) = eye(N_train) - W - W' + W' * W;
end
W_sparse = sparse(W_sparse);
S = sparse(S);
train_time = toc;

