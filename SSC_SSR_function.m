function [V_out,B_out,M,select_index] = SSC_SSR_function( X_label,X_unlabel,param)
% Semi-supervised classification based on subspace sparse representation
% Guoxian Yu, Guoji Zhang, Zili Zhang, Zhiwen Yu, Lin Deng
% 
% 2016-05-04

%% 
label_label = param.label_label;
numofclass = param.numofclass;
T = param.T;% Number of base classifiers
P = param.P;% Dimensionality of random subspace
Mu = param.Mu;
alpha = param.alpha;
beta = param.beta;
%% 
X_train = [X_label X_unlabel];%column first  
N_train = size(X_train,2);
N_label = size(X_label,2);
N_unlabel = size(X_unlabel,2);
D = size(X_label,1);% Dimensionality of original data
%% Randomly generate T binary indicative vectors rt

param_SR.lambda = 0.0001; % 
param_SR.numThreads = -1; % number of processors/cores to use; the default choice is -1 and uses all the cores of the machine
param_SR.mode = 1;
Y = zeros(N_train,numofclass);
H = zeros(N_train);
for i = 1:N_label
    y = zeros(1,numofclass);
    y(label_label(i)) = 1;
    Y(i,:) = y;
end
h = [ones(N_label,1);zeros(N_unlabel,1)];
H = diag(h);
e = ones(N_train,1);
Hc = H - H * e * e' * H' / N_train;
Fc = zeros(numofclass,N_train,T);
It = diag(ones(1,T)) * Mu;
select_index = zeros(P,T);
V_out = [];
B_out = [];
for i = 1:T
    a = rand(D,1);
    [~,index] = sort(a);
    index = index(1:P);
    select_index(:,i) = index(:);
    X_train_T = X_train(index,:);
%     X_train_T = X_train_T ./ repmat(sqrt(sum(X_train_T .^ 2)) + eps,[size(X_train_T,1) 1]);% 按列归一化
    
    % SR
    W = zeros(N_train);
    for index_i = 1:N_train
        index_sample = 1:N_train;
        x_i = X_train_T(:,index_i);
        ind_sampleeclass = index_sample;
        ind_sampleeclass(index_i) = [];
        B = X_train_T(:,ind_sampleeclass);
        s = mexLasso(x_i,B,param_SR);
        s = full(s);
        s = s(:);
        mod_i = mod(index_i,N_train);
        if mod_i == 1%第一个
            s = [0;s];
        elseif mod_i == 0% 最后一个
            s = [s;0];
        else% 中间
            s11 = s(1:mod_i - 1);
            s12 = s(mod_i:end);
            s = [s11;0;s12];
        end
        W(index_i,:) = s(:)';
    end
%     L_new = diag(sum(W,1)) - W;
    L_new = eye(N_train) - W - W' + W' * W;
%     V = inv(X_train_T * Hc * X_train_T' + alpha * X_train_T * L * X_train_T') * X_train_T * Hc * Y;% Eq.(19)
%     B = (Y' - V' * X_train_T) * H * e / N_train;% Eq.(20)
%     V = inv(X_train_T * Hc * X_train_T' + alpha * X_train_T * L * X_train_T' + beta * diag(ones(P,1))) * X_train_T * Hc * Y;% A regularization term is added to avoid the singular problem
%     fc = V' * X_train_T + repmat(B,[1,N_train]);
    V = inv(X_train * Hc * X_train' + alpha * X_train * L_new * X_train') * X_train * Hc * Y;% Eq.(19)
    B = (Y' - V' * X_train) * H * e / N_train;% Eq.(20)
    V = inv(X_train * Hc * X_train' + alpha * X_train * L_new * X_train' + beta * diag(ones(D,1))) * X_train * Hc * Y;% A regularization term is added to avoid the singular problem
    fc = V(index,:)' * X_train_T + repmat(B,[1,N_train]);
    Fc(:,:,i) = fc;
    V_out = [V_out V];
    B_out = [B_out B];
end
M = [];
for i = 1:numofclass
    Flc = Fc(i,:,:);
    Flc = reshape(Flc,[N_train,T]);
    F_lc = Flc(1:N_label,:);
    Y_lc = Y(1:N_label,i);
    w = inv(F_lc' * F_lc + It) * F_lc' * Y_lc;
    M = [M w];
end
M = M';


