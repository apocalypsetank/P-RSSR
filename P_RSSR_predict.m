function [right_ratio,train_time,test_time] = P_RSSR_predict(X_label,X_unlabel,X_test,W_sparse,select_index,param,beta,gamma,sigma)
% column first

%%

%% 
Dim = param.Dim;
C = param.numofclass;
label_test = param.label_test;
label_label = param.label_label;
%%
N_label = size(X_label,2);
N_unlabel = size(X_unlabel,2);
N_train = N_label + N_unlabel;
L_optimal = zeros(N_train);
T = param.T;
P = param.P;
I = diag(ones(P,1));
e = ones(N_train,1);
H = diag([ones(N_label,1);zeros(N_unlabel,1)]);
Hc = H - (H * e * e' * H') / N_train;
% I = diag(ones(N_train,1));
% L_all = zeros(N_train);
X_train = [X_label X_unlabel];
Y = zeros(N_train,C);
for i = 1:N_label
    y = zeros(1,C);
    y(label_label(i)) = 1;
    Y(i,:) = y;
end
N_test = size(X_test,2);
label_predict_T = zeros(T,N_test);
train_time_temp = zeros(T,1);
test_time_temp = zeros(T,1);

%% use Gaussian Kernel 
I = diag(ones(N_train,1));
for i = 1:T
    tic
    num1 = (i - 1) * N_train + 1;
    num2 = i * N_train;
    W = W_sparse(num1:num2,:);
    L = eye(N_train) - W - W' + W' * W;
    index = select_index(i,:);
    X_train_T = X_train(index,:);
    X_test_T = X_test(index,:);
    
    [K_all,~] = cal_dis([X_train_T,X_test_T],'col');
    K_all = exp(-K_all .* K_all / sigma);
    K_train = K_all(1:N_train,1:N_train);
    Alpha = (H * K_train + beta * L * K_train + gamma * I) \ Y;

    

    train_time_temp(i) = toc;
%     train_time = train_time + time_temp * N_train / (N_train + N_test);
    %% test
    tic
    K_test = K_all(1:N_train,N_train + 1:end);
    clear K_all;% clear K_all


    temp = K_test' * Alpha;
    [~,label_predict_T(i,:)] = max(temp,[],2);
    test_time_temp(i) = toc;
end
label_predict = zeros(1,N_test);
train_time = sum(train_time_temp);
tic
for i = 1:N_test
    t = tabulate(label_predict_T(:,i));
    t1 = t(:,1);
    [~,ind] = max(t(:,2));
    label_predict(i) = t1(ind(1));
end
df = label_predict(:) - label_test(:);
ind = find(df == 0);
right_ratio = numel(ind) / N_test;
test_time = toc;
test_time = test_time + sum(test_time_temp);
%}

