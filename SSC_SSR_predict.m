function right_ratio = SSC_SSR_predict(X_test,param)
% Semi-supervised classification based on subspace sparse representation
% Guoxian Yu, Guoji Zhang, Zili Zhang, Zhiwen Yu, Lin Deng
% test the performance of SSC-SSR
% column first
% 2016-05-04
label_test = param.label_test;
V_out = param.V_out;
B_out = param.B_out;
M = param.M;
N_test = size(X_test,2);
select_index = param.select_index;
T = param.T;
numofclass = param.numofclass;
F = 0;
for i = 1:T
    index = select_index(:,i);
    X_test_T = X_test(index,:);
%     X_test_T = X_test_T ./ repmat(sqrt(sum(X_test_T .^ 2)) + eps,[size(X_test_T,1) 1]);% 按列归一化
    V = V_out(:,(i-1) * numofclass + 1:i * numofclass);
    B = B_out(:,i);
    f = V(index,:)' * X_test_T + repmat(B,[1,N_test]);% c*N_test
    Mt = M(:,i);
    Mt = reshape(Mt,[numel(Mt),1]);
    Mt = repmat(Mt,[1,N_test]);
    F = F + Mt .* f;
end
[~,label_predict] = max(F,[],1);
df = label_predict(:) - label_test(:);
ind = find(df == 0);
right_ratio = numel(ind) / N_test;