function eigvector_full = lda_function(X,label,numOfClass)
% LDA
% Zhuang Zhao 
% 2015-07-07
% Input:
% X              training sample,each row is a sample
% label          label of training samples
% numOfClass     number of all label classes
% 
% Output
% eigvector_full  eigvector of LDA
samples = X;
samplemean = mean(X);
newsamplemean = zeros(numOfClass , size(X,2));
for i=1:numOfClass
    ind = find(label == i);
    x = samples(ind,:);
    newsamplemean(i,:) = mean(x);
end
sw=0;
for i = 1:numOfClass
    ind = find(label == i);
    num = length(ind);
    for j = 1:num
        sw = sw + (samples(ind(j),:) - newsamplemean(i,:))' * (samples(ind(j),:) - newsamplemean(i,:));
    end
end
% figure,mesh(sw),title('sw lda')
sb=0;
for i=1:numOfClass
    ind = find(label == i);
    num = length(ind);
    sb = sb + num * (newsamplemean(i,:) - samplemean)' * (newsamplemean(i,:) - samplemean);
end

invSw=inv(sw);
newspace = invSw * sb;
[x, y] = eig(newspace);
eigvalue = diag(y);
eigvalue = real(eigvalue);
% [~,index,~] = unique(-real(eigvalue));
[~,index] = sort(real(eigvalue),'descend');
eigvector_full = real(x(:, index));
