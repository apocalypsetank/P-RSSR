function [newsample, basevector,eig_value] = pca_self(patterns,num)
% Input:
% patterns       input samples. each column is a samples
% num            if num is bigger than 1, we select the form num features
%                if num if less than 1, we select the energy as num

% Output:  
% newsample      the mapped sample
% basevector     eigvector of PCA
% eig_value      eigvalue of input sample
% 主分量分析程序，patterns表示输入模式向量，一列表示一个样本，num为控制变量，当num大于1的时候表示
% 要求的特征数为num，当num大于0小于等于1的时候表示求取的特征数的能量为num
% 输出：basevector表示求取的最大特征值对应的特征向量，大小是将为后的维数*原始维数
% newsample表示在basevector映射下获得的样本表示。
[~,u] = size(patterns);
totalsamplemean = mean(patterns,2);
totalsamplemean = repmat(totalsamplemean,[1 u]);
% gensample = patterns;
gensample = patterns - totalsamplemean * 1;
% for i=1:u
%     gensample(i,:)=patterns(i,:)-totalsamplemean;
% end
% sigma = gensample' *gensample;
sigma = cov(gensample');
[U, V]=eig(sigma);
d=diag(V);
[d1, index] = sort(real(d),'descend');
if num>1
    for i=1:num
        vector(:,i)=U(:,index(i));
        %         base(:,i)= gensample * vector(:,i) / d(index(i))^(1/2);
        %         base = gensample' * vector(:,num) * diag(d(1:num) .^ (-0.5));
        %         另外一种形式的标准化
        %         base = xmean' * vsort(:,1:p) * diag(dsort(1:p).^(-1/2));
    end
else
    sumv=sum(d1);
    for i=1:u
        if sum(d1(1:i))/sumv>=num
            num_t = i;
            break;
        end
    end
    %     base = gensample' * vector(:,num) * diag(d(1:num) .^ (-0.5));
    for i=1:num_t
        vector(:,i)=U(:,index(i));
        %         base(:,i)= gensample * vector(:,i) / d(index(i))^(1/2);
    end
end
newsample = vector' * patterns;
basevector = vector;
eig_value = d;