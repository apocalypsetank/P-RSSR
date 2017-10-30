function [newsample, basevector,eig_value] = pca_self(patterns,num)
% Input:
% patterns       input samples. each column is a samples
% num            if num is bigger than 1, we select the form num features
%                if num if less than 1, we select the energy as num

% Output:  
% newsample      the mapped sample
% basevector     eigvector of PCA
% eig_value      eigvalue of input sample
% ��������������patterns��ʾ����ģʽ������һ�б�ʾһ��������numΪ���Ʊ�������num����1��ʱ���ʾ
% Ҫ���������Ϊnum����num����0С�ڵ���1��ʱ���ʾ��ȡ��������������Ϊnum
% �����basevector��ʾ��ȡ���������ֵ��Ӧ��������������С�ǽ�Ϊ���ά��*ԭʼά��
% newsample��ʾ��basevectorӳ���»�õ�������ʾ��
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
        %         ����һ����ʽ�ı�׼��
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