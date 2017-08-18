% function prob = cal_local_entropy(X_label,X_unlabel,param)
function out = cal_local_entropy(X_label,X_unlabel,param)
% use local entropy to calculate the probabilty of all pixels
% param.M and param.N is the local window size, in this paper, 
% param.M = param.N = 3;
M = param.M;
N = param.N;
Num_train = size(X_label,2) + size(X_unlabel,2);
f_local_entropy = zeros(M,N,Num_train);
X = [X_label X_unlabel];
s_max = [3.16992500144231,4.64385618977472,5.61470984411520,6.33985000288459,6.91886323727454,7.40087943628214,...
         7.81378119121694,8.17492568250060,8.49585502688714,8.78463484555748,9.04712391211386,9.28771237954921];
s_min = [2.04605160356741,0.817454192119471,1.51525277423528,2.30276317669135,3.11458644831917,3.90522277416243,...
         4.64801519354613,5.33055401137885,5.94971068738255,6.50768912962657,7.00934343164725,7.46053832397253];
index = param.index;
local_size = 3:2:25;
parfor i = 1: Num_train
    f = reshape(X(:,i),[M N]);
    
%   en_max = max(f_en(:));
%   en_min = min(f_en(:));
%    f_local = (f_en -en_min) ./ (en_max - en_min);
%3*3
%     f_en = -local_entropy(f,3,3);
%     f_local = (f_en + 3.1699) ./ (3.1699 - 0.2877);%0-1

% 5*5
%     f_en = -local_entropy(f,5,5);    
%     f_local = (f_en + 4.6439) ./ (4.6439 - 0.8175);%0-1
    
%     %7*7
%     f_en = -local_entropy(f,7,7);    
%     f_local = (f_en + 5.5857) ./ (5.5857 - 1.5153);%0-1
    
%     %9*9
%     f_en = -local_entropy(f,9,9);    
%     f_local = (f_en + 6.3224) ./ (6.3224 - 2.3028);%0-1
    
%     %11*11
%     f_en = -local_entropy(f,11,11);    
%     f_local = (f_en + 6.9072) ./ (6.9072 - 3.1146);%0-1

%     %11*11
    f_en = -local_entropy(f,local_size(index),local_size(index));    
    f_local = (f_en + s_max(index)) ./ (s_max(index) - s_min(index));%0-1

    f_local_entropy(:,:,i) = f_local;
%     subplot(4,5,i),imshow(f_local,[ ]),title(num2str(i));
end

entropy = sum(f_local_entropy,3);%
ind_0 = find(entropy <= mean(entropy(:)));
ind_1 = find(entropy > mean(entropy(:)));
s_0 = sum(entropy(ind_0));
s_1 = sum(entropy(ind_1));
out.ind_0 = ind_0;
out.ind_1 = ind_1;
out.s_0 = s_0;
out.s_1 = s_1;
% prob = entropy / sum(entropy(:));
% prob = prob';

