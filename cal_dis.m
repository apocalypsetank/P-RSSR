% function [distance,cal_time] = cal_dis(X,method)
function distance = cal_dis(X,method)
% 
% Input:
%     X: input samples 
%     method: 'col', each column of X is a sample
%             'row', each row of X is a sample.
% Output:
%   distance: distance between each sample.
% 
% 
% tic
if (nargin < 2)
    method = 'col';
end


if strcmp(method,'col');
    N = size(X,2);
    X2 = sum(X .^2 ,1);
    distance = repmat(X2,N,1) + repmat(X2',1,N) - 2 * X' * X;
    distance = distance - diag(diag(distance));
%     ind = distance < 0;
%     distance(ind) = 0;
else
    N = size(X,1);
    X2 = sum(X .^2 ,2);
    distance = repmat(X2,1,N) + repmat(X2',N,1) - 2 * X * X';
    distance = distance - diag(diag(distance));
%     ind = distance < 0;
%     distance(ind) = 0;
end
% cal_time = toc;