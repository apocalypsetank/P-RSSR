function entropy_out = local_Shannonentropy(f,M,N)
% calculate Shannon entropy of the image with M*N window size.
% f - input image (gray image only)
% M and N local window size
% 
% 2016-10-14
if size(f,3) == 3
    f = rgb2gray(f);
end
f = padarray(f,[1 1],'symmetric');
[row,col] = size(f);
f = double(f);
p = im2col(f,[M,N],'sliding');
p = uint8(p);
row_new = row - M + 1;
col_new = col - N + 1;
entropy_out = zeros(1,row_new * col_new);
parfor i = 1:row_new * col_new
    t = p(:,i);
    entropy_out(i) = entropy(t);
end
entropy_out = (entropy_out - min(entropy_out(:))) ./ (max(entropy_out(:)) - min(entropy_out(:)));
% p_sum = repmat(sum(p) + eps,[M * N 1]);
% p = p ./ p_sum;
% p = -sum(p .* log2(p + eps));
entropy_out = reshape(entropy_out,[row_new,col_new]);