function entropy_out = local_entropy(f,M,N)
% calculate the local entropy of image f with the window size M*N
% f:input image (gray image only)
% M: window size
% N: window size
% 
if size(f,3) == 3
    f = rgb2gray(f);
end
M1 = floor(M / 2);
N1 = floor(N / 2);
f = padarray(f,[M1 N1],'symmetric');
[row,col] = size(f);
f = double(f);
p = im2col(f,[M,N],'sliding');
row_new = row - M + 1;
col_new = col - N + 1;
p_sum = repmat(sum(p) + eps,[M * N 1]);
p = p ./ p_sum;
p = -sum(p .* log2(p + eps));
entropy_out = reshape(p,[row_new,col_new]);
