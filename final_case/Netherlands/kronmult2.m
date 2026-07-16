function X = kronmult2(Q, X)
% 通用 Kronecker Product (支援非方陣與1x1 scalar)
% 與 MATLAB kron(Q1,Q2,...)*X 完全一致
% X 必須是 column vector

N = length(Q);
dims_out = cellfun(@(A) size(A,1), Q);
dims_in  = cellfun(@(A) size(A,2), Q); 

% 檢查 X 尺寸
if numel(X) ~= prod(dims_in)
    error('X size mismatch: expected %d rows but got %d.', prod(dims_in), numel(X));
end

% 反轉 Q 和 dims
Q = Q(end:-1:1);
dims_in = dims_in(end:-1:1);
dims_out = dims_out(end:-1:1);

X = reshape(X, dims_in(:)');


for i = N:-1:1
    sz = size(X);
    if length(sz) < N
        sz = [sz, ones(1, N-length(sz))];
        X = reshape(X, sz);
    end

    order = [i, 1:i-1, i+1:N];
    %permute 前都先補成 N 維
    X = reshape(X, sz);  
    X = permute(X, order);

    X = reshape(X, sz(i), []);
    X = Q{i} * X;
    sz(i) = size(Q{i},1);

  
    newsz = [sz(i), sz([1:i-1, i+1:N])];
    if length(newsz) < N
        newsz = [newsz, ones(1, N-length(newsz))];
    end
    X = reshape(X, newsz);
    X = ipermute(X, order);
end

X = X(:);
end
