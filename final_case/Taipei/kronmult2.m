function X = kronmult2(Q, X)
%KRONMULT2 Multiply a vector by a sequence of Kronecker factors.
%
% This function evaluates
%
%     X_out = kron(Q{1},kron(Q{2},...,Q{N}))*X_in
%
% without explicitly constructing the complete Kronecker matrix. It
% supports rectangular factors and 1-by-1 scalar factors. The result is
% consistent with MATLAB column-major ordering.
%
% INPUTS
%   Q : Cell array containing N matrix factors. If
%
%           Q{i} has size m_i-by-n_i,
%
%       the complete Kronecker matrix has size
%
%           prod(m_i)-by-prod(n_i).
%
%   X : Input column vector containing prod(n_i) elements.
%
% OUTPUT
%   X : Output column vector containing prod(m_i) elements.
%
% EXAMPLE
%   For Q = {A,B,C}, this function returns the same result as
%
%       kron(A,kron(B,C))*X
%
%   while avoiding explicit construction of the full Kronecker matrix.
%
% IMPLEMENTATION
%   The input vector is interpreted as an N-dimensional tensor. Each
%   Kronecker factor is applied along its corresponding tensor dimension
%   using reshape, permute, and ordinary matrix multiplication.

N = length(Q);
dims_out = cellfun(@(A) size(A,1), Q);
dims_in  = cellfun(@(A) size(A,2), Q); 

% Verify that the input size is compatible with all Kronecker factors.
if numel(X) ~= prod(dims_in)
    error('X size mismatch: expected %d rows but got %d.', prod(dims_in), numel(X));
end

% Reverse the factor and dimension order to match MATLAB column-major
% Kronecker ordering during the tensor operations below.
Q = Q(end:-1:1);
dims_in = dims_in(end:-1:1);
dims_out = dims_out(end:-1:1);

X = reshape(X, dims_in(:)');

% Apply one matrix factor along each tensor dimension.
for i = N:-1:1
    sz = size(X);

    % MATLAB removes trailing singleton dimensions. Restore them so that
    % the intermediate array always has exactly N tensor dimensions.
    if length(sz) < N
        sz = [sz, ones(1, N-length(sz))];
        X = reshape(X, sz);
    end

    % Move the active tensor dimension to the first position.
    order = [i, 1:i-1, i+1:N];

    % Ensure that all N dimensions are present before permuting the array.
    X = reshape(X, sz);  
    X = permute(X, order);

    % Collapse the remaining dimensions and apply the current factor.
    X = reshape(X, sz(i), []);
    X = Q{i} * X;
    sz(i) = size(Q{i},1);

    % Restore the tensor structure and return the dimensions to their
    % original order.
    newsz = [sz(i), sz([1:i-1, i+1:N])];
    if length(newsz) < N
        newsz = [newsz, ones(1, N-length(newsz))];
    end
    X = reshape(X, newsz);
    X = ipermute(X, order);
end

% Return the result in column-vector form.
X = X(:);
end
