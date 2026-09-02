function [data_gb] = DW_sampler_new2(data_gb,x_cor,y_cor,z,sof_v,sof_h,nu_v,nu_h,bhp,alpha,y,Cs,sof_v_t,sof_h_t,M)
%DW_SAMPLER_NEW2 Sequentially impute missing non-lattice observations.
%
% This function implements the depth-wise (DW) sampler used for the
% Baytown non-lattice dataset. Missing observations are completed one depth
% at a time. At each depth, the sampler first uses previously completed
% depths to update the latent trend weights and then conditions the missing
% entries on the available observations inside a local vertical window.
%
% INPUTS
%   data_gb : Data matrix of size nv-by-(nh*M). Columns are grouped by
%             parameter, with nh sounding locations in each parameter
%             block. Missing observations are represented by NaN.
%   x_cor   : Horizontal x-coordinates of the nh sounding locations.
%   y_cor   : Horizontal y-coordinates of the nh sounding locations.
%   z       : Vertical coordinates of the nv depth points.
%   sof_v   : Vertical SOF of the Matern residual component.
%   sof_h   : Horizontal SOF of the Matern residual component.
%   nu_v    : Vertical Matern smoothness parameter.
%   nu_h    : Horizontal Matern smoothness parameter.
%   bhp     : Residual covariance amplitude.
%   alpha   : Prior precision vector of the latent trend weights.
%   y       : Model structure containing y.phiz and y.phih, the retained
%             vertical and horizontal trend bases at the training points.
%   Cs      : M-by-M cross-parameter covariance matrix.
%   sof_v_t : Vertical SOF of the Gaussian trend. It determines the local
%             vertical window used in the sequential sampler.
%   sof_h_t : Horizontal trend SOF retained for interface compatibility;
%             it is not used directly in the current implementation.
%   M       : Number of modeled geotechnical parameters.
%
% OUTPUT
%   data_gb : Completed nv-by-(nh*M) matrix in the original input layout.
%             Observed entries are retained, whereas missing entries are
%             replaced by conditional samples or conditional means.
%
% ARRAY ORDER
%   Internally, data_gb is reshaped to [nv,nh,M]. MATLAB linear indexing
%   therefore follows
%
%       depth -> sounding/location -> parameter.
%
% SAMPLING OUTLINE
%   For each depth k:
%     1. Define a local vertical window around z(k).
%     2. Use previously completed depths as lattice conditioning data.
%     3. Update the posterior mean and precision of the trend weights.
%     4. Construct the local Gaussian mean and covariance.
%     5. Condition missing entries on locally observed entries.
%     6. Draw and write back the missing values.
tic
jitter = 1e-6;
m  = M;
nv = size(data_gb,1);
nh = (size(data_gb,2))/m;
% Convert the storage matrix into an explicit
% depth-by-sounding-by-parameter array.
data_gb = reshape(data_gb,nv,nh,m);
[nv, nh, m] = size(data_gb);

%% ===== 1) Construct the residual correlation matrices =====

% Vertical Matern residual correlation.
disz = abs(z'-z);
Rv = Matern_R(nu_v,sof_v,disz); Rv(isnan(Rv)) = 1; Rv = Rv + jitter*eye(size(Rv));

% Horizontal Euclidean distances and Matern residual correlation.
disx = abs(x_cor'-x_cor); 
disy = abs(y_cor'-y_cor);
dish = sqrt(disx.^2+disy.^2);
Rh = Matern_R(nu_h,sof_h,dish); Rh(isnan(Rh)) = 1; Rh = Rh + jitter*eye(size(Rh));

%% ===== 2) Prepare the trend bases and prior precision =====

% Include the residual amplitude in the cross-parameter covariance.
Cs = bhp * Cs;

% The eigenvectors define the cross-parameter trend basis. Scaling Cs by
% bhp changes its eigenvalues but not its eigenvectors.
[phi_t_Cs, omege_t_Cs] = eig(Cs);
eigvals = diag(omege_t_Cs);
[~, idx] = sort(eigvals, 'descend');
phi_t_Cs = phi_t_Cs(:, idx);

% The vertical and horizontal trend bases are constructed before this
% function is called. alpha contains the associated prior precisions.
phiz = y.phiz;
phih = y.phih;
invO = diag(alpha);

%% ===== 3) Initialize observation masks and bookkeeping arrays =====

n_total = nv*nh*m;

% All local conditioning entries at each sequential depth step.
o_indk = false(n_total, nv);

% Previously completed lattice entries above the current depth.
lat_oindk = false(n_total, nv);

% Originally observed non-lattice entries inside the local window.
o_indk_non_lat = false(n_total, nv);

% Missing entries located at the current depth only.
u_indk = false(n_total, nv);

% Union of local missing and observed entries.
uo_indk = false(n_total, nv);

% Observation and missing-data masks in [depth,sounding,parameter] form.
o_indm = ~isnan(data_gb);
u_indm = ~o_indm;

% Switch to vector storage while preserving the documented ordering.
data_gb = data_gb(:);

%% ===== 4) Precompute the depth-dependent local blocks =====
invRv_lat = cell(nv, 1);
Rv_lat_1 = cell(nv, 1);
Rv_lat_total_lat = cell(nv, 1);
phiz_mini_1 = cell(nv, 1);
phiz_min_max = cell(nv, 1);
block_in_save = cell(nv,1);

for k = 1:nv
    % Define a local window centered at z(k). The current implementation
    % uses plus or minus 0.02 times the vertical trend SOF.
    block_in_up  = z >= z(k) - 0.02*sof_v_t;
    block_in_low = z <= z(k) + 0.02*sof_v_t;
    block_in_k = (block_in_up + block_in_low) >= 2;
    block_in_save{k} = block_in_k;

    block_out_k = ~block_in_k;

    % All entries above depth k have already been completed by the sampler
    % and may therefore be used as conditioning data.
    temp = o_indm;
    if k > 1
        temp(1:k-1, :, :) = true;
    end
    temp(block_out_k, :, :) = false;
    o_indk(:, k) = temp(:);

    % Previously completed lattice entries inside the local window.
    temp_lat = temp;
    temp_lat(k:end, :, :) = false;
    lat_oindk(:, k) = temp_lat(:);

    % Originally observed non-lattice entries at the current or later
    % depths inside the local window.
    temp_non_lat = temp;
    temp_non_lat(1:k-1, :, :) = false;
    temp_non_lat(block_out_k, :, :) = false;
    o_indk_non_lat(:, k) = temp_non_lat(:);

    % Missing entries that must be completed at the current depth.
    temp_u = u_indm;
    temp_u([1:k-1, k+1:nv], :, :) = false;
    u_indk(:, k) = temp_u(:);

    uo_indk(:, k) = any([u_indk(:, k), o_indk_non_lat(:, k)], 2);

    % Retain only completed depths above k for the lattice update.
    block_in_lat = block_in_k;
    block_in_lat(k:end) = false;

    % Cache the residual-correlation and trend-basis blocks required at
    % this depth step.
    Rv_lat = Rv(block_in_lat, block_in_lat);
    invRv_lat{k} = Rv_lat \ eye(sum(block_in_lat));

    Rv_lat_1{k} = Rv(block_in_k, block_in_k);
    Rv_lat_total_lat{k} = Rv(block_in_k, block_in_lat);

    phiz_mini_1{k} = phiz(block_in_lat, :);
    phiz_min_max{k} = phiz(block_in_k, :);
end

%% ===== 5) Precompute matrices that do not change with depth =====

% Residual precision matrices in the parameter and horizontal directions.
inv_Cs = Cs\eye(m);
inv_Rh = Rh\eye(size(Rh));

% Joint parameter-horizontal residual covariance.
Cs_Rh = kron(Cs, Rh);

% Components of the posterior precision of the latent trend weights.
phitCs_invCs_phitCs = phi_t_Cs' * inv_Cs * phi_t_Cs;
phih_invRh_phih = phih' * inv_Rh * phih;
K_ch = kron(phitCs_invCs_phitCs, phih_invRh_phih);

% Kronecker factors reused during the sequential depth updates.
phi_t_Cs_phih = kron(phi_t_Cs, phih);      % (m*nh) x (m*mh)
A1 = (phi_t_Cs_phih)';                      % (m*mh) x (m*nh)
phihT_invRh  = phih' * inv_Rh;
phiCsT_invCs = phi_t_Cs' * inv_Cs;
nhm = nh*m;

%% ===== 6) Sequentially sample the missing data at each depth =====

for i = 1:nv
    %% Step 6.1: posterior precision of the latent trend weights

    % Use the completed lattice entries above the current depth to update
    % the latent trend weights. A is the posterior precision matrix.
    phizT_invRv = phiz_mini_1{i}' * invRv_lat{i};
    Kz = phizT_invRv * phiz_mini_1{i};
    A = invO + kron(K_ch, Kz);
    A = (A + A')/2 + 1e-12*eye(size(A,1));
    LA = chol(A,'lower');

    %% Step 6.2: posterior mean of the latent trend weights

    tmp = kronmult3({phiCsT_invCs, phihT_invRh, phizT_invRv}, data_gb(lat_oindk(:,i),1));
    mu_w = LA'\(LA\tmp(:));

    %% Step 6.3: local mean before non-lattice conditioning

    % Propagate the posterior trend mean into the local window and add the
    % residual contribution conditioned on the completed lattice depths.
    Rv_invRv_plus_minus = Rv_lat_total_lat{i} * invRv_lat{i};
    utL      = kronmult3({phi_t_Cs, phih, phiz_min_max{i}}, mu_w);
    utL_min1 = kronmult3({phi_t_Cs, phih, phiz_mini_1{i}}, mu_w);
    tmp3 = kronmult3({eye(m), eye(nh), Rv_invRv_plus_minus}, (data_gb(lat_oindk(:,i),1) - utL_min1));
    uL = utL(:) + tmp3(:);

    % Vertical trend basis after conditioning on the completed depths.
    Phi_z_cond = phiz_min_max{i} - Rv_invRv_plus_minus * phiz_mini_1{i};

    % Vertical residual covariance after conditioning on completed depths.
    condRv = Rv_lat_1{i} - Rv_invRv_plus_minus * (Rv_lat_total_lat{i})';

    %% Step 6.4: identify local missing and observed entries

    L3 = false(nv,nh,m);
    L3(block_in_save{i},:,:) = true;
    idxL = find(L3(:));   % Local-to-global map; depth varies fastest.
    o_mask_L = ismember(idxL, find(o_indk_non_lat(:,i)));
    u_mask_L = ismember(idxL, find(u_indk(:,i)));
    cols_u = find(u_mask_L);     % Local indices of missing entries.
    cols_o = find(o_mask_L);
    nu_loc = numel(cols_u);
    no_loc = numel(cols_o);

    % Nothing is missing at this depth.
    if nu_loc == 0
        continue;
    end

    % Preserve the original behavior: if no locally observed entry is
    % available, fill the missing entries with the local conditional mean.
    if no_loc == 0
        data_gb(idxL(cols_u),1) = uL(cols_u);
        continue;
    end

    %% Step 6.5: covariance from uncertainty in the trend weights

    nbz = size(Phi_z_cond,1);

    cols_need = [cols_u; cols_o];
    Tneed = build_kron_columns_fast(A1, Phi_z_cond.', cols_need, nbz, nhm);  % nAlpha x (nu+no)

    Zneed = LA \ Tneed;                % Equivalent to A^(-1/2)*Tneed.
    Gneed = Zneed' * Zneed;            % Equivalent to Tneed'*A^(-1)*Tneed.
    Gneed = (Gneed + Gneed')/2;

    term1_uu = Gneed(1:nu_loc, 1:nu_loc);
    term1_uo = Gneed(1:nu_loc, nu_loc+1:end);
    term1_oo = Gneed(nu_loc+1:end, nu_loc+1:end);

    %% Step 6.6: covariance from the conditioned residual field

    % The complete residual term is kron(Cs_Rh,condRv). Only the required
    % missing-missing, missing-observed, and observed-observed blocks are
    % constructed. The local ordering is
    %
    %   local column = (sounding-parameter index - 1)*nbz + depth index.
    [term2_uu, term2_uo, term2_oo] = kron_blocks_MATLAB(Cs_Rh, condRv, cols_u, cols_o, nbz);

    %% Step 6.7: combine the trend and residual covariance blocks

    Suu = term1_uu + term2_uu;
    Suo = term1_uo + term2_uo;
    Soo = term1_oo + term2_oo;
    Suu = (Suu + Suu')/2 + jitter*eye(size(Suu,1));
    Soo = (Soo + Soo')/2 + jitter*eye(size(Soo,1));

    %% Step 6.8: condition missing entries on local observations

    % Partition the local Gaussian distribution into missing (u) and
    % observed (o) components.
    mu_u = uL(cols_u);
    mu_o = uL(cols_o);
    y_o  = data_gb(idxL(cols_o),1);

    Loo = chol(Soo,'lower');
    tmpS = (Suo / Loo') / Loo;

    Eu = mu_u + tmpS*(y_o - mu_o);
    Cu = Suu - tmpS*Suo';
    Cu = (Cu + Cu')/2 + jitter*eye(size(Cu,1));

    %% Step 6.9: draw and write back the missing entries

    Lu = chol(Cu, 'lower');

    % Original writeback used by the current model. It assumes that the
    % global u_indk ordering is consistent with the local cols_u ordering.
    nou = sum(u_mask_L);  %#ok<NASGU>
    data_gb(u_indk(:,i), 1) = Eu + Lu*randn(nu_loc,1);

    % Alternative explicit local-to-global writeback retained for reference.
    % Enabling it would change the executable behavior and should therefore
    % be verified separately before replacing the original line.
    % data_gb(idxL(cols_u),1) = Eu + Lu*randn(nu_loc,1);

end

% Restore the original nv-by-(nh*M) storage format.
data_gb = reshape(data_gb,nv,nh*m);
time_DW = toc
end 

%% =====================================================================
% BUILD_KRON_COLUMNS_FAST constructs selected columns of kron(A1,B1).
%
% INPUTS
%   A1   : First Kronecker factor.
%   B1   : Second Kronecker factor.
%   cols : Requested column indices in MATLAB Kronecker ordering.
%   nbz  : Number of B1 columns associated with the depth dimension.
%   nhm  : Number of A1 columns associated with the combined
%          sounding-parameter dimension.
%
% OUTPUT
%   Tsel : Requested columns of kron(A1,B1), constructed without forming
%          the complete Kronecker matrix.
%
% MATLAB column ordering:
%   col = (hm-1)*nbz + z, where hm=1,...,nhm and z=1,...,nbz.
%   The corresponding full column is kron(A1(:,hm),B1(:,z)).
%% =====================================================================
function Tsel = build_kron_columns_fast(A1, B1, cols, nbz, nhm)
    pA = size(A1,1);
    pB = size(B1,1);
    ncols = numel(cols);
    z_col  = mod(cols-1, nbz) + 1;
    hm_col = floor((cols-1)./nbz) + 1;

    if any(hm_col<1 | hm_col>nhm)
        error('build_kron_columns_fast: hm_col out of range.');
    end

    % Construct each unique requested column once and then restore any
    % duplicate requests using the index map ic.
    key = hm_col(:) + nhm*(z_col(:)-1);
    [keyU,~,ic] = unique(key,'stable');
    TselU = zeros(pA*pB, numel(keyU));
    for k = 1:numel(keyU)
        z  = floor((keyU(k)-1)/nhm) + 1;
        hm = keyU(k) - (z-1)*nhm;
        va = A1(:,hm);
        vb = B1(:,z);
        TselU(:,k) = kron(va, vb);
    end
    Tsel = TselU(:,ic);
end
%% =====================================================================
% KRON_BLOCKS_MATLAB extracts covariance blocks from kron(CsRh,Rz).
%
% INPUTS
%   CsRh   : Joint parameter-horizontal covariance matrix.
%   Rz     : Vertical covariance matrix in the local depth window.
%   cols_u : Local indices of missing entries.
%   cols_o : Local indices of observed entries.
%   nbz    : Number of vertical entries in the local window.
%
% OUTPUTS
%   Kuu : Missing-missing covariance block.
%   Kuo : Missing-observed covariance block.
%   Koo : Observed-observed covariance block.
%
% The blocks are assembled with Hadamard products, avoiding construction
% of the complete Kronecker matrix.
%% =====================================================================
function [Kuu, Kuo, Koo] = kron_blocks_MATLAB(CsRh, Rz, cols_u, cols_o, nbz)

    zu = mod(cols_u-1, nbz) + 1;
    hu = floor((cols_u-1)/nbz) + 1;

    zo = mod(cols_o-1, nbz) + 1;
    ho = floor((cols_o-1)/nbz) + 1;

    Kuu = CsRh(hu, hu) .* Rz(zu, zu);
    Kuo = CsRh(hu, ho) .* Rz(zu, zo);
    Koo = CsRh(ho, ho) .* Rz(zo, zo);
end


%% =====================================================================
% KRONMULT3 evaluates a three-factor Kronecker matrix-vector product.
%
%   y = (A kron B kron C)*x
%
% The complete Kronecker matrix is never formed. The implementation uses
% MATLAB column-major ordering and supports multiple right-hand sides.
%
% INPUTS
%   Ops        : Cell array {A,B,C} containing the three matrix factors.
%   x          : Input vector or matrix with nA*nB*nC rows.
%   transFlags : Optional three-element logical vector. A true entry
%                transposes the corresponding factor before multiplication.
%
% OUTPUT
%   y : Product with mA*mB*mC rows.
%% =====================================================================
function y = kronmult3(Ops, x, transFlags)

    if nargin < 3 || isempty(transFlags)
        transFlags = [0 0 0];
    end

    A = Ops{1}; B = Ops{2}; C = Ops{3};
    if transFlags(1), A = A'; end
    if transFlags(2), B = B'; end
    if transFlags(3), C = C'; end

    [mA,nA] = size(A);
    [mB,nB] = size(B);
    [mC,nC] = size(C);

    if isvector(x), x = x(:); end
    r = size(x,2);

    if size(x,1) ~= nA*nB*nC
        error("kronmult3: x size mismatch. Need %d rows, got %d.", nA*nB*nC, size(x,1));
    end

    X = reshape(x, [nC, nB, nA, r]);

    % Apply C along the first tensor dimension.
    X = reshape(X, [nC, nB*nA*r]);
    X = C * X;
    X = reshape(X, [mC, nB, nA, r]);

    % Apply B along the second tensor dimension.
    X = permute(X, [2 1 3 4]);
    X = reshape(X, [nB, mC*nA*r]);
    X = B * X;
    X = reshape(X, [mB, mC, nA, r]);
    X = permute(X, [2 1 3 4]);

    % Apply A along the third tensor dimension.
    X = permute(X, [3 1 2 4]);
    X = reshape(X, [nA, mC*mB*r]);
    X = A * X;
    X = reshape(X, [mA, mC, mB, r]);
    X = permute(X, [2 3 1 4]);

    y = reshape(X, [mC*mB*mA, r]);
end
