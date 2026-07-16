function [data_gb] = DW_sampler_new2(data_gb,x_cor,y_cor,z,sof_v,sof_h,nu_v,nu_h,bhp,alpha,y,Cs,sof_v_t,sof_h_t,M)
tic
jitter = 1e-6;
m  = M;
nv = size(data_gb,1);
nh = (size(data_gb,2))/m;
data_gb = reshape(data_gb,nv,nh,m);
[nv, nh, m] = size(data_gb);

%% residual kernels (same as your code)
disz = abs(z'-z);
Rv = Matern_R(nu_v,sof_v,disz); Rv(isnan(Rv)) = 1; Rv = Rv + jitter*eye(size(Rv));
disx = abs(x_cor'-x_cor); 
disy = abs(y_cor'-y_cor);
dish = sqrt(disx.^2+disy.^2);
Rh = Matern_R(nu_h,sof_h,dish); Rh(isnan(Rh)) = 1; Rh = Rh + jitter*eye(size(Rh));
%% phi (same as your code)
Cs = bhp * Cs;
[phi_t_Cs, omege_t_Cs] = eig(Cs);
eigvals = diag(omege_t_Cs);
[~, idx] = sort(eigvals, 'descend');
phi_t_Cs = phi_t_Cs(:, idx);
phiz = y.phiz;
phih = y.phih;
invO = diag(alpha);

%% masks / bookkeeping (same as your code)
n_total = nv*nh*m;
o_indk = false(n_total, nv);
lat_oindk = false(n_total, nv);
o_indk_non_lat = false(n_total, nv);
u_indk = false(n_total, nv);
uo_indk = false(n_total, nv);

o_indm = ~isnan(data_gb);
u_indm = ~o_indm;

data_gb = data_gb(:);

%% precompute per-depth block
invRv_lat = cell(nv, 1);
Rv_lat_1 = cell(nv, 1);
Rv_lat_total_lat = cell(nv, 1);
phiz_mini_1 = cell(nv, 1);
phiz_min_max = cell(nv, 1);
block_in_save = cell(nv,1);

for k = 1:nv
    block_in_up  = z >= z(k) - 0.02*sof_v_t;
    block_in_low = z <= z(k) + 0.02*sof_v_t;
    block_in_k = (block_in_up + block_in_low) >= 2;
    block_in_save{k} = block_in_k;

    block_out_k = ~block_in_k;

    temp = o_indm;
    if k > 1
        temp(1:k-1, :, :) = true;
    end
    temp(block_out_k, :, :) = false;
    o_indk(:, k) = temp(:);

    temp_lat = temp;
    temp_lat(k:end, :, :) = false;
    lat_oindk(:, k) = temp_lat(:);

    temp_non_lat = temp;
    temp_non_lat(1:k-1, :, :) = false;
    temp_non_lat(block_out_k, :, :) = false;
    o_indk_non_lat(:, k) = temp_non_lat(:);

    temp_u = u_indm;
    temp_u([1:k-1, k+1:nv], :, :) = false;
    u_indk(:, k) = temp_u(:);

    uo_indk(:, k) = any([u_indk(:, k), o_indk_non_lat(:, k)], 2);

    block_in_lat = block_in_k;
    block_in_lat(k:end) = false;

    Rv_lat = Rv(block_in_lat, block_in_lat);
    invRv_lat{k} = Rv_lat \ eye(sum(block_in_lat));

    Rv_lat_1{k} = Rv(block_in_k, block_in_k);
    Rv_lat_total_lat{k} = Rv(block_in_k, block_in_lat);

    phiz_mini_1{k} = phiz(block_in_lat, :);
    phiz_min_max{k} = phiz(block_in_k, :);
end

%% fixed matrices (same as your code)
inv_Cs = Cs\eye(m);
inv_Rh = Rh\eye(size(Rh));

Cs_Rh = kron(Cs, Rh);

phitCs_invCs_phitCs = phi_t_Cs' * inv_Cs * phi_t_Cs;
phih_invRh_phih = phih' * inv_Rh * phih;
K_ch = kron(phitCs_invCs_phitCs, phih_invRh_phih);
phi_t_Cs_phih = kron(phi_t_Cs, phih);      % (m*nh) x (m*mh)
A1 = (phi_t_Cs_phih)';                      % (m*mh) x (m*nh)
phihT_invRh  = phih' * inv_Rh;
phiCsT_invCs = phi_t_Cs' * inv_Cs;
nhm = nh*m;

%% ===========================
% Main loop
%% ===========================
for i = 1:nv
    % i
    %% (73) A 
    phizT_invRv = phiz_mini_1{i}' * invRv_lat{i};
    Kz = phizT_invRv * phiz_mini_1{i};
    A = invO + kron(K_ch, Kz);
    A = (A + A')/2 + 1e-12*eye(size(A,1));
    LA = chol(A,'lower');
    %% (74) mu_w 
    tmp = kronmult3({phiCsT_invCs, phihT_invRh, phizT_invRv}, data_gb(lat_oindk(:,i),1));
    mu_w = LA'\(LA\tmp(:));

    %% (76) uL (same, but faster kronmult3)
    Rv_invRv_plus_minus = Rv_lat_total_lat{i} * invRv_lat{i};
    utL      = kronmult3({phi_t_Cs, phih, phiz_min_max{i}}, mu_w);
    utL_min1 = kronmult3({phi_t_Cs, phih, phiz_mini_1{i}}, mu_w);
    tmp3 = kronmult3({eye(m), eye(nh), Rv_invRv_plus_minus}, (data_gb(lat_oindk(:,i),1) - utL_min1));
    uL = utL(:) + tmp3(:);

    %% Phi_z_cond (same as your code: uses residual Rv_invRv_plus_minus)
    Phi_z_cond = phiz_min_max{i} - Rv_invRv_plus_minus * phiz_mini_1{i};
    %% condRv_plus_minus (same)
    condRv = Rv_lat_1{i} - Rv_invRv_plus_minus * (Rv_lat_total_lat{i})';
    %% local indices (same meaning)
    L3 = false(nv,nh,m);
    L3(block_in_save{i},:,:) = true;
    idxL = find(L3(:));   % MATLAB ordering: z fastest, then hm
    o_mask_L = ismember(idxL, find(o_indk_non_lat(:,i)));
    u_mask_L = ismember(idxL, find(u_indk(:,i)));
    cols_u = find(u_mask_L);     % local column indices
    cols_o = find(o_mask_L);
    nu_loc = numel(cols_u);
    no_loc = numel(cols_o);
    if nu_loc == 0
        continue;
    end
    if no_loc == 0
        data_gb(idxL(cols_u),1) = uL(cols_u);
        continue;
    end

    %% ===========================
    nbz = size(Phi_z_cond,1);

    cols_need = [cols_u; cols_o];
    Tneed = build_kron_columns_fast(A1, Phi_z_cond.', cols_need, nbz, nhm);  % nAlpha x (nu+no)

    Zneed = LA \ Tneed;                % = A^{-1/2} * Tneed
    Gneed = Zneed' * Zneed;            % = Tneed' * A^{-1} * Tneed
    Gneed = (Gneed + Gneed')/2;

    term1_uu = Gneed(1:nu_loc, 1:nu_loc);
    term1_uo = Gneed(1:nu_loc, nu_loc+1:end);
    term1_oo = Gneed(nu_loc+1:end, nu_loc+1:end);

    %% ===========================
    % term2 block: ttt2 = kron(Cs_Rh, condRv)   (same as your ttt2)
    % but compute only blocks using MATLAB ordering:
    % local col = (hm-1)*nbz + z
    %% ===========================
    [term2_uu, term2_uo, term2_oo] = kron_blocks_MATLAB(Cs_Rh, condRv, cols_u, cols_o, nbz);

    %% CL blocks (same as CL = term1 + term2)
    Suu = term1_uu + term2_uu;
    Suo = term1_uo + term2_uo;
    Soo = term1_oo + term2_oo;
    Suu = (Suu + Suu')/2 + jitter*eye(size(Suu,1));
    Soo = (Soo + Soo')/2 + jitter*eye(size(Soo,1));

    %% conditional update (same math)
    mu_u = uL(cols_u);
    mu_o = uL(cols_o);
    y_o  = data_gb(idxL(cols_o),1);

    Loo = chol(Soo,'lower');
    tmpS = (Suo / Loo') / Loo;

    Eu = mu_u + tmpS*(y_o - mu_o);
    Cu = Suu - tmpS*Suo';
    Cu = (Cu + Cu')/2 + jitter*eye(size(Cu,1));

    %% sample (same)
    Lu = chol(Cu, 'lower');

    % ======= KEEP EXACTLY YOUR ORIGINAL WRITEBACK (math-same assumption) =======
    nou = sum(u_mask_L);  %#ok<NASGU>
    data_gb(u_indk(:,i), 1) = Eu + Lu*randn(nu_loc,1);

    % ======= SAFER WRITEBACK (recommended, but would change your indexing assumption) =======
    % data_gb(idxL(cols_u),1) = Eu + Lu*randn(nu_loc,1);

end

data_gb = reshape(data_gb,nv,nh*m);
time_DW = toc
end 

%% =====================================================================
% Build selected columns of kron(A1,B1) using MATLAB ordering:
% col = (hm-1)*nbz + z
%   hm in 1..nhm, z in 1..nbz
% Column = kron(A1(:,hm), B1(:,z))
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

    % compress duplicates (cheap)
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
% Fast 3-factor Kronecker MV without forming kron
% y = (A ⊗ B ⊗ C) * x  (MATLAB column-major consistent)
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

    % C along dim1
    X = reshape(X, [nC, nB*nA*r]);
    X = C * X;
    X = reshape(X, [mC, nB, nA, r]);

    % B along dim2
    X = permute(X, [2 1 3 4]);
    X = reshape(X, [nB, mC*nA*r]);
    X = B * X;
    X = reshape(X, [mB, mC, nA, r]);
    X = permute(X, [2 1 3 4]);

    % A along dim3
    X = permute(X, [3 1 2 4]);
    X = reshape(X, [nA, mC*mB*r]);
    X = A * X;
    X = reshape(X, [mA, mC, mB, r]);
    X = permute(X, [2 3 1 4]);

    y = reshape(X, [mC*mB*mA, r]);
end
