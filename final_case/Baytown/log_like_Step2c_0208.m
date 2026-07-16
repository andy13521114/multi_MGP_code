function  [log_like] = log_like_Step2c_0208(data,nan_ind,x,y,z,sof_v,sof_h,nu_v,nu_h,bhp,alpha,Y,Cs,ahp)
% log_like_Step2c_0206 (FAST, EXACTLY SAME MATH as your original 0206)
% ============================================================
% Key speed fixes (ALL exact):
%  1) NEVER form Sigma_mm = kron(Cs_mm,kron(Rh_mm,Rv_mm)).
%     Use exact Kronecker Cholesky solves: (Cs⊗Rh⊗Rv)\B
%  2) Vectorize Sigma_om, Sigma_oo (no loops).
%  3) Cache Phi_o/Phi_m and index decoding (geometry/basis unchanged across iTMCMC calls).
%  4) Avoid inv(); use chol solves everywhere.
% ============================================================

jitter = 1e-6;

%% ===== 0) Cs eig (same) =====
[phi_t_Cs, D] = eig(Cs);
eigvals = diag(D);
[~, idx] = sort(eigvals, 'descend');
phi_t_Cs = phi_t_Cs(:, idx);

%% ===== 1) Define CPT lattice subset (same) =====
Cs = bhp * Cs;

% holes_cpt = 6:9;
% pid_cpt   = [3 4 5];
 % holes_cpt = 1:13;
 % pid_cpt   = [ 4 7 8];
 holes_cpt = 1:6;
 pid_cpt   = [1 2];
M_all = size(Cs,1);
M_CPT = numel(pid_cpt);

nh_all = size(data,2)/M_all;
nh_lat = numel(holes_cpt);

cols_lat = [];
for p = pid_cpt
    cols_lat = [cols_lat, (p-1)*nh_all + holes_cpt];
end

nan_ind_CPT = nan_ind(:, cols_lat);
data_CPT    = data(:, cols_lat);

nan_r_lat   = sum(nan_ind_CPT,2);
empty_depth = find(nan_r_lat==nh_lat);

nan_ind_CPT(empty_depth,:) = [];
data_CPT(empty_depth,:)    = [];
z(empty_depth)             = [];
nan_r_lat(empty_depth)     = [];

nv = size(data_CPT,1);

z_minus = (nan_r_lat==0);
z_plus  = true(nv,1);

%% ===== 2) Global masks (same) =====
lat_mask = false(size(nan_ind));
lat_mask(z_minus, cols_lat) = true;

obs_mask  = (nan_ind == 0);
o_mask    = obs_mask & ~lat_mask;

o_idx  = find(o_mask(:));
Nobs   = numel(o_idx);

m_mask_full = false(size(data));
m_mask_full(z_minus, cols_lat) = true;
m_idx = find(m_mask_full(:));
Nm    = numel(m_idx);

%% ===== 3) Build Rv, Rh =====
y_lat = y(holes_cpt);
x_lat = x(holes_cpt);

disz = abs(z'-z);
R_v = Matern_R(nu_v, sof_v, disz);  R_v(isnan(R_v)) = 1;
R_v = (R_v + R_v')/2 + jitter*eye(size(R_v));

disx = abs(x'-x); disy = abs(y'-y);
dish = sqrt(disx.^2 + disy.^2);
R_h = Matern_R(nu_h, sof_h, dish);  R_h(isnan(R_h)) = 1;
R_h = (R_h + R_h')/2 + jitter*eye(size(R_h));

disx_lat = abs(x_lat'-x_lat);
disy_lat = abs(y_lat'-y_lat);
dish_lat = sqrt(disx_lat.^2 + disy_lat.^2);
R_h_lat  = Matern_R(nu_h, sof_h, dish_lat); R_h_lat(isnan(R_h_lat)) = 1;
R_h_lat  = (R_h_lat + R_h_lat')/2 + jitter*eye(size(R_h_lat));

% exact solves (avoid inv)
Lh_lat = chol(R_h_lat,'lower');
Lv_m   = chol(R_v(z_minus,z_minus),'lower');
invRh      = Lh_lat'\(Lh_lat\eye(nh_lat));
invRv_minus = Lv_m'\(Lv_m\eye(sum(z_minus)));

%% ===== 4) Basis Phi =====
phih     = Y.phih;
phih_lat = phih(holes_cpt,:);

phiz = Y.phiz;
phiz(empty_depth,:) = [];

mz = size(phiz,2);
mh = size(phih,2);

invO = diag(alpha);

%% ===== 5) Build w posterior: C, mu (same math, faster numerics) =====
Cs_lat = Cs(pid_cpt,pid_cpt);
Cs_lat = (Cs_lat + Cs_lat')/2;
Lcsl   = chol(Cs_lat + 1e-12*eye(M_CPT),'lower');
inv_Cs_lat = Lcsl'\(Lcsl\eye(M_CPT));

phi_t_Cs_lat      = phi_t_Cs(pid_cpt,:);
phi_t_Cs_nlat_lat = phi_t_Cs(pid_cpt,:);

phih_invRh_phih               = phih_lat' * (invRh * phih_lat);
phiz_minus_invRv_phiz_minus   = phiz(z_minus,:)' * (invRv_minus * phiz(z_minus,:));
phitCs_invCs_phitCs           = phi_t_Cs_lat' * inv_Cs_lat * phi_t_Cs_lat;

A_w = invO + kron(phitCs_invCs_phitCs, kron(phih_invRh_phih, phiz_minus_invRv_phiz_minus));
A_w = (A_w + A_w')/2 + 1e-12*eye(size(A_w));
LAw = chol(A_w,'lower');

C = LAw'\(LAw\eye(size(A_w,1)));
C = (C + C')/2;

tmp = data_CPT(z_minus,:);
y_minus = tmp(:);

phihT_invRh       = phih_lat' * invRh;
phizT_invRv_minus = phiz(z_minus,:)' * invRv_minus;

phiCsT_invCs = (phi_t_Cs_nlat_lat' * inv_Cs_lat);

tmp_mu = kron(phihT_invRh, phizT_invRv_minus) ...
       * reshape(y_minus, sum(z_minus)*nh_lat, M_CPT) ...
       * phiCsT_invCs';
mu = C * tmp_mu(:);

%% ===== 6) Mean Ey_o (same) =====
E_t_minus = kron(phih_lat, phiz(z_minus,:)) ...
          * reshape(mu, mz*mh, M_all) ...
          * phi_t_Cs_lat';
E_t_minus = E_t_minus(:);

E_t_full = kron(phi_t_Cs, kron(phih, phiz)) * mu;

Rv_invRv_plus_minus = R_v(z_plus, z_minus) * invRv_minus;
Rh_invRh_plus_minus = R_h(:, holes_cpt) * invRh;
Cs_invCs_plus_minus = Cs(:, pid_cpt) * inv_Cs_lat;

tmp3 = kron(Rh_invRh_plus_minus, Rv_invRv_plus_minus) ...
     * reshape(y_minus - E_t_minus, sum(z_minus)*nh_lat, M_CPT) ...
     * Cs_invCs_plus_minus';
tmp3 = tmp3(:);

E_yplus_minus = E_t_full + tmp3;

yvec     = data(:);
y_plus_o = yvec(o_idx);
Ey_o     = E_yplus_minus(o_idx);

%% ============================================================
% 7) O_o = Phi_o - Sigma_om * (Sigma_mm \ Phi_m)
% 8) Sigma_oo_cond = Sigma_oo - Sigma_mo' * (Sigma_mm \ Sigma_mo)
%      where Sigma_mm = Cs_mm ⊗ Rh_mm ⊗ Rv_mm  (CPT lattice minus only)
%      DO NOT FORM Sigma_mm explicitly
%% ============================================================

persistent CACHE
key = fast_cache_key(nv,nh_all,M_all,holes_cpt,pid_cpt,z_minus,o_idx,size(phiz,2),size(phih,2),phi_t_Cs(1,1));

if isempty(CACHE) || ~isfield(CACHE,'key') || ~strcmp(CACHE.key, key)
    CACHE = struct();
    CACHE.key = key;

    % decode o
    [iz_o, hid_o, pid_o] = decode_global_index(o_idx, nv, nh_all);

    % build m triple list
    iz_m0  = find(z_minus);
    hid_m0 = holes_cpt(:);
    pid_m0 = pid_cpt(:);

    [IZ, IH, IP] = ndgrid(1:numel(iz_m0), 1:numel(hid_m0), 1:numel(pid_m0));
    iz_m  = iz_m0(IZ(:));
    hid_m = hid_m0(IH(:));
    pid_m = pid_m0(IP(:));

    if numel(iz_m) ~= Nm
        error("Nm mismatch: numel(m_idx)=%d, but cross-product Nm=%d", Nm, numel(iz_m));
    end

    CACHE.iz_o  = iz_o;  CACHE.hid_o = hid_o;  CACHE.pid_o = pid_o;
    CACHE.iz_m  = iz_m;  CACHE.hid_m = hid_m;  CACHE.pid_m = pid_m;

    % build Phi_o, Phi_m once (dominant cost in your original)
    Kcs = size(phi_t_Cs,2);
    Kh  = size(phih,2);
    Kz  = size(phiz,2);
    K   = Kcs * Kh * Kz;

    Phi_o = zeros(Nobs, K);
    for ii = 1:Nobs
        row_cs = phi_t_Cs(pid_o(ii), :);
        row_h  = phih(hid_o(ii), :);
        row_z  = phiz(iz_o(ii), :);
        Phi_o(ii,:) = kron(row_cs, kron(row_h, row_z));
    end

    Phi_m = zeros(Nm, K);
    for kk = 1:Nm
        row_cs = phi_t_Cs(pid_m(kk), :);
        row_h  = phih(hid_m(kk), :);
        row_z  = phiz(iz_m(kk), :);
        Phi_m(kk,:) = kron(row_cs, kron(row_h, row_z));
    end

    CACHE.Phi_o = Phi_o;
    CACHE.Phi_m = Phi_m;
end

iz_o  = CACHE.iz_o;   hid_o = CACHE.hid_o;   pid_o = CACHE.pid_o;
iz_m  = CACHE.iz_m;   hid_m = CACHE.hid_m;   pid_m = CACHE.pid_m;
Phi_o = CACHE.Phi_o;
Phi_m = CACHE.Phi_m;

% --- Kronecker blocks for Sigma_mm (exact) ---
Cs_mm = (Cs(pid_cpt,pid_cpt) + Cs(pid_cpt,pid_cpt)')/2 + 1e-10*eye(M_CPT);
Rh_mm = (R_h_lat + R_h_lat')/2 + 1e-10*eye(nh_lat);
Rv_mm = (R_v(z_minus,z_minus) + R_v(z_minus,z_minus)')/2 + 1e-10*eye(sum(z_minus));

Lc_mm = chol(Cs_mm,'lower');
Lh_mm = chol(Rh_mm,'lower');
Lv_mm = chol(Rv_mm,'lower');

% --- Sigma_om / Sigma_oo vectorized (NO LOOPS) ---
Sigma_om = Cs(pid_o, pid_m) .* R_h(hid_o, hid_m) .* R_v(iz_o, iz_m);   % Nobs x Nm

% Sigma_oo also vectorized
Sigma_oo = Cs(pid_o, pid_o) .* R_h(hid_o, hid_o) .* R_v(iz_o, iz_o);   % Nobs x Nobs
Sigma_oo = (Sigma_oo + Sigma_oo')/2 + 1e-10*eye(Nobs);

% --- Phi_cond_true_o = Sigma_om * (Sigma_mm \ Phi_m) ---
Phi_m_solved = kron_solve3(Lc_mm, Lh_mm, Lv_mm, Phi_m);     % Nm x K
Phi_cond_true_o = Sigma_om * Phi_m_solved;                  % Nobs x K
O_o = Phi_o - Phi_cond_true_o;

% --- Sigma_oo_cond = Sigma_oo - Sigma_mo'*(Sigma_mm\Sigma_mo) ---
Sigma_mo = Sigma_om.';                                      % Nm x Nobs
X = kron_solve3(Lc_mm, Lh_mm, Lv_mm, Sigma_mo);             % Nm x Nobs
Sigma_oo_cond = Sigma_oo - Sigma_mo' * X;                   % Nobs x Nobs
Sigma_oo_cond = (Sigma_oo_cond + Sigma_oo_cond')/2 + 1e-10*eye(Nobs);

% --- ttt1 = O_o * C * O_o' (same math, faster via chol(C)) ---
LC = chol(C + 1e-12*eye(size(C)),'lower');
G  = O_o * LC;              % Nobs x K
ttt1 = G * G.';
ttt1 = (ttt1 + ttt1')/2 + 1e-10*eye(Nobs);

ttt2 = Sigma_oo_cond;

%% ===== 9) Var + log-like (same) =====
Var_plus_o_minus = ttt1 + ttt2;
Var_plus_o_minus = (Var_plus_o_minus + Var_plus_o_minus')/2;

Var_plus_o_minus = Var_plus_o_minus + (1e-6 + 1e-3*median(diag(Var_plus_o_minus))) * eye(Nobs);

L = chol(Var_plus_o_minus,'lower');
r = y_plus_o - Ey_o;
v = L\r;

log_like_plus_o = -0.5*Nobs*log(2*pi) - sum(log(diag(L))) - 0.5*(v'*v);

%% ===== 10) lattice-minus log-like (keep your original exactly) =====
inv_Cs_lat = inv_Cs_lat; %#ok<NASGU> % already computed but keep naming

L_Cs = chol(Cs_lat,'lower');
log_Cs = sum(log(diag(L_Cs)));

vvv = {phi_t_Cs_lat'*inv_Cs_lat, phih_lat'*invRh, (phiz(z_minus,:)'*invRv_minus)};
v1 = kronmult3({inv_Cs_lat, invRh, invRv_minus}, y_minus(:));
tmp1 = y_minus' * v1(:);

v1 = kronmult3(vvv, reshape(y_minus, sum(z_minus)*nh_lat*M_CPT, 1));
tmp2 = v1' * C * v1;

e = tmp1 - tmp2;

log_detRv_minus = 2*sum(log(diag(chol(R_v(z_minus,z_minus),'lower'))));
log_detRh       = 2*sum(log(diag(chol(R_h_lat,'lower'))));

LC2 = chol(C + 1e-12*eye(size(C)),'lower');
d = -2*sum(log(diag(LC2))) ...
    + M_CPT*sum(z_minus)*log_detRh ...
    + M_CPT*nh_lat*log_detRv_minus ...
    + 2*nh_lat*sum(z_minus)*log_Cs ...
    - sum(log(alpha));

log_like_minus = -0.5*M_CPT*nh_lat*sum(z_minus)*log(2*pi) - 0.5*d - 0.5*e;

log_like = log_like_minus + log_like_plus_o;

end % =================== END MAIN ===================


%% ===== helper: decode_global_index (same as yours) =====
function [iz, hid, pid] = decode_global_index(g, nv, nh_all)
iz  = mod(g-1, nv) + 1;
j   = floor((g-1)/nv) + 1;
pid = floor((j-1)/nh_all) + 1;
hid = mod((j-1), nh_all) + 1;
end


%% ===== helper: exact Kronecker solve (Cs ⊗ Rh ⊗ Rv) \ B =====
function X = kron_solve3(Lc, Lh, Lv, B)
% B: (nz*nh*m) x R
[m,~]  = size(Lc);
[nh,~] = size(Lh);
[nz,~] = size(Lv);
R      = size(B,2);

X = reshape(B, [nz, nh, m, R]);

% solve Rv (dim-1)
X = reshape(X, [nz, nh*m*R]);
X = Lv \ X;
X = Lv' \ X;
X = reshape(X, [nz, nh, m, R]);

% solve Rh (dim-2)
X = permute(X, [2 1 3 4]);     % [nh, nz, m, R]
X = reshape(X, [nh, nz*m*R]);
X = Lh \ X;
X = Lh' \ X;
X = reshape(X, [nh, nz, m, R]);
X = permute(X, [2 1 3 4]);

% solve Cs (dim-3)
X = permute(X, [3 1 2 4]);     % [m, nz, nh, R]
X = reshape(X, [m, nz*nh*R]);
X = Lc \ X;
X = Lc' \ X;
X = reshape(X, [m, nz, nh, R]);
X = permute(X, [2 3 1 4]);

X = reshape(X, [nz*nh*m, R]);
end


%% ===== helper: cache key =====
function key = fast_cache_key(nv,nh_all,M_all,holes_cpt,pid_cpt,z_minus,o_idx,mz,mh,seed)
key = sprintf('%d_%d_%d_%d_%d_%d_%d_%d_%d_%g', ...
    nv,nh_all,M_all,holes_cpt(1),pid_cpt(1),sum(z_minus),numel(o_idx),mz,mh,seed);
end
