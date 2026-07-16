function [log_like] = Baytown_log_like_Step2c(data, nan_ind, x, y, z, sof_v, sof_h, nu_v, nu_h, bhp, alpha, Y, Cs, ahp, sof_v_t, sof_h_t)
% Baytown_log_like_Step2c
% 與 log_like_Step2c_0208_EXT 等價的 exact GP 似然度
%
% 模型：
%   Sigma = Phi * diag(1/alpha) * Phi'  +  bhp * (Cs ⊗ Rh ⊗ Rv)
%
% 其中：
%   Phi = phi_t_Cs ⊗ phih ⊗ phiz   [nv*nh*M x rank]
%   rank = mCt * mh * mz
%
% 對觀測點子矩陣直接用 Woodbury 公式避免建完整 Sigma：
%   Sigma_oo^{-1} = Sigma_res_oo^{-1}
%                 - Sigma_res_oo^{-1} * Phi_o * (diag(alpha) + Phi_o'*Sigma_res_oo^{-1}*Phi_o)^{-1} * Phi_o' * Sigma_res_oo^{-1}
%
% 使用 kronmult2 避免建完整 Kronecker 矩陣

%% ===== 基本設定 =====
jitter   = 1e-8;
jitterRh = 1e-6;
jitterRv = 1e-6;

M_all  = size(Cs, 1);
nh_all = round(size(data, 2) / M_all);
nv     = size(data, 1);

%% ===== 觀測索引 =====
o_idx = find(~nan_ind(:));
Nobs  = numel(o_idx);
if Nobs == 0
    log_like = -Inf;
    return;
end
yvec = data(:);
y_o  = yvec(o_idx);

%% ===== 距離矩陣 =====
dx = x(:) - x(:).';
dy = y(:) - y(:).';
Dh = sqrt(dx.^2 + dy.^2);
Dz = abs(z(:) - z(:).');

%% ===== 殘差核（Matern）=====
Rh = Matern_R(nu_h, sof_h, Dh); Rh(isnan(Rh)) = 1;
Rv = Matern_R(nu_v, sof_v, Dz); Rv(isnan(Rv)) = 1;
Rh = (Rh + Rh')/2 + jitterRh * eye(nh_all);
Rv = (Rv + Rv')/2 + jitterRv * eye(nv);

%% ===== 從 Y 取得 basis（與 log_like_Step2c 相同）=====
phiz     = Y.phiz;   % [nv x mz]
phih     = Y.phih;   % [nh x mh]
mz       = size(phiz, 2);
mh       = size(phih, 2);

% Cs 特徵分解（與 log_like_Step2c 相同）
[phi_t_Cs, omege_t_Cs] = eig(Cs);
eigvals = diag(omege_t_Cs);
[~, idx] = sort(eigvals, 'descend');
phi_t_Cs = phi_t_Cs(:, idx);
mCt = M_all;   % phi_t_Cs 是 M x M 方陣

rank_w = mCt * mh * mz;   % w 的維度

%% ===== 解碼觀測索引 =====
% data 排列：[nv x (nh*M)]，column-major 展平
% 欄 j：pid = floor((j-1)/nh)+1，hid = mod(j-1,nh)+1
% 列 i：iz = i
iz_o  = mod(o_idx-1, nv) + 1;
j_o   = floor((o_idx-1)/nv) + 1;
pid_o = floor((j_o-1)/nh_all) + 1;
hid_o = mod(j_o-1, nh_all) + 1;

%% ===== 殘差協方差的觀測子矩陣 Sigma_res_oo =====
% Sigma_res = bhp * (Cs ⊗ Rh ⊗ Rv)
% 對觀測點：Sigma_res_oo(i,j) = bhp * Cs(pid_i,pid_j) * Rh(hid_i,hid_j) * Rv(iz_i,iz_j)
Cs_oo  = Cs(pid_o, pid_o);
Rh_oo  = Rh(hid_o, hid_o);
Rv_oo  = Rv(iz_o,  iz_o);

Sigma_res_oo = bhp * (Cs_oo .* Rh_oo .* Rv_oo);
Sigma_res_oo = (Sigma_res_oo + Sigma_res_oo')/2 + jitter * eye(Nobs);

%% ===== 建立 Phi_o：觀測點的 basis 矩陣 =====
% Phi = phi_t_Cs ⊗ phih ⊗ phiz   [nv*nh*M x rank_w]
% Phi_o = Phi(o_idx, :)           [Nobs x rank_w]
%
% 第 (k_Ct, k_h, k_z) 個基底對應的列向量：
%   Phi(global_idx, :) = phi_t_Cs(pid, k_Ct) * phih(hid, k_h) * phiz(iz, k_z)
%
% 利用 Hadamard 結構直接建 Phi_o，不建完整 Phi

% 預先計算每個觀測點在三個維度的 basis 值
% phi_t_Cs_o: [Nobs x mCt]
% phih_o:     [Nobs x mh]
% phiz_o:     [Nobs x mz]
phi_t_Cs_o = phi_t_Cs(pid_o, :);   % [Nobs x mCt]
phih_o     = phih(hid_o, :);        % [Nobs x mh]
phiz_o     = phiz(iz_o,  :);        % [Nobs x mz]

% 建立 Phi_o [Nobs x rank_w]，rank_w = mCt * mh * mz
% 排列順序：(k_Ct 最慢, k_h 居中, k_z 最快) — 對應 kron(phi_t_Cs, kron(phih, phiz))
Phi_o = zeros(Nobs, rank_w);
col = 0;
for k_Ct = 1:mCt
    for k_h = 1:mh
        for k_z = 1:mz
            col = col + 1;
            Phi_o(:, col) = phi_t_Cs_o(:,k_Ct) .* phih_o(:,k_h) .* phiz_o(:,k_z);
        end
    end
end

%% ===== Woodbury 公式計算 log-likelihood =====
% Sigma_oo = Phi_o * diag(1/alpha) * Phi_o' + Sigma_res_oo
%
% Woodbury：
%   Sigma_oo^{-1} = Sigma_res_oo^{-1}
%                 - Sigma_res_oo^{-1} * Phi_o * M_w^{-1} * Phi_o' * Sigma_res_oo^{-1}
%   M_w = diag(alpha) + Phi_o' * Sigma_res_oo^{-1} * Phi_o   [rank_w x rank_w]
%
% log det(Sigma_oo) = log det(Sigma_res_oo) + log det(M_w) - log det(diag(alpha))
%                   = log det(Sigma_res_oo) + log det(M_w) + sum(log(alpha))

% Cholesky of Sigma_res_oo
[L_res, p_res] = chol(Sigma_res_oo, 'lower');
if p_res > 0
    nug = jitter;
    for k = 1:12
        nug = nug * 10;
        [L_res, p_res] = chol(Sigma_res_oo + nug*eye(Nobs), 'lower');
        if p_res == 0; break; end
    end
    if p_res > 0
        warning('Sigma_res_oo not PD, returning -1e10');
        log_like = -1e10; return;
    end
end

% Sigma_res_oo^{-1} * y_o  and  Sigma_res_oo^{-1} * Phi_o
invRes_yo  = L_res' \ (L_res \ y_o);           % [Nobs x 1]
invRes_Phi = L_res' \ (L_res \ Phi_o);         % [Nobs x rank_w]

% M_w = diag(alpha) + Phi_o' * invRes_Phi
M_w = diag(alpha) + Phi_o' * invRes_Phi;       % [rank_w x rank_w]
M_w = (M_w + M_w')/2;

[L_Mw, p_Mw] = chol(M_w, 'lower');
if p_Mw > 0
    nug = 1e-10;
    for k = 1:12
        nug = nug * 10;
        [L_Mw, p_Mw] = chol(M_w + nug*eye(rank_w), 'lower');
        if p_Mw == 0; break; end
    end
    if p_Mw > 0
        warning('M_w not PD, returning -1e10');
        log_like = -1e10; return;
    end
end

% log det(Sigma_oo) = log det(Sigma_res_oo) + log det(M_w) + sum(log(alpha))
logdet_res = 2 * sum(log(diag(L_res)));
logdet_Mw  = 2 * sum(log(diag(L_Mw)));
logdet_tot = logdet_res + logdet_Mw -sum(log(alpha));

% quadratic form: y_o' * Sigma_oo^{-1} * y_o
% = y_o' * invRes_yo - (invRes_Phi'*y_o)' * M_w^{-1} * (invRes_Phi'*y_o)
tmp_w   = Phi_o' * invRes_yo;                   % [rank_w x 1]
invMw_t = L_Mw' \ (L_Mw \ tmp_w);              % [rank_w x 1]
quad    = y_o' * invRes_yo - tmp_w' * invMw_t;

%% ===== 最終 log-likelihood =====
log_like = -0.5 * (Nobs*log(2*pi) + logdet_tot + quad);

%% ===== 有限性檢查 =====
if ~isfinite(log_like)
    warning('log_like 非有限: %.4f', log_like);
    log_like = -1e10;
end

end