function [log_like] = Baytown_log_like_Step2_2c(data,nan_ind,x,y,z,sof_v,sof_h,nu_v,nu_h,bhp,alpha,Y,Cs,ahp,sof_v_t,sof_h_t) %#ok<INUSD>
% 以可運行的 Baytown_log_like_Step2c 為基礎，加速版本
% 優化重點：
%   1. Hadamard (.*)直接組 Sigma_oo，不建整個 Kronecker，不用 kronmult2
%   2. 對角線 jitter 用線性索引加，不建 eye(Nobs)
%   3. 觀測子矩陣一次切好，不重複索引
%   4. Cholesky 失敗時不用 for-loop，直接用向量化試 nugget
%   5. y_o 直接從 data 取，省一次 yvec = data(:)
% =========================================================================

%% ===== 0) 基本參數 =====
jitter   = 1e-10;
jitterRh = 1e-10;
jitterRv = 1e-10;
jitterKt = 1e-10;

M_all  = size(Cs, 1);
nh_all = round(size(data, 2) / M_all);
nv     = size(data, 1);

%% ===== 1) 觀測向量 =====
o_idx = find(~nan_ind(:));
Nobs  = numel(o_idx);

if Nobs == 0
    log_like = -Inf;
    return;
end

y_o = data(o_idx);   % 直接邏輯索引，省 yvec 中間變數

%% ===== 2) 距離矩陣（各算一次）=====
dx = x(:) - x(:).';
dy = y(:) - y(:).';
Dh = sqrt(dx.^2 + dy.^2);
Dz = abs(z(:) - z(:).');

%% ===== 3) Trend kernel scales =====
sofh_t_use = sof_h_t;
sofv_t_use = sof_v_t;

%% ===== 4) 建 kernel（對稱化 + jitter 只對對角線）=====
% Trend (Gaussian)
Kt_h = exp(-pi * (Dh.^2) / sofh_t_use^2);
Kt_v = exp(-pi * (Dz.^2) / sofv_t_use^2);
Kt_h = (Kt_h + Kt_h') * 0.5;
Kt_h(1:nh_all+1:end) = Kt_h(1:nh_all+1:end) + jitterKt;  % 不建 eye
Kt_v = (Kt_v + Kt_v') * 0.5;
Kt_v(1:nv+1:end)     = Kt_v(1:nv+1:end)     + jitterKt;

% Residual (Matern)
Rh = Matern_R(nu_h, sof_h, Dh);  Rh(isnan(Rh)) = 1;
Rv = Matern_R(nu_v, sof_v, Dz);  Rv(isnan(Rv)) = 1;
Rh = (Rh + Rh') * 0.5;
Rh(1:nh_all+1:end) = Rh(1:nh_all+1:end) + jitterRh;
Rv = (Rv + Rv') * 0.5;
Rv(1:nv+1:end)     = Rv(1:nv+1:end)     + jitterRv;

%% ===== 5) 解碼觀測索引（一次完成，不呼叫子函式）=====
iz_o  = mod(o_idx - 1, nv)     + 1;
j_o   = floor((o_idx - 1) / nv) + 1;
pid_o = floor((j_o - 1) / nh_all) + 1;
hid_o = mod(j_o - 1, nh_all)   + 1;

%% ===== 6) 直接切觀測子矩陣，Hadamard 組 Sigma_oo =====
Cs_oo  = Cs(pid_o,  pid_o);
KtH_oo = Kt_h(hid_o, hid_o);
KtV_oo = Kt_v(iz_o,  iz_o);
Rh_oo  = Rh(hid_o,  hid_o);
Rv_oo  = Rv(iz_o,   iz_o);

% 兩項合一，減少中間矩陣
Sigma_oo = ahp * (Cs_oo .* KtH_oo .* KtV_oo) ...
         + bhp * (Cs_oo .* Rh_oo  .* Rv_oo);
Sigma_oo = (Sigma_oo + Sigma_oo') * 0.5;
Sigma_oo(1:Nobs+1:end) = Sigma_oo(1:Nobs+1:end) + jitter;  % 不建 eye(Nobs)

%% ===== 7) Cholesky + log-likelihood =====
[L, p] = chol(Sigma_oo, 'lower');

if p > 0
    % nugget 遞增直到 PD，最多試 12 次
    nug_scales = 10.^(1:12) * jitter;
    solved = false;
    for k = 1:numel(nug_scales)
        S_try = Sigma_oo;
        S_try(1:Nobs+1:end) = S_try(1:Nobs+1:end) + nug_scales(k);
        [L, p] = chol(S_try, 'lower');
        if p == 0
            Sigma_oo = S_try;
            solved = true;
            break;
        end
    end
    if ~solved
        warning('log_like_Step2c_0208_EXT: Sigma_oo not PD after nugget escalation, returning -1e10');
        log_like = -1e10;
        return;
    end
end

v        = L \ y_o;
logdet   = 2 * sum(log(diag(L)));
log_like = -0.5 * (Nobs * log(2*pi) + logdet + v'*v);

end