function [log_like] = Baytown_log_like_Step2_2c(data,nan_ind,x,y,z,sof_v,sof_h,nu_v,nu_h,bhp,alpha,Y,Cs,ahp,sof_v_t,sof_h_t) %#ok<INUSD>
%BAYTOWN_LOG_LIKE_STEP2_2C Evaluate the MGPR log-likelihood for Baytown.
%
% This accelerated implementation constructs only the covariance matrix
% associated with the available observations. It avoids forming the full
% Kronecker covariance matrix, which can require excessive memory for
% non-lattice multivariate data.
%
% INPUTS
%   data    : Observation matrix of size nv-by-(nh*M). The column order is
%             sounding/location first within each parameter block.
%   nan_ind : Logical matrix with the same size as data. A true entry
%             indicates a missing observation.
%   x, y    : Horizontal coordinates of the nh sounding locations.
%   z       : Vertical coordinates of the nv depth points.
%   sof_v   : Vertical SOF of the residual component.
%   sof_h   : Horizontal SOF of the residual component.
%   nu_v    : Vertical Matern smoothness parameter.
%   nu_h    : Horizontal Matern smoothness parameter.
%   bhp     : Residual covariance amplitude.
%   alpha   : Trend-weight prior precision vector. It is retained in the
%             interface for compatibility but is not used in this direct
%             covariance implementation.
%   Y       : Model structure retained for interface compatibility.
%   Cs      : M-by-M cross-parameter covariance matrix.
%   ahp     : Trend covariance amplitude.
%   sof_v_t : Vertical SOF of the Gaussian trend component.
%   sof_h_t : Horizontal SOF of the Gaussian trend component.
%
% OUTPUT
%   log_like : Joint Gaussian log-likelihood of all available observations.
%
% OBSERVATION ORDER
%   MATLAB linear indexing follows the order
%
%       depth -> sounding/location -> parameter.
%
% COVARIANCE MODEL
%   For two available observations, the covariance is
%
%       Sigma = ahp * Cs .* Kt_h .* Kt_v
%             + bhp * Cs .* Rh   .* Rv,
%
%   where Kt_h and Kt_v are Gaussian trend kernels, and Rh and Rv are
%   Matern residual kernels. The corresponding observed-data covariance
%   is assembled directly using element-wise Hadamard products.
%
% COMPUTATIONAL FEATURES
%   1. Construct Sigma_oo directly with Hadamard products.
%   2. Avoid forming the full Kronecker covariance matrix.
%   3. Add diagonal jitter through linear indexing without forming eye.
%   4. Decode all observed indices only once.
%   5. Increase the diagonal nugget if the Cholesky factorization fails.
% =========================================================================

%% ===== 0) Numerical settings and dimensions =====
jitter   = 1e-10;
jitterRh = 1e-10;
jitterRv = 1e-10;
jitterKt = 1e-10;

M_all  = size(Cs, 1);
nh_all = round(size(data, 2) / M_all);
nv     = size(data, 1);

%% ===== 1) Extract the observed-data vector =====
o_idx = find(~nan_ind(:));
Nobs  = numel(o_idx);

if Nobs == 0
    log_like = -Inf;
    return;
end

y_o = data(o_idx);   % Direct indexing avoids creating data(:) separately.

%% ===== 2) Construct the spatial distance matrices =====
dx = x(:) - x(:).';
dy = y(:) - y(:).';
Dh = sqrt(dx.^2 + dy.^2);
Dz = abs(z(:) - z(:).');

%% ===== 3) Assign the trend-kernel scales =====
sofh_t_use = sof_h_t;
sofv_t_use = sof_v_t;

%% ===== 4) Construct the trend and residual kernels =====

% Gaussian kernels for the spatially varying trend component.
Kt_h = exp(-pi * (Dh.^2) / sofh_t_use^2);
Kt_v = exp(-pi * (Dz.^2) / sofv_t_use^2);
Kt_h = (Kt_h + Kt_h') * 0.5;
Kt_h(1:nh_all+1:end) = Kt_h(1:nh_all+1:end) + jitterKt;  % Add jitter without forming eye.
Kt_v = (Kt_v + Kt_v') * 0.5;
Kt_v(1:nv+1:end)     = Kt_v(1:nv+1:end)     + jitterKt;

% Matern kernels for the spatially correlated residual component.
Rh = Matern_R(nu_h, sof_h, Dh);  Rh(isnan(Rh)) = 1;
Rv = Matern_R(nu_v, sof_v, Dz);  Rv(isnan(Rv)) = 1;
Rh = (Rh + Rh') * 0.5;
Rh(1:nh_all+1:end) = Rh(1:nh_all+1:end) + jitterRh;
Rv = (Rv + Rv') * 0.5;
Rv(1:nv+1:end)     = Rv(1:nv+1:end)     + jitterRv;

%% ===== 5) Decode the observed-data indices =====

% Convert each linear observation index into its depth, sounding, and
% parameter indices. This decoding is performed once and reused below.
iz_o  = mod(o_idx - 1, nv)     + 1;
j_o   = floor((o_idx - 1) / nv) + 1;
pid_o = floor((j_o - 1) / nh_all) + 1;
hid_o = mod(j_o - 1, nh_all)   + 1;

%% ===== 6) Assemble the observed-data covariance directly =====

% Extract only the covariance entries required by the available data.
Cs_oo  = Cs(pid_o,  pid_o);
KtH_oo = Kt_h(hid_o, hid_o);
KtV_oo = Kt_v(iz_o,  iz_o);
Rh_oo  = Rh(hid_o,  hid_o);
Rv_oo  = Rv(iz_o,   iz_o);

% Combine the trend and residual covariance terms without constructing
% full Kronecker matrices or unnecessary intermediate arrays.
Sigma_oo = ahp * (Cs_oo .* KtH_oo .* KtV_oo) ...
         + bhp * (Cs_oo .* Rh_oo  .* Rv_oo);
Sigma_oo = (Sigma_oo + Sigma_oo') * 0.5;
Sigma_oo(1:Nobs+1:end) = Sigma_oo(1:Nobs+1:end) + jitter;  % Add jitter without forming eye(Nobs).

%% ===== 7) Cholesky factorization and Gaussian log-likelihood =====
[L, p] = chol(Sigma_oo, 'lower');

if p > 0
    % Increase the diagonal nugget until the covariance matrix becomes
    % positive definite. At most 12 nugget levels are attempted.
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
