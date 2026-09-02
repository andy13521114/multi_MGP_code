function log_like = GP_Matern_3D(x,y,Cs)
%GP_MATERN_3D Evaluate the Taipei MGPR data log-likelihood for TMCMC.
%
% This function converts the seven TMCMC parameters from the sampling
% space to physical covariance parameters, constructs the truncated
% Gaussian-process trend bases, reshapes the multivariate observations,
% and evaluates their joint log-likelihood.
%
% INPUTS
%   x  : 1-by-7 vector of sampled model parameters:
%          x(1) = log(1/bhp), where bhp is the residual amplitude
%          x(2) = log(sof_v), the vertical residual SOF
%          x(3) = log(sof_h), the horizontal residual SOF
%          x(4) = log(nu_v), the vertical Matern smoothness
%          x(5) = log(1/ahp), where ahp is the trend amplitude
%          x(6) = log(sof_v_t), the vertical trend SOF
%          x(7) = log(sof_h_t), the horizontal trend SOF
%   y  : Structure containing centered observations, coordinates,
%        distance matrices, and the trend-basis truncation threshold.
%   Cs : M-by-M cross-parameter covariance matrix.
%
% OUTPUT
%   log_like : Joint Gaussian data log-likelihood of all available
%              observations. No additional parameter prior is added in
%              this function.
%
% MODEL ASSUMPTION
%   The same Matern smoothness is used in both spatial directions:
%   nu_h = nu_v.

%% ===== 1) Convert sampled parameters to physical values =====

M = size(Cs,1);
bhp = 1/exp(x(1)); ln_sof_v = x(2); ln_sof_h = x(3); ln_nu_v = x(4);  ahp = 1/exp(x(5)); ln_sof_v_t = x(6); ln_sof_h_t = x(7);
nu_v = exp(ln_nu_v); nu_h = nu_v; sof_v = exp(ln_sof_v); sof_h = exp(ln_sof_h); sof_v_t = exp(ln_sof_v_t); sof_h_t = exp(ln_sof_h_t);

%% ===== 2) Construct the truncated Gaussian trend bases =====

% phiz and phih are the retained vertical and horizontal trend bases.
% alpha is the prior precision vector of the latent trend weights. The
% cross-parameter covariance Cs is included in this construction.
[y.phiz,y.phih,alpha] = GP_matrices(ahp,sof_v_t,sof_h_t,y,Cs);

%% ===== 3) Arrange the observations for likelihood evaluation =====

% Reshape the centered observation vector into an nz-by-(nh*M) matrix.
% The corresponding MATLAB linear order is
%
%   depth -> sounding/location -> parameter.
%
% Missing observations are identified by nan_ind and excluded by the
% likelihood function.
nh = size(y.temp_h,1); nz = size(y.temp_z,1); reshape_t = reshape(y.t,nz,nh*M); nan_ind = isnan(reshape_t);

%% ===== 4) Evaluate the joint MGPR data log-likelihood =====

% The active likelihood uses the low-rank trend representation generated
% above together with the Matern residual covariance and Cs.
%[log_like]  = log_like_Step2c_0208_EXT(reshape_t,nan_ind,y.X,y.Y,y.z,sof_v,sof_h,nu_v,nu_h,bhp,alpha,y,Cs,ahp,sof_v_t,sof_h_t);
[log_like]  = log_like_Step2c(reshape_t,nan_ind,y.X,y.Y,y.z,sof_v,sof_h,nu_v,nu_h,bhp,alpha,y,Cs);
% disp(['log_like_2C_new = ', num2str(log_like)]);
% disp(['log_like_2C_intial = ', num2str(log_like_ci)]);
