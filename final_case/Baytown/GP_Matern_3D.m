function log_like = GP_Matern_3D(x, y, Cs)
%GP_MATERN_3D Evaluate the posterior log-likelihood of the MGPR model.
%
% This function is called by TMCMC. It converts the seven sampled
% parameters from the logarithmic sampling space to the physical
% parameter space, constructs the Gaussian-process trend basis, evaluates
% the data log-likelihood, and adds the weak regularization priors.
%
% INPUTS
%   x  : 1-by-7 vector of sampled model parameters:
%
%        x(1) = log(1 / bhp)
%        x(2) = log(sof_v)
%        x(3) = log(sof_h)
%        x(4) = log(nu_v)
%        x(5) = log(1 / ahp)
%        x(6) = log(sof_v_t)
%        x(7) = log(sof_h_t)
%
%   y  : Structure containing the model data and spatial information:
%
%        y.t          - Centered observations stored as a vector
%        y.X          - Horizontal x-coordinates of training locations
%        y.Y          - Horizontal y-coordinates of training locations
%        y.z          - Vertical coordinates
%        y.temp_h     - Horizontal distance matrix
%        y.temp_z     - Vertical distance matrix
%        y.eig_thresh - Cumulative eigenvalue threshold for trend bases
%
%   Cs : Cross-parameter covariance matrix.
%
% OUTPUT
%   log_like : Posterior log-likelihood used by TMCMC. It is calculated as
%
%              log_like = data log-likelihood + log-prior
%
% MODEL PARAMETERS
%   bhp     : Residual covariance amplitude
%   sof_v   : Vertical scale of fluctuation of the residual component
%   sof_h   : Horizontal scale of fluctuation of the residual component
%   nu_v    : Vertical Matern smoothness parameter
%   nu_h    : Horizontal Matern smoothness parameter
%   ahp     : Trend covariance amplitude
%   sof_v_t : Vertical scale of fluctuation of the trend component
%   sof_h_t : Horizontal scale of fluctuation of the trend component
%
% The same Matern smoothness parameter is currently used in the vertical
% and horizontal directions, i.e., nu_h = nu_v.


%% ===== 1. Convert sampled parameters to physical values =====

% Number of modeled geotechnical parameters.
M = size(Cs, 1);

% Amplitude parameters are represented by their inverse logarithms.
bhp = 1 / exp(x(1));
ahp = 1 / exp(x(5));

% Residual covariance parameters in logarithmic space.
ln_sof_v = x(2);
ln_sof_h = x(3);
ln_nu_v  = x(4);

% Trend covariance parameters in logarithmic space.
ln_sof_v_t = x(6);
ln_sof_h_t = x(7);

% Transform the residual parameters back to the physical parameter space.
sof_v = exp(ln_sof_v);
sof_h = exp(ln_sof_h);
nu_v  = exp(ln_nu_v);

% Use the same Matern smoothness in both spatial directions.
nu_h = nu_v;

% Transform the trend SOFs back to the physical parameter space.
sof_v_t = exp(ln_sof_v_t);
sof_h_t = exp(ln_sof_h_t);


%% ===== 2. Construct the Gaussian-process trend basis =====

% GP_matrices performs the eigendecomposition of the vertical,
% horizontal, and cross-parameter trend covariance matrices.
%
% phiz  : Truncated vertical trend basis
% phih  : Truncated horizontal trend basis
% alpha : Prior precision vector of the latent trend weights
[y.phiz, y.phih, alpha] = ...
    GP_matrices(ahp, sof_v_t, sof_h_t, y, Cs);


%% ===== 3. Arrange observations for the likelihood calculation =====

% The centered observation vector y.t follows MATLAB column-major order:
%
%   depth -> sounding/location -> parameter
%
% It is reshaped into an nz-by-(nh*M) matrix, where:
%
%   nz = number of vertical grid points
%   nh = number of training locations
%   M  = number of modeled parameters
nh = size(y.temp_h, 1);
nz = size(y.temp_z, 1);

reshape_t = reshape(y.t, nz, nh * M);

% Identify missing observations. Missing entries are excluded from the
% likelihood calculation.
nan_ind = isnan(reshape_t);


%% ===== 4. Evaluate the data log-likelihood =====

% The current implementation uses log_like_Step2c.
%
% This function evaluates the MGPR likelihood using:
%   1. A Gaussian-process trend component
%   2. A Whittle-Matern residual component
%   3. The cross-parameter covariance matrix Cs
%   4. Only the available observations indicated by nan_ind
log_like_post = log_like_Step2c( ...
    reshape_t, ...
    nan_ind, ...
    y.X, ...
    y.Y, ...
    y.z, ...
    sof_v, ...
    sof_h, ...
    nu_v, ...
    nu_h, ...
    bhp, ...
    alpha, ...
    y, ...
    Cs);


%% ===== 5. Evaluate the weak parameter priors =====

% The prior terms weakly regularize the model parameters and reduce
% overfitting, particularly when the site observations are limited.
log_prior = log_prior_fun( ...
    bhp, ...
    sof_v, ...
    sof_h, ...
    nu_v, ...
    nu_h, ...
    ahp, ...
    sof_v_t, ...
    sof_h_t, ...
    y);


%% ===== 6. Return the posterior log-likelihood =====

% TMCMC uses the sum of the data log-likelihood and the log-prior.
log_like = log_like_post + log_prior;

% To run the model without the weak regularization priors, replace the
% preceding line with:
%
% log_like = log_like_post;

end


function lp = log_prior_fun( ...
    bhp, sofv, sofh, nuv, nuh, ahp, sofv_t, sofh_t, y)
%LOG_PRIOR_FUN Evaluate weak regularization priors for MGPR parameters.
%
% The prior terms are introduced mainly to reduce unrealistic or highly
% overfitted combinations of covariance parameters.
%
% INPUTS
%   bhp    : Residual covariance amplitude
%   sofv   : Vertical residual SOF
%   sofh   : Horizontal residual SOF
%   nuv    : Vertical Matern smoothness parameter
%   nuh    : Horizontal Matern smoothness parameter
%   ahp    : Trend covariance amplitude
%   sofv_t : Vertical trend SOF
%   sofh_t : Horizontal trend SOF
%   y      : Structure containing the site distance matrices
%
% OUTPUT
%   lp : Sum of the unnormalized log-prior terms.
%
% The normalization constants are omitted because they do not depend on
% the sampled parameters and therefore do not affect TMCMC comparisons.


%% ===== 1. Initialize the log-prior =====

lp = 0;


%% ===== 2. Define prior penalty functions =====

% Gaussian penalty applied to log-transformed positive parameters.
l2log = @(value, mu, sigma) ...
    -0.5 * ((log(value) - mu) / sigma)^2;

% Gaussian penalty applied directly in the physical parameter space.
l2 = @(value, mu, sigma) ...
    -0.5 * ((value - mu) / sigma)^2;


%% ===== 3. Determine representative spatial scales =====

% Use 2 m as a weak prior center for the vertical residual SOF.
mu_sofv  = log(2.0);
sig_sofv = 1.0;

% Use the median nonzero distance between training locations as the weak
% prior center for the horizontal residual SOF.
Dh = y.temp_h;
dmed = median(Dh(Dh > 0), 'omitnan');

% Use 10 m as a fallback value if the median distance is unavailable.
if ~isfinite(dmed) || dmed <= 0
    dmed = 10;
end

mu_sofh  = log(dmed);
sig_sofh = 1.0;


%% ===== 4. Residual SOF priors =====

% These priors discourage unrealistically short or long residual
% correlation scales.
lp = lp + l2log(sofv, mu_sofv, sig_sofv);
lp = lp + l2log(sofh, mu_sofh, sig_sofh);


%% ===== 5. Trend SOF priors =====

% The trend component is weakly encouraged to be smoother than the
% residual component. The prior centers the trend SOFs at approximately
% three times the corresponding residual SOFs.
lp = lp + l2log( ...
    sofv_t, log(max(sofv, 1e-6) * 3), 1.0);

lp = lp + l2log( ...
    sofh_t, log(max(sofh, 1e-6) * 3), 1.0);

% A trend SOF much larger than the corresponding site dimension produces
% an almost spatially constant trend in that direction.
%
% If both sofv_t and sofh_t are much larger than the vertical and
% horizontal site dimensions, respectively, the MGPR trend approaches
% the t-const model.


%% ===== 6. Covariance-amplitude priors =====

% Apply weak log-scale priors to the residual and trend amplitudes.
lp = lp + l2log(bhp, log(10), 1.5);
lp = lp + l2log(ahp, log(10), 1.5);


%% ===== 7. Trend-to-residual amplitude-ratio prior =====

% Weakly encourage the trend and residual amplitudes to have a comparable
% order of magnitude.
lp = lp + l2log(ahp / bhp, log(1.0), 1.0);


%% ===== 8. Matern smoothness priors =====

% Apply weak Gaussian priors directly to the Matern smoothness parameters.
lp = lp + l2(nuv, 1.2, 0.6);
lp = lp + l2(nuh, 1.2, 0.6);

% Note:
% The current model sets nuh = nuv. Consequently, the two preceding terms
% apply the same smoothness penalty twice. This behavior is preserved from
% the original implementation.

end
