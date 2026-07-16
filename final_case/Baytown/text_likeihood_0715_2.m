%% ================================================================
% STRESS TEST:
% Baytown_log_like_Step2_2c
%
% Direct full Kronecker analytical likelihood
%           VS
% Hadamard observed covariance likelihood
%
% Exact model:
%
% Sigma =
%   ahp * kron(Cs,kron(Kt_h,Kt_v))
% + bhp * kron(Cs,kron(Rh,Rv))
% + jitter * I
%
% Ordering:
%   z fastest
%   hole
%   parameter slowest
%
% Same as data(:)
% ================================================================

clear;
clc;
close all;

rng(20260715);

%% ================================================================
% 0. Test settings
% ================================================================

NCASE = 1000;

TOL_ABS = 1e-8;
TOL_REL = 1e-10;

MAX_FAIL_PRINT = 20;

assert( ...
    exist('Matern_R','file') == 2, ...
    'Cannot find Matern_R.m');

assert( ...
    exist('Baytown_log_like_Step2c','file') == 2, ...
    'Cannot find Baytown_log_like_Step2_2c.m');


%% ================================================================
% Same numerical constants as Baytown likelihood
% ================================================================

jitter   = 1e-10;
jitterRh = 1e-10;
jitterRv = 1e-10;
jitterKt = 1e-10;


%% ================================================================
% Missing pattern types
% ================================================================

pattern_names = { ...
    'random', ...
    'exact_nh_missing', ...
    'full_depth_missing', ...
    'parameter_block', ...
    'hole_block', ...
    'heavy_random', ...
    'checkerboard', ...
    'extreme_sparse', ...
    'different_parameter_rates'};

NPATTERN = numel(pattern_names);


%% ================================================================
% Global records
% ================================================================

abs_diff_all = nan(NCASE,1);
rel_diff_all = nan(NCASE,1);

logL_exact_all = nan(NCASE,1);
logL_fast_all  = nan(NCASE,1);

M_all_record  = nan(NCASE,1);
nh_record     = nan(NCASE,1);
nv_record     = nan(NCASE,1);
Nobs_record   = nan(NCASE,1);
rcond_record  = nan(NCASE,1);

pattern_record = cell(NCASE,1);
Cs_mode_record = cell(NCASE,1);

failure_count = 0;

worst.abs_diff = -Inf;


fprintf('\n');
fprintf('============================================================\n');
fprintf(' BAYTOWN MULTI-PARAMETER NON-LATTICE STRESS TEST\n');
fprintf('============================================================\n');

fprintf('Number of cases = %d\n',NCASE);
fprintf('Absolute tol    = %.1e\n',TOL_ABS);
fprintf('Relative tol    = %.1e\n',TOL_REL);


%% ================================================================
% Main stress test
% ================================================================

for icase = 1:NCASE

    %% ============================================================
    % 1. Random dimensions
    % =============================================================

    M  = randi([2,4]);
    nh = randi([4,8]);
    nv = randi([8,18]);

    ncol = nh*M;
    Nfull = nv*nh*M;


    %% ============================================================
    % 2. Random spatial coordinates
    % =============================================================

    x = sort(12*rand(nh,1));

    y = 3*(rand(nh,1)-0.5);

    z = sort( ...
        1.5 + 3.5*rand(nv,1));


    %% ============================================================
    % 3. Random hyperparameters
    % =============================================================

    % residual SOF
    sof_v = log_uniform(0.10,5.0);
    sof_h = log_uniform(0.50,30.0);

    % residual nu
    nu_v = 0.20 + 2.80*rand;
    nu_h = 0.20 + 2.80*rand;

    % residual amplitude
    bhp = log_uniform(0.02,5.0);

    % trend amplitude
    ahp = log_uniform(0.02,5.0);

    % trend Gaussian SOF
    sof_v_t = log_uniform(0.10,8.0);
    sof_h_t = log_uniform(0.50,50.0);


    %% ============================================================
    % 4. Random Cs
    % =============================================================

    cs_modes = { ...
        'weak', ...
        'strong', ...
        'near', ...
        'negative', ...
        'random'};

    cs_mode = cs_modes{ ...
        mod(icase-1,numel(cs_modes))+1};

    Cs = build_Cs(M,cs_mode);

    eigCs = eig(Cs);

    if min(eigCs) <= 0

        error( ...
            'Cs is not SPD in case %d.', ...
            icase);

    end


    %% ============================================================
    % 5. Placeholder arguments
    %
    % Baytown_log_like_Step2_2c does NOT use alpha or Y
    % =============================================================

    alpha = 1;
    Y = struct();


    %% ============================================================
    % 6. Distance matrices
    % =============================================================

    dx = x(:)-x(:).';
    dy = y(:)-y(:).';

    Dh = sqrt( ...
        dx.^2 ...
        + dy.^2);

    Dz = abs( ...
        z(:)-z(:).');


    %% ============================================================
    % 7. Trend Gaussian kernels
    %
    % MUST match Baytown_log_like_Step2_2c exactly
    % =============================================================

    Kt_h = exp( ...
        -pi*(Dh.^2)/(sof_h_t^2));

    Kt_v = exp( ...
        -pi*(Dz.^2)/(sof_v_t^2));


    Kt_h = ...
        (Kt_h+Kt_h')*0.5;

    Kt_h(1:nh+1:end) = ...
        Kt_h(1:nh+1:end) ...
        + jitterKt;


    Kt_v = ...
        (Kt_v+Kt_v')*0.5;

    Kt_v(1:nv+1:end) = ...
        Kt_v(1:nv+1:end) ...
        + jitterKt;


    %% ============================================================
    % 8. Residual Matérn kernels
    %
    % MUST match Baytown_log_like_Step2_2c exactly
    % =============================================================

    Rh = Matern_R( ...
        nu_h, ...
        sof_h, ...
        Dh);

    Rh(isnan(Rh)) = 1;

    Rh = ...
        (Rh+Rh')*0.5;

    Rh(1:nh+1:end) = ...
        Rh(1:nh+1:end) ...
        + jitterRh;


    Rv = Matern_R( ...
        nu_v, ...
        sof_v, ...
        Dz);

    Rv(isnan(Rv)) = 1;

    Rv = ...
        (Rv+Rv')*0.5;

    Rv(1:nv+1:end) = ...
        Rv(1:nv+1:end) ...
        + jitterRv;


    %% ============================================================
    % 9. DIRECT FULL ANALYTICAL COVARIANCE
    % =============================================================

    Sigma_trend = ahp * kron( ...
        Cs, ...
        kron(Kt_h,Kt_v));

    Sigma_res = bhp * kron( ...
        Cs, ...
        kron(Rh,Rv));

    Sigma_full = ...
        Sigma_trend ...
        + Sigma_res;

    Sigma_full = ...
        (Sigma_full+Sigma_full')*0.5;


    % Same final jitter as Baytown observed covariance
    Sigma_full(1:Nfull+1:end) = ...
        Sigma_full(1:Nfull+1:end) ...
        + jitter;


    %% ============================================================
    % 10. Generate exact Gaussian sample
    % =============================================================

    [LS,pS] = chol( ...
        Sigma_full, ...
        'lower');

    if pS > 0

        error( ...
            'Sigma_full is not PD in case %d.', ...
            icase);

    end


    y_full = ...
        LS*randn(Nfull,1);

    data_full = reshape( ...
        y_full, ...
        nv, ...
        nh*M);


    %% ============================================================
    % 11. Missing pattern
    % =============================================================

    pattern_type = pattern_names{ ...
        mod(icase-1,NPATTERN)+1};


    missing = make_missing_pattern( ...
        nv, ...
        nh, ...
        M, ...
        pattern_type);


    data = data_full;

    data(missing) = NaN;

    nan_ind = isnan(data);


    %% ============================================================
    % 12. Observed indices
    % =============================================================

    o_idx = find( ...
        ~nan_ind(:));

    Nobs = numel(o_idx);

    y_o = data(o_idx);


    if Nobs == 0

        error( ...
            'No observations in case %d.', ...
            icase);

    end


    %% ============================================================
    % 13. DIRECT EXACT OBSERVED LIKELIHOOD
    % =============================================================

    Sigma_oo_exact = ...
        Sigma_full(o_idx,o_idx);

    Sigma_oo_exact = ...
        (Sigma_oo_exact ...
        + Sigma_oo_exact')*0.5;


    [L_exact,p_exact] = chol( ...
        Sigma_oo_exact, ...
        'lower');


    if p_exact > 0

        error( ...
            'Sigma_oo_exact not PD in case %d.', ...
            icase);

    end


    v_exact = ...
        L_exact\y_o;

    logdet_exact = ...
        2*sum(log(diag(L_exact)));

    logL_exact = ...
        -0.5*( ...
        Nobs*log(2*pi) ...
        + logdet_exact ...
        + v_exact'*v_exact);


    %% ============================================================
    % 14. BAYTOWN FAST HADAMARD LIKELIHOOD
    % =============================================================

    logL_fast = ...
        Baytown_log_like_Step2_2c( ...
        data, ...
        nan_ind, ...
        x, ...
        y, ...
        z, ...
        sof_v, ...
        sof_h, ...
        nu_v, ...
        nu_h, ...
        bhp, ...
        alpha, ...
        Y, ...
        Cs, ...
        ahp, ...
        sof_v_t, ...
        sof_h_t);


    %% ============================================================
    % 15. Difference
    % =============================================================

    abs_diff = abs( ...
        logL_fast-logL_exact);

    rel_diff = abs_diff / max( ...
        1, ...
        abs(logL_exact));


    %% ============================================================
    % 16. Records
    % =============================================================

    abs_diff_all(icase) = ...
        abs_diff;

    rel_diff_all(icase) = ...
        rel_diff;

    logL_exact_all(icase) = ...
        logL_exact;

    logL_fast_all(icase) = ...
        logL_fast;

    M_all_record(icase) = M;
    nh_record(icase) = nh;
    nv_record(icase) = nv;
    Nobs_record(icase) = Nobs;

    rcond_record(icase) = ...
        rcond(Sigma_oo_exact);

    pattern_record{icase} = ...
        pattern_type;

    Cs_mode_record{icase} = ...
        cs_mode;


    %% ============================================================
    % 17. Worst case
    % =============================================================

    if abs_diff > worst.abs_diff

        worst.abs_diff = abs_diff;
        worst.rel_diff = rel_diff;

        worst.icase = icase;

        worst.M = M;
        worst.nh = nh;
        worst.nv = nv;

        worst.Nobs = Nobs;

        worst.pattern = ...
            pattern_type;

        worst.cs_mode = ...
            cs_mode;

        worst.logL_exact = ...
            logL_exact;

        worst.logL_fast = ...
            logL_fast;

        worst.rcond = ...
            rcond(Sigma_oo_exact);

        worst.sof_v = sof_v;
        worst.sof_h = sof_h;

        worst.nu_v = nu_v;
        worst.nu_h = nu_h;

        worst.bhp = bhp;
        worst.ahp = ahp;

        worst.sof_v_t = sof_v_t;
        worst.sof_h_t = sof_h_t;

        worst.minCsEig = ...
            min(eigCs);

        worst.maxCsEig = ...
            max(eigCs);

        worst.missing_count = ...
            sum(nan_ind,2);

    end


    %% ============================================================
    % 18. Failure
    % =============================================================

    is_failure = ...
        abs_diff > TOL_ABS ...
        && rel_diff > TOL_REL;


    if is_failure

        failure_count = ...
            failure_count+1;


        if failure_count <= MAX_FAIL_PRINT

            fprintf('\n');
            fprintf('*** FAILURE %d ***\n', ...
                failure_count);

            fprintf( ...
                'Case       = %d\n', ...
                icase);

            fprintf( ...
                'Pattern    = %s\n', ...
                pattern_type);

            fprintf( ...
                'Cs mode    = %s\n', ...
                cs_mode);

            fprintf( ...
                'M/nh/nv    = %d / %d / %d\n', ...
                M,nh,nv);

            fprintf( ...
                'Nobs       = %d\n', ...
                Nobs);

            fprintf( ...
                'Exact      = %.15f\n', ...
                logL_exact);

            fprintf( ...
                'Fast       = %.15f\n', ...
                logL_fast);

            fprintf( ...
                'Abs diff   = %.15e\n', ...
                abs_diff);

            fprintf( ...
                'Rel diff   = %.15e\n', ...
                rel_diff);

            fprintf( ...
                'rcond      = %.15e\n', ...
                rcond(Sigma_oo_exact));

            fprintf( ...
                'Missing/depth:\n');

            disp( ...
                sum(nan_ind,2)');

        end

    end


    %% ============================================================
    % Progress
    % =============================================================

    if mod(icase,100) == 0

        fprintf( ...
            'Case %4d / %4d | max abs diff = %.3e | failures = %d\n', ...
            icase, ...
            NCASE, ...
            max(abs_diff_all(1:icase)), ...
            failure_count);

    end

end


%% ================================================================
% 19. Global results
% ================================================================

[max_abs_diff,idx_abs] = max( ...
    abs_diff_all);

[max_rel_diff,idx_rel] = max( ...
    rel_diff_all);


fprintf('\n\n');
fprintf('============================================================\n');
fprintf(' FINAL GLOBAL RESULTS\n');
fprintf('============================================================\n');

fprintf( ...
    'Total cases              = %d\n', ...
    NCASE);

fprintf( ...
    'Total failures           = %d\n', ...
    failure_count);

fprintf('\n');

fprintf( ...
    'max |fast-exact|         = %.15e\n', ...
    max_abs_diff);

fprintf( ...
    'mean |fast-exact|        = %.15e\n', ...
    mean(abs_diff_all));

fprintf( ...
    'median |fast-exact|      = %.15e\n', ...
    median(abs_diff_all));

fprintf('\n');

fprintf( ...
    'max relative difference  = %.15e\n', ...
    max_rel_diff);

fprintf( ...
    'mean relative difference = %.15e\n', ...
    mean(rel_diff_all));


%% ================================================================
% 20. Worst case details
% ================================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf(' WORST ABSOLUTE DIFFERENCE CASE\n');
fprintf('============================================================\n');

fprintf( ...
    'Case              = %d\n', ...
    worst.icase);

fprintf( ...
    'Pattern           = %s\n', ...
    worst.pattern);

fprintf( ...
    'Cs mode           = %s\n', ...
    worst.cs_mode);

fprintf( ...
    'M / nh / nv       = %d / %d / %d\n', ...
    worst.M, ...
    worst.nh, ...
    worst.nv);

fprintf( ...
    'Nobs              = %d\n', ...
    worst.Nobs);

fprintf('\n');

fprintf( ...
    'Exact likelihood  = %.15f\n', ...
    worst.logL_exact);

fprintf( ...
    'Fast likelihood   = %.15f\n', ...
    worst.logL_fast);

fprintf('\n');

fprintf( ...
    'Absolute diff     = %.15e\n', ...
    worst.abs_diff);

fprintf( ...
    'Relative diff     = %.15e\n', ...
    worst.rel_diff);

fprintf( ...
    'rcond(Sigma_oo)   = %.15e\n', ...
    worst.rcond);

fprintf('\n');

fprintf( ...
    'sof_v / sof_h     = %.6g / %.6g\n', ...
    worst.sof_v, ...
    worst.sof_h);

fprintf( ...
    'nu_v / nu_h       = %.6g / %.6g\n', ...
    worst.nu_v, ...
    worst.nu_h);

fprintf( ...
    'bhp / ahp         = %.6g / %.6g\n', ...
    worst.bhp, ...
    worst.ahp);

fprintf( ...
    'sof_v_t / sof_h_t = %.6g / %.6g\n', ...
    worst.sof_v_t, ...
    worst.sof_h_t);

fprintf( ...
    'Cs eig min / max  = %.6e / %.6e\n', ...
    worst.minCsEig, ...
    worst.maxCsEig);

fprintf( ...
    'Missing per depth:\n');

disp( ...
    worst.missing_count');


%% ================================================================
% 21. Pattern summary
% ================================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf(' PATTERN SUMMARY\n');
fprintf('============================================================\n');

for ip = 1:NPATTERN

    name = pattern_names{ip};

    idx = strcmp( ...
        pattern_record, ...
        name);

    fprintf( ...
        '%-28s N=%4d  max=%.3e  mean=%.3e\n', ...
        name, ...
        sum(idx), ...
        max(abs_diff_all(idx)), ...
        mean(abs_diff_all(idx)));

end


%% ================================================================
% 22. M summary
% ================================================================

fprintf('\n');
fprintf('============================================================\n');
fprintf(' PARAMETER NUMBER SUMMARY\n');
fprintf('============================================================\n');

for Mtest = 2:4

    idx = ...
        M_all_record == Mtest;

    fprintf( ...
        'M=%d  N=%4d  max=%.3e  mean=%.3e\n', ...
        Mtest, ...
        sum(idx), ...
        max(abs_diff_all(idx)), ...
        mean(abs_diff_all(idx)));

end


%% ================================================================
% 23. Final PASS / FAIL
% ================================================================

fprintf('\n');
fprintf('============================================================\n');

if failure_count == 0

    fprintf( ...
        'PASS: BAYTOWN HADAMARD LIKELIHOOD MATCHES THE\n');

    fprintf( ...
        '      FULL KRONECKER ANALYTICAL LIKELIHOOD.\n');

else

    fprintf( ...
        'FAIL: %d CASES EXCEED NUMERICAL TOLERANCE.\n', ...
        failure_count);

end

fprintf('============================================================\n');


%% ================================================================
% LOCAL FUNCTIONS
% ================================================================

function value = log_uniform(vmin,vmax)

value = 10^( ...
    log10(vmin) ...
    + rand*( ...
    log10(vmax)-log10(vmin)));

end


%% ================================================================

function Cs = build_Cs(M,mode)

switch lower(mode)

    case 'weak'

        rho = 0.20;

        R = ...
            (1-rho)*eye(M) ...
            + rho*ones(M);


    case 'strong'

        rho = 0.90;

        R = ...
            (1-rho)*eye(M) ...
            + rho*ones(M);


    case 'near'

        rho = 0.98;

        R = ...
            (1-rho)*eye(M) ...
            + rho*ones(M);


    case 'negative'

        rho = -0.65;

        R = toeplitz( ...
            rho.^(0:M-1));


    case 'random'

        A = randn(M);

        R = ...
            A*A' ...
            + 0.20*eye(M);

        d = sqrt(diag(R));

        R = ...
            R./(d*d');


    otherwise

        error( ...
            'Unknown Cs mode: %s', ...
            mode);

end


% Different marginal variances
scales = ...
    linspace(0.7,1.5,M)';

D = diag(scales);

Cs = D*R*D;

Cs = ...
    (Cs+Cs')*0.5;


if min(eig(Cs)) <= 0

    error( ...
        'Generated Cs is not SPD.');

end

end


%% ================================================================

function missing = make_missing_pattern( ...
    nv,nh,M,pattern_type)

ncol = nh*M;


switch lower(pattern_type)

    %% ------------------------------------------------------------
    case 'random'

        p = ...
            0.02 + 0.58*rand;

        missing = ...
            rand(nv,ncol) < p;


    %% ------------------------------------------------------------
    case 'exact_nh_missing'

        missing = ...
            rand(nv,ncol) < 0.15;

        r = min(3,nv);

        missing(r,:) = false;

        idx = randperm( ...
            ncol, ...
            nh);

        missing(r,idx) = true;


    %% ------------------------------------------------------------
    case 'full_depth_missing'

        missing = ...
            rand(nv,ncol) < 0.15;

        r = min(3,nv);

        missing(r,:) = true;


    %% ------------------------------------------------------------
    case 'parameter_block'

        missing = ...
            rand(nv,ncol) < 0.10;

        pid = randi(M);

        cols = ...
            (pid-1)*nh ...
            + (1:nh);

        rows = ...
            3:max(3,floor(0.80*nv));

        missing(rows,cols) = true;


    %% ------------------------------------------------------------
    case 'hole_block'

        missing = ...
            rand(nv,ncol) < 0.10;

        hid = randi(nh);

        cols = ...
            hid ...
            + (0:M-1)*nh;

        missing(3:nv,cols) = true;


    %% ------------------------------------------------------------
    case 'heavy_random'

        missing = ...
            rand(nv,ncol) < 0.65;


    %% ------------------------------------------------------------
    case 'checkerboard'

        missing = ...
            false(nv,ncol);

        for iz = 1:nv

            for jc = 1:ncol

                missing(iz,jc) = ...
                    mod(iz+2*jc,4) == 0;

            end

        end


    %% ------------------------------------------------------------
    case 'extreme_sparse'

        missing = ...
            true(nv,ncol);

        % one observation per partial depth
        for iz = 3:nv

            jc = randi(ncol);

            missing(iz,jc) = false;

        end


    %% ------------------------------------------------------------
    case 'different_parameter_rates'

        missing = ...
            false(nv,ncol);

        for pid = 1:M

            p = ...
                0.05 ...
                + 0.70*(pid-1)/max(M-1,1);

            cols = ...
                (pid-1)*nh ...
                + (1:nh);

            missing(:,cols) = ...
                rand(nv,nh) < p;

        end


    otherwise

        error( ...
            'Unknown missing pattern: %s', ...
            pattern_type);

end


%% ---------------------------------------------------------------
% Retain two complete depth rows
% ---------------------------------------------------------------

n_complete = min(2,nv);

missing(1:n_complete,:) = false;


%% ---------------------------------------------------------------
% Ensure at least one partial non-lattice row
% ---------------------------------------------------------------

missing_count = ...
    sum(missing,2);

partial = ...
    missing_count > 0 ...
    & missing_count < ncol;

if ~any(partial)

    r = min(3,nv);

    missing(r,:) = false;
    missing(r,1) = true;

end

end