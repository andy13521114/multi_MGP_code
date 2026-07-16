clc; clear; close all;
tic_total = tic;

time_tmcmc = NaN;
time_crf   = NaN;
%% ===================== USER SETTINGS =====================
T_mcmc   = 1000;                 % Global TMCMC samples
Scenario = 1;

param_name_all = {'fs','qt','u2'};
log_ind_all    = [1 1 0];       % fs/log, qt/log, u2/linear
use_param      = [1 1 1 ];       % [1 1 1] = all three params

zmin_keep = 6.0;
zmax_keep = 10.0;


DO_TMCMC_EACH_FOLD = true;
T_mcmc_fast = 80;
tmcmc_beta = 0.5;

CI_LO = 2.5;
CI_HI = 97.5;

N_CRF_PLOT = 3;             % plot 20 CRF realizations; change to 10 if you only want 10

unit_name = 'kPa';              % all raw-space outputs and metrics are in kPa

% ===== paper-like role setting =====
green_idx = [94 92 91 89 88];
red_idx   = [86  84 94 91 88 81 79];
% Output: only one MGPR MAT file saved directly in the current MATLAB folder.
save_dir = pwd;
out_file = fullfile(save_dir, 'Bounds_Multi3Para_kPa.mat');

%rng(1);

%% ========================================================================
% (0) LOAD DATA
%% ========================================================================
Sx = load('XID_storage.mat');
Sy = load('YID_storage.mat');
Sc = load('CPTR_storage.mat');

if isfield(Sx,'XID'), X_all = double(Sx.XID(:));
else, error('XID_storage.mat does not contain XID'); end

if isfield(Sy,'YID'), Y_all = double(Sy.YID(:));
else, error('YID_storage.mat does not contain YID'); end

if isfield(Sc,'CPTR'), CPTR = Sc.CPTR;
else, error('CPTR_storage.mat does not contain CPTR'); end

assert(iscell(CPTR), 'CPTR must be a cell array');
nh = numel(CPTR);
assert(numel(X_all)==nh && numel(Y_all)==nh, 'Coordinate length mismatch');

X_all = X_all + (rand(size(X_all))-0.5)*1e-6;
Y_all = Y_all + (rand(size(Y_all))-0.5)*1e-6;

fprintf('Total holes loaded: nh = %d\n', nh);

%% ========================================================================
% (0.5) DEFINE PAPER-LIKE 99-HOLE SETUP
%% ========================================================================
all_idx = (1:nh)';

green_idx = unique(green_idx(:)', 'stable');
red_idx   = unique(red_idx(:)', 'stable');
selected_13_idx = unique([green_idx red_idx], 'stable');

assert(numel(green_idx)==5);
assert(numel(red_idx)==7);
%assert(numel(selected_13_idx)==13);
%assert(isempty(intersect(green_idx, red_idx)));

dist_blue = hypot(X_all - 50, Y_all - 10);
[blue_dist, blue_idx] = min(dist_blue);
fprintf('Blue outlier: hole=%d, coord=(%.3f,%.3f), dist=%.3f\n', ...
    blue_idx, X_all(blue_idx), Y_all(blue_idx), blue_dist);
assert(~ismember(blue_idx, selected_13_idx));

usable_idx = setdiff(all_idx, blue_idx, 'stable');
extra_exclude = [12, 80, 82, 83, 85, 89, 92 17 25 40 42 75 69];   % 異常孔
usable_idx = setdiff(usable_idx, extra_exclude, 'stable');
train_idx  = setdiff(usable_idx, selected_13_idx, 'stable');
train_idx  = setdiff(usable_idx, selected_13_idx, 'stable');
test_holes = red_idx;



fprintf('Training holes (gray) = %d\n', numel(train_idx));
fprintf('Green holdout holes   = %d\n', numel(green_idx));
fprintf('Red validation holes  = %d\n', numel(test_holes));

%% ========================================================================
% (1) BUILD COMMON LATTICE
% col 3=z, col 5=fs, col 6=u2, col 9=qt
% 原始 CPTR 欄位為 MPa，這裡統一轉成 kPa
%% ========================================================================
dz_grid   = 0.02;
z_grid = (zmin_keep:dz_grid:zmax_keep)';
nz0 = numel(z_grid);

fs_lat = nan(nz0, nh);
qt_lat = nan(nz0, nh);
u2_lat = nan(nz0, nh);

for h = 1:nh
    A = CPTR{h};
    if isempty(A) || ~isnumeric(A) || size(A,2) < 9, continue; end

    z0  = double(A(:,3));

    % 原始 CPTR 欄位為 MPa，統一轉成 kPa
    fs0 = double(A(:,5)) * 1000;   % fs: MPa -> kPa
    u20 = double(A(:,6)) * 1000;   % u2: MPa -> kPa
    qt0 = double(A(:,9)) * 1000;   % qt: MPa -> kPa

    z0  = z0(:);
    fs0 = fs0(:);
    u20 = u20(:);
    qt0 = qt0(:);

    if ~any(isfinite(z0)), continue; end

    if median(z0(isfinite(z0))) < 0
        dep = abs(z0);
    else
        dep = z0 - min(z0(isfinite(z0)));
    end

    [dep, ord] = sort(dep, 'ascend');
    fs0=fs0(ord); qt0=qt0(ord); u20=u20(ord);

    dep_round = round(dep/dz_grid)*dz_grid;
    [dep_u, ~, ic] = unique(dep_round);

    fs_u = accumarray(ic, fs0, [], @nanmean, NaN);
    qt_u = accumarray(ic, qt0, [], @nanmean, NaN);
    u2_u = accumarray(ic, u20, [], @nanmean, NaN);

    m1 = isfinite(dep_u) & isfinite(fs_u);
    if nnz(m1)>=2
        fs_lat(:,h) = interp1(dep_u(m1), fs_u(m1), z_grid, 'linear', NaN);
    end
    m2 = isfinite(dep_u) & isfinite(qt_u);
    if nnz(m2)>=2
        qt_lat(:,h) = interp1(dep_u(m2), qt_u(m2), z_grid, 'linear', NaN);
    end
    m3 = isfinite(dep_u) & isfinite(u2_u);
    if nnz(m3)>=2
        u2_lat(:,h) = interp1(dep_u(m3), u2_u(m3), z_grid, 'linear', NaN);
    end
end
% 扣靜水壓，單位 kPa
% gamma_w = 9.81 kPa/m, z_grid = m
% u0_grid = kPa
gamma_w = 9.81;
u0_grid = gamma_w * z_grid;
u2_lat  = u2_lat - u0_grid;
% positivity for log params
if log_ind_all(1)>0.5, fs_lat(fs_lat<=0) = NaN; end
if log_ind_all(2)>0.5, qt_lat(qt_lat<=0) = NaN; end
% u2 is linear, no positivity constraint

fprintf('Initial lattice: nz=%d, z in [%.3f, %.3f]\n', nz0, min(z_grid), max(z_grid));
fprintf('Raw-space unit after conversion: %s\n', unit_name);

%% ========================================================================
% (1.2) FILTER SELECTED PARAMETERS
%% ========================================================================
active_idx = find(use_param == 1);
M = numel(active_idx);
if M == 0, error('請至少選擇一個參數！'); end

param_name = param_name_all(active_idx);
log_ind    = log_ind_all(active_idx);

raw_all_temp = {fs_lat, qt_lat, u2_lat};
raw_all = cell(1, M);
for i = 1:M
    raw_all{i} = raw_all_temp{active_idx(i)};
end

fprintf('\n>>> Selected: %s (M=%d) <<<\n', strjoin(param_name,', '), M);

%% ========================================================================
% (1.5) REMOVE BAD DEPTH ROWS (based on training holes only)
%% ========================================================================
train_data_subset = [];
for p = 1:M
    train_data_subset = [train_data_subset, raw_all{p}(:, train_idx)];
end

good_rows = find(all(isfinite(train_data_subset), 2));
fprintf('Complete rows in 86 training holes = %d / %d\n', numel(good_rows), nz0);

if numel(good_rows) < 20
    error('Too few complete depth rows after filtering.');
end

z_grid = z_grid(good_rows);
for p = 1:M
    raw_all{p} = raw_all{p}(good_rows, :);
end
nz = numel(z_grid);
fprintf('After cleaning: nz=%d, z in [%.3f, %.3f]\n', nz, min(z_grid), max(z_grid));

%% ========================================================================
% (2) TRAIN / TEST SPLIT
%% ========================================================================
train_idx  = unique(train_idx, 'stable');
test_holes = unique(test_holes, 'stable');
assert(isempty(intersect(train_idx, test_holes)));

num_groups = numel(test_holes);   % ← 確保在此定義
nh_tr      = numel(train_idx);

fprintf('\nTrain=%d holes, Test=%d holes\n', nh_tr, num_groups);

raw_tr = cell(1,M);
for p = 1:M
    raw_tr{p} = raw_all{p}(:, train_idx);
end
X_tr = X_all(train_idx);
Y_tr = Y_all(train_idx);

%% ========================================================================
% (4) TRANSFORM: ln + Z-score (from training holes only)
%% ========================================================================
Z_map   = cell(1,M);
Zdat_tr = cell(1,M);

for p = 1:M
    vtr = raw_tr{p}(:);
    if log_ind(p)>0.5
        vtr(vtr<=0) = NaN;
        vtr = log(vtr);
    end
    [~, mp] = zscore_forward(vtr);
    ztr = apply_zscore_transform(raw_tr{p}, mp, log_ind(p)>0.5);
    Z_map{p}   = mp;
    Zdat_tr{p} = ztr;
end

t_tr = zeros(nz*nh_tr, M);
for p = 1:M
    t_tr(:,p) = reshape(Zdat_tr{p}, [], 1);
end
fprintf('NaN count in t_tr = %d\n', nnz(~isfinite(t_tr)));

%% ========================================================================
% (5) DISTANCE MATRICES
%% ========================================================================
temp_x = abs(X_tr*ones(1,nh_tr) - (X_tr*ones(1,nh_tr))');
temp_y = abs(Y_tr*ones(1,nh_tr) - (Y_tr*ones(1,nh_tr))');
temp_h = sqrt(temp_x.^2 + temp_y.^2);
temp_z = abs(z_grid*ones(1,nz) - (z_grid*ones(1,nz))');

z_ele  = z_grid;
nz_ele = numel(z_ele);

X_ele_dummy = X_all(test_holes(1));
Y_ele_dummy = Y_all(test_holes(1));
temp_x_ele = abs([X_tr;X_ele_dummy] - [X_tr;X_ele_dummy]');
temp_y_ele = abs([Y_tr;Y_ele_dummy] - [Y_tr;Y_ele_dummy]');
temp_h_ele = sqrt(temp_x_ele.^2 + temp_y_ele.^2);
temp_z_ele = abs([z_grid;z_ele] - [z_grid;z_ele]');

y = struct();
y.z = z_grid; y.X = X_tr; y.Y = Y_tr; y.t = t_tr;
y.temp_h = temp_h; y.temp_z = temp_z;
y.temp_h_ele = temp_h_ele; y.temp_z_ele = temp_z_ele;

%% ========================================================================
% (6) ESTIMATE Cs + TMCMC
%% ========================================================================
disp('>> Starting TMCMC on 86 training holes...');
tic_tmcmc = tic;
%load('Cs_CPT_0710.mat')
%Cs=Cs_CPT_sample;
Cs = estimate_Cs_from_cov(y.t, M);
%Cs=eye(2);
%Cs = [1 0.5415 ;0.5415 1];
Cs = makeSPD(Cs, 1e-8);

[phi_t_Cs, D] = eig(Cs);
[~, eigsort] = sort(diag(D), 'descend');
phi_t_Cs = phi_t_Cs(:, eigsort);

inv_Cs = inv(Cs);
L_Cs   = chol(Cs, 'lower');
y.eig_thresh = 0.999;

% bounds: [bhp, sofv, sofh, nuv, ahp, sofv_t, sofh_t]
% x_low = [-log(3),   log(0.1), log(0.3), log(0.3), -log(3),   log(max(temp_z(:))/5),  log(max(temp_h(:))/5)];
% x_up  = [-log(0.1), log(10),  log(20),  log(3),   -log(0.1), log(max(temp_z(:))*10), log(max(temp_h(:))*10)];
% x_low = [-log(1),   log(0.1), log(0.3), log(0.3), -log(2),   log(99999),  log(max(temp_h(:))/5)];
% x_up  = [-log(0.1), log(10),  log(20),  log(3),   -log(0.1), log(100000), log(max(temp_h(:))*10)];
x_low = [-log(3),   log(0.1), log(0.1), log(0.1), -log(3),   log(max(temp_z(:))/10),  log(max(temp_h(:))/10)];
x_up  = [-log(0.1), log(10),  log(20),  log(20),   -log(0.1), log(max(temp_z(:))*10), log(max(temp_h(:))*10)];

[x, ln_S, ~, ~, ~] = iTMCMC_fun_mod1('GP_Matern_3D', y, x_low, x_up, T_mcmc, tmcmc_beta, Cs);
Ns = size(x,1);
time_tmcmc = toc(tic_tmcmc);

disp('>> TMCMC Done.');
fprintf('TMCMC runtime = %.2f sec = %.2f min = %.2f hr\n', ...
    time_tmcmc, time_tmcmc/60, time_tmcmc/3600);

bhp    = 1./exp(x(:,1));
sofv   = exp(x(:,2));
sofh   = exp(x(:,3));
nuv    = exp(x(:,4));
nuh    = nuv;
ahp    = 1./exp(x(:,5));
sofv_t = exp(x(:,6));
sofh_t = exp(x(:,7));

%% ========================================================================
% METRICS & STORAGE INIT
%% ========================================================================
metrics.RMSE_raw = nan(num_groups, M);
metrics.R2_raw   = nan(num_groups, M);
metrics.RMSE_Z   = nan(num_groups, M);
metrics.R2_Z     = nan(num_groups, M);

STORE_raw_y_true = cell(num_groups, 1);
STORE_raw_lo     = cell(num_groups, 1);
STORE_raw_hi     = cell(num_groups, 1);
STORE_raw_mean   = cell(num_groups, 1);
STORE_raw_sample = cell(num_groups, 1);
STORE_Z_y_true   = cell(num_groups, 1);
STORE_Z_lo       = cell(num_groups, 1);
STORE_Z_hi       = cell(num_groups, 1);
STORE_Z_mean     = cell(num_groups, 1);
STORE_Z_sample   = cell(num_groups, 1);
STORE_crf_draw_id = nan(num_groups, 1);

% Store multiple CRF realizations for plotting.
% Dimension for each cell: nz × M × n_take, where n_take <= N_CRF_PLOT.
STORE_raw_samples20   = cell(num_groups, 1);
STORE_Z_samples20     = cell(num_groups, 1);
STORE_crf_draw_ids20  = cell(num_groups, 1);

%% ========================================================================
% (7) PREDICTION LOOP
%% ========================================================================
%% ========================================================================
% (7) PREDICTION LOOP / CRF SIMULATION TIME
%% ========================================================================
tic_crf = tic;

for group_i = 1:num_groups
    hold_i = test_holes(group_i);
    fprintf('\n=== Predicting RED hole %d (%d/%d) ===\n', hold_i, group_i, num_groups);

    raw_ho  = cell(1,M);
    Zdat_ho = cell(1,M);
    for p = 1:M
        raw_ho{p} = raw_all{p}(:, hold_i);
        zho = apply_zscore_transform(raw_ho{p}, Z_map{p}, log_ind(p)>0.5);
        Zdat_ho{p} = zho(:);
    end

    X_ele = X_all(hold_i);
    Y_ele = Y_all(hold_i);
    temp_x_e = abs([X_tr;X_ele] - [X_tr;X_ele]');
    temp_y_e = abs([Y_tr;Y_ele] - [Y_tr;Y_ele]');
    y.temp_h_ele = sqrt(temp_x_e.^2 + temp_y_e.^2);

    t_pred_model_samp = nan(nz_ele, M, Ns);
    reshape_data = reshape(y.t, nz*nh_tr, M);

    for k = 1:Ns
        [y.phiz, y.phih, y.phiz_ele, y.phih_ele, ~, ln_alpha] = ...
            GP_matrices_Step3(ahp(k), sofv_t(k), sofh_t(k), y, Cs);

        A = diag(exp(ln_alpha));

        R_h = Matern_R(nuh(k), sofh(k), temp_h) + 1e-6*eye(nh_tr);
        R_z = Matern_R(nuv(k), sofv(k), temp_z) + 1e-6*eye(nz);
        inv_Rh = R_h \ eye(nh_tr);
        inv_Rz = R_z \ eye(nz);

        temp_term = kron(y.phih'*inv_Rh, y.phiz'*inv_Rz) * reshape_data * inv_Cs' * phi_t_Cs;
        if any(~isfinite(temp_term(:))), continue; end

        K = A + (1/bhp(k)) * kron(phi_t_Cs'*inv_Cs*phi_t_Cs, ...
            kron(y.phih'*inv_Rh*y.phih, y.phiz'*inv_Rz*y.phiz));
        K = (K+K')/2 + 1e-10*eye(size(K));

        try
            L = chol(K,'lower');
        catch
            K = (K+K')/2 + 1e-8*eye(size(K));
            try, L = chol(K,'lower'); catch, continue; end
        end

        rhs  = (1/bhp(k)) * temp_term(:);
        muW  = L' \ (L \ rhs);
        w    = muW + L' \ (L \ randn(size(muW)));
        if any(~isfinite(w)), continue; end

        dd_train = kronmult2({phi_t_Cs, y.phih, y.phiz}, w);
        t_d = reshape_data(:) - dd_train;

        [E_X, L_h, L_z] = vertical_dense_stats( ...
            sofv(k), sofh(k), nuv(k), nuh(k), ...
            X_ele, Y_ele, z_ele, X_tr, Y_tr, z_grid, t_d, M);

        Q = {L_Cs, L_h, L_z};
        noise_pred = E_X + sqrt(bhp(k)) * kronmult2(Q, randn(numel(Y_ele)*numel(z_ele)*M, 1));
        d_pred = kronmult2({phi_t_Cs, y.phih_ele, y.phiz_ele}, w);

        t_pred = d_pred(:) + noise_pred(:);
        t_pred_model_samp(:,:,k) = reshape(t_pred, nz_ele, M);
    end

    t_pred_mean_model = mean(t_pred_model_samp, 3, 'omitnan');
    t_pred_lo_model   = prctile(t_pred_model_samp, CI_LO, 3);
    t_pred_hi_model   = prctile(t_pred_model_samp, CI_HI, 3);

    %% back-transform
    y_true_raw = nan(nz_ele, M);
    y_true_Z   = nan(nz_ele, M);
    for p = 1:M
        y_true_raw(:,p) = raw_ho{p}(:);
        y_true_Z(:,p)   = Zdat_ho{p}(:);
    end

    t_pred_mean_raw   = nan(nz_ele, M);
    t_pred_lo_raw     = nan(nz_ele, M);
    t_pred_hi_raw     = nan(nz_ele, M);
    t_pred_sample_raw = nan(nz_ele, M);   % one complete random CRF realization in raw/original scale
    t_pred_sample_Z   = nan(nz_ele, M);   % the same CRF realization in Z-score space

    % ====================================================================
    % Choose multiple posterior draws as CRF realizations.
    % The same draw IDs are used for fs, qt, and u2, so each realization
    % preserves the cross-parameter posterior consistency.
    % ====================================================================
    valid_draw_mask = reshape(all(all(isfinite(t_pred_model_samp), 1), 2), [], 1);
    valid_draw_all = find(valid_draw_mask);

    if isempty(valid_draw_all)
        % Fallback: choose draws that contain any finite value.
        valid_draw_mask2 = reshape(any(any(isfinite(t_pred_model_samp), 1), 2), [], 1);
        valid_draw_all = find(valid_draw_mask2);
    end

    if isempty(valid_draw_all)
        valid_draw_all = 1;
    end

    n_take = min(N_CRF_PLOT, numel(valid_draw_all));

    % Use fixed first n_take draws for reproducibility.
    % If you want random CRFs every run, replace the next line with:
    % draw_ids20 = valid_draw_all(randperm(numel(valid_draw_all), n_take));
    draw_ids20 = valid_draw_all(1:n_take);

    % The first selected CRF is still saved to the old fields:
    % realization / CRF_sample / sample.
    valid_draw = draw_ids20(1);
    STORE_crf_draw_id(group_i) = valid_draw;
    STORE_crf_draw_ids20{group_i} = draw_ids20(:);

    Z_one_draw = t_pred_model_samp(:,:,valid_draw);
    t_pred_sample_Z = Z_one_draw;

    % Store multiple Z-space CRF samples: nz × M × n_take
    t_pred_samples20_Z   = t_pred_model_samp(:,:,draw_ids20);
    t_pred_samples20_raw = nan(nz_ele, M, n_take);

    for p = 1:M
        Z_samp = reshape(t_pred_model_samp(:,p,:), nz_ele, Ns);

        raw_samp = Z_samp * Z_map{p}.sd + Z_map{p}.mu;
        raw_one  = Z_one_draw(:,p) * Z_map{p}.sd + Z_map{p}.mu;

        % Multiple CRF samples for plotting.
        Z_20 = reshape(t_pred_samples20_Z(:,p,:), nz_ele, n_take);
        raw_20 = Z_20 * Z_map{p}.sd + Z_map{p}.mu;

        if log_ind(p)>0.5
            raw_samp = exp(raw_samp);
            raw_one  = exp(raw_one);
            raw_20   = exp(raw_20);
        end

        t_pred_sample_raw(:,p) = raw_one;
        t_pred_samples20_raw(:,p,:) = reshape(raw_20, nz_ele, 1, n_take);

        t_pred_mean_raw(:,p)   = median(raw_samp, 2, 'omitnan');
        t_pred_lo_raw(:,p)     = prctile(raw_samp, CI_LO, 2);
        t_pred_hi_raw(:,p)     = prctile(raw_samp, CI_HI, 2);
    end

    for p = 1:M
        [rmse_raw, r2_raw] = compute_rmse_r2(y_true_raw(:,p), t_pred_mean_raw(:,p));
        metrics.RMSE_raw(group_i,p) = rmse_raw;
        metrics.R2_raw(group_i,p)   = r2_raw;
        [rmse_z, r2_z] = compute_rmse_r2(y_true_Z(:,p), t_pred_mean_model(:,p));
        metrics.RMSE_Z(group_i,p) = rmse_z;
        metrics.R2_Z(group_i,p)   = r2_z;
    end

    STORE_raw_y_true{group_i} = y_true_raw;
    STORE_raw_lo{group_i}     = t_pred_lo_raw;
    STORE_raw_hi{group_i}     = t_pred_hi_raw;
    STORE_raw_mean{group_i}   = t_pred_mean_raw;
    STORE_raw_sample{group_i} = t_pred_sample_raw;
    STORE_Z_y_true{group_i}   = y_true_Z;
    STORE_Z_lo{group_i}       = t_pred_lo_model;
    STORE_Z_hi{group_i}       = t_pred_hi_model;
    STORE_Z_mean{group_i}     = t_pred_mean_model;
    STORE_Z_sample{group_i}   = t_pred_sample_Z;

    % Multiple CRF realizations for the second plotting section.
    STORE_raw_samples20{group_i} = t_pred_samples20_raw;
    STORE_Z_samples20{group_i}   = t_pred_samples20_Z;

    fprintf('Finished predicting RED hole %d.\n', hold_i);
end
time_crf = toc(tic_crf);

fprintf('\nCRF / prediction runtime = %.2f sec = %.2f min = %.2f hr\n', ...
    time_crf, time_crf/60, time_crf/3600);
time_total = toc(tic_total);
%% ========================================================================
% (8) EXPORT STRUCT ONLY
%     Content:
%       z
%       test_holes
%       param_name
%       Hxx.fs / Hxx.qt / Hxx.u2:
%           actual
%           median
%           low95
%           high95
%           realization  = one random CRF realization
%           CRF_sample   = same as realization, kept for compatibility
%           sample       = same as realization, kept for compatibility
%
%     注意：
%       這裡不存檔，只整理 ExportMGPR struct。
%% ========================================================================
ExportMGPR = struct();

ExportMGPR.model_name = 'MGPR';
ExportMGPR.unit       = unit_name;
ExportMGPR.z          = z_ele(:);
ExportMGPR.test_holes = test_holes(:);
ExportMGPR.param_name = param_name;
ExportMGPR.log_ind    = log_ind;
ExportMGPR.active_idx = active_idx;
ExportMGPR.CI_percentile = [CI_LO CI_HI];
ExportMGPR.crf_draw_id   = STORE_crf_draw_id;
ExportMGPR.N_CRF_PLOT    = N_CRF_PLOT;
ExportMGPR.crf_draw_ids20 = STORE_crf_draw_ids20;

for group_i = 1:num_groups
    h_id  = test_holes(group_i);
    h_tag = sprintf('H%d', h_id);

    ExportMGPR.(h_tag).hole_id = h_id;
    ExportMGPR.(h_tag).x = X_all(h_id);
    ExportMGPR.(h_tag).y = Y_all(h_id);

    for p = 1:M
        pn = param_name{p};

        ExportMGPR.(h_tag).(pn).actual      = STORE_raw_y_true{group_i}(:,p);
        ExportMGPR.(h_tag).(pn).median      = STORE_raw_mean{group_i}(:,p);
        ExportMGPR.(h_tag).(pn).low95       = STORE_raw_lo{group_i}(:,p);
        ExportMGPR.(h_tag).(pn).high95      = STORE_raw_hi{group_i}(:,p);

        % One random CRF realization, same posterior draw across all parameters.
        ExportMGPR.(h_tag).(pn).realization = STORE_raw_sample{group_i}(:,p);
        ExportMGPR.(h_tag).(pn).CRF_sample  = STORE_raw_sample{group_i}(:,p);
        ExportMGPR.(h_tag).(pn).sample      = STORE_raw_sample{group_i}(:,p);

        % Multiple CRF realizations in raw/original scale.
        % Dimension: nz × n_take. Default n_take = 20 unless fewer valid draws exist.
        ExportMGPR.(h_tag).(pn).CRF_samples20 = squeeze(STORE_raw_samples20{group_i}(:,p,:));
    end
end

% Save ExportMGPR in current folder
save(out_file, '-struct', 'ExportMGPR');

fprintf('\n[完成] ExportMGPR 已整理完成。若要儲存 MAT，請取消下一行 save 註解：%s\n', out_file);


%% ========================================================================
% (9) PRINT CONDITIONAL PREDICTION METRICS ONLY
%     1. Marginal CI coverage
%     2. MAE
%     3. RMSE
%        Here RMSE is computed as:
%        RMSE = sqrt(mean((median - actual)^2 + sigma_CI^2))
%        sigma_CI = (CI_high - CI_low) / (2*zcrit)
%     4. R2
%     5. Corr2
%     6. Mean CI width
%     7. Normalized CI width
%     8. Joint CI coverage in Z-space
%
%     不存檔，只 print。
%% ========================================================================

if ~exist('unit_name','var') || isempty(unit_name)
    unit_name = 'kPa';
end

% ===== z critical value for converting 95% CI width to sigma =====
% For CI_LO=2.5 and CI_HI=97.5, zcrit_rmse ≈ 1.96
zcrit_rmse = -sqrt(2) * erfcinv(2*(CI_HI/100));

if ~isfinite(zcrit_rmse) || zcrit_rmse <= 0
    zcrit_rmse = 1.95996398454005;
end

fprintf('\n%s\n', repmat('=',1,125));
fprintf('  MGPR Conditional Prediction Metrics\n');
fprintf('  Test holes: ');
fprintf('H%d ', test_holes);
fprintf('\n');
fprintf('  Marginal CI: %.1f%% - %.1f%% percentile interval\n', CI_LO, CI_HI);
fprintf('  RMSE definition: sqrt(mean((median-actual)^2 + sigma_CI^2))\n');
fprintf('  sigma_CI = (CI_high - CI_low) / (2*zcrit), zcrit = %.5f\n', zcrit_rmse);
fprintf('  Joint CI: 95%% chi-square ellipsoid in normalized/log-transformed Z-space\n');
fprintf('%s\n', repmat('=',1,125));

fprintf('\n===== RAW SCALE CHECK (all validation holes) =====\n');
for p = 1:M
    yy = [];
    for group_i = 1:num_groups
        yy = [yy; STORE_raw_y_true{group_i}(:,p)]; %#ok<AGROW>
    end
    yy = yy(isfinite(yy));

    if isempty(yy)
        fprintf('%s actual raw scale: no finite data\n', param_name{p});
    else
        fprintf('%s actual raw scale: min=%.4g, median=%.4g, max=%.4g %s\n', ...
            param_name{p}, min(yy), median(yy), max(yy), unit_name);
    end
end
fprintf('==================================================\n\n');


%% ========================================================================
%  A. Marginal CI metrics: pooled over all test holes and all depths
%% ========================================================================

MET_PRINT = struct();
MET_PRINT.RMSE_definition = 'sqrt(mean((median-actual)^2 + sigma_CI^2)), sigma_CI=(CI_high-CI_low)/(2*zcrit)';
MET_PRINT.zcrit_rmse = zcrit_rmse;

fprintf('\n%s\n', repmat('-',1,125));
fprintf('  Marginal Conditional Prediction Metrics: pooled over all test holes and depths\n');
fprintf('%s\n', repmat('-',1,125));

fprintf('%-8s | %8s | %12s | %12s | %10s | %10s | %12s | %12s | %12s\n', ...
    'Param', 'N', 'MAE', 'RMSE', 'R2', 'Corr2', 'CI cover', 'Mean width', 'Norm width');

fprintf('%s\n', repmat('-',1,125));

for p = 1:M

    Y_all_p    = [];
    Pred_all_p = [];
    Lo_all_p   = [];
    Hi_all_p   = [];

    for group_i = 1:num_groups
        yv = STORE_raw_y_true{group_i}(:,p);
        pv = STORE_raw_mean{group_i}(:,p);
        lv = STORE_raw_lo{group_i}(:,p);
        hv = STORE_raw_hi{group_i}(:,p);

        valid = isfinite(yv) & isfinite(pv) & isfinite(lv) & isfinite(hv);

        Y_all_p    = [Y_all_p;    yv(valid)];
        Pred_all_p = [Pred_all_p; pv(valid)];
        Lo_all_p   = [Lo_all_p;   lv(valid)];
        Hi_all_p   = [Hi_all_p;   hv(valid)];
    end

    Np = numel(Y_all_p);

    if Np == 0
        mae_p        = NaN;
        rmse_p       = NaN;
        r2_p         = NaN;
        corr2_p      = NaN;
        cov_p        = NaN;
        mean_width_p = NaN;
        norm_width_p = NaN;
    else
        err_p = Pred_all_p - Y_all_p;

        % ===== MAE: point estimate error =====
        mae_p = mean(abs(err_p));

        % ===== RMSE: squared point error + predictive variance =====
        width_p = Hi_all_p - Lo_all_p;
        width_p = max(width_p, 0);

        sigma_ci_p = width_p / (2*zcrit_rmse);

        valid_rmse = isfinite(err_p) & ...
                     isfinite(sigma_ci_p) & ...
                     sigma_ci_p >= 0;

        if nnz(valid_rmse) == 0
            rmse_p = NaN;
        else
            rmse_p = sqrt(mean(err_p(valid_rmse).^2 + sigma_ci_p(valid_rmse).^2));
        end

        % ===== R2: still based on point estimate median =====
        ss_res = sum((Y_all_p - Pred_all_p).^2);
        ss_tot = sum((Y_all_p - mean(Y_all_p)).^2);

        if ss_tot < 1e-12
            r2_p = NaN;
        else
            r2_p = 1 - ss_res / ss_tot;
        end

        % ===== Corr2: still based on point estimate median =====
        if Np >= 3 && std(Y_all_p) > 0 && std(Pred_all_p) > 0
            Ctmp = corrcoef(Y_all_p, Pred_all_p);
            corr2_p = Ctmp(1,2)^2;
        else
            corr2_p = NaN;
        end

        % ===== marginal CI coverage =====
        inside_p = (Y_all_p >= Lo_all_p) & (Y_all_p <= Hi_all_p);
        cov_p = mean(inside_p);

        % ===== CI width =====
        mean_width_p = mean(width_p);
        norm_width_p = mean_width_p / max(std(Y_all_p), eps);
    end

    MET_PRINT.marginal_pooled(p).param     = param_name{p};
    MET_PRINT.marginal_pooled(p).N         = Np;
    MET_PRINT.marginal_pooled(p).MAE       = mae_p;
    MET_PRINT.marginal_pooled(p).RMSE      = rmse_p;
    MET_PRINT.marginal_pooled(p).R2        = r2_p;
    MET_PRINT.marginal_pooled(p).Corr2     = corr2_p;
    MET_PRINT.marginal_pooled(p).Coverage  = cov_p;
    MET_PRINT.marginal_pooled(p).MeanWidth = mean_width_p;
    MET_PRINT.marginal_pooled(p).NormWidth = norm_width_p;

    fprintf('%-8s | %8d | %12.5g | %12.5g | %10.4f | %10.4f | %10.2f%% | %12.5g | %12.5g\n', ...
        param_name{p}, Np, mae_p, rmse_p, r2_p, corr2_p, ...
        100*cov_p, mean_width_p, norm_width_p);
end

fprintf('%s\n', repmat('-',1,125));
fprintf('  Unit for MAE / RMSE / CI width: %s\n', unit_name);
fprintf('  RMSE includes predictive uncertainty estimated from marginal CI width.\n');
fprintf('  Norm width = mean CI width / std(actual)\n');


%% ========================================================================
%  B. Marginal CI metrics: by hole
%% ========================================================================

fprintf('\n%s\n', repmat('-',1,125));
fprintf('  Marginal Conditional Prediction Metrics: by hole\n');
fprintf('%s\n', repmat('-',1,125));

fprintf('%-8s | %-8s | %8s | %12s | %12s | %10s | %12s | %12s\n', ...
    'Hole', 'Param', 'N', 'MAE', 'RMSE', 'CI cover', 'Mean width', 'Norm width');

fprintf('%s\n', repmat('-',1,125));

for group_i = 1:num_groups

    h_id = test_holes(group_i);
    h_name = sprintf('H%d', h_id);

    for p = 1:M
        yv = STORE_raw_y_true{group_i}(:,p);
        pv = STORE_raw_mean{group_i}(:,p);
        lv = STORE_raw_lo{group_i}(:,p);
        hv = STORE_raw_hi{group_i}(:,p);

        valid = isfinite(yv) & isfinite(pv) & isfinite(lv) & isfinite(hv);

        yv = yv(valid);
        pv = pv(valid);
        lv = lv(valid);
        hv = hv(valid);

        Nh = numel(yv);

        if Nh == 0
            mae_h        = NaN;
            rmse_h       = NaN;
            cov_h        = NaN;
            mean_width_h = NaN;
            norm_width_h = NaN;
        else
            err_h = pv - yv;

            % ===== MAE: point estimate error =====
            mae_h = mean(abs(err_h));

            % ===== RMSE: squared point error + predictive variance =====
            width_h = hv - lv;
            width_h = max(width_h, 0);

            sigma_ci_h = width_h / (2*zcrit_rmse);

            valid_rmse_h = isfinite(err_h) & ...
                           isfinite(sigma_ci_h) & ...
                           sigma_ci_h >= 0;

            if nnz(valid_rmse_h) == 0
                rmse_h = NaN;
            else
                rmse_h = sqrt(mean(err_h(valid_rmse_h).^2 + sigma_ci_h(valid_rmse_h).^2));
            end

            % ===== marginal CI coverage =====
            inside_h = (yv >= lv) & (yv <= hv);
            cov_h = mean(inside_h);

            % ===== CI width =====
            mean_width_h = mean(width_h);
            norm_width_h = mean_width_h / max(std(yv), eps);
        end

        MET_PRINT.marginal_by_hole(group_i,p).hole      = h_id;
        MET_PRINT.marginal_by_hole(group_i,p).param     = param_name{p};
        MET_PRINT.marginal_by_hole(group_i,p).N         = Nh;
        MET_PRINT.marginal_by_hole(group_i,p).MAE       = mae_h;
        MET_PRINT.marginal_by_hole(group_i,p).RMSE      = rmse_h;
        MET_PRINT.marginal_by_hole(group_i,p).Coverage  = cov_h;
        MET_PRINT.marginal_by_hole(group_i,p).MeanWidth = mean_width_h;
        MET_PRINT.marginal_by_hole(group_i,p).NormWidth = norm_width_h;

        fprintf('%-8s | %-8s | %8d | %12.5g | %12.5g | %10.2f%% | %12.5g | %12.5g\n', ...
            h_name, param_name{p}, Nh, mae_h, rmse_h, ...
            100*cov_h, mean_width_h, norm_width_h);
    end
end

fprintf('%s\n', repmat('-',1,125));


%% ========================================================================
%  C. Joint CI coverage in Z-space
%
%     使用 Z-space:
%       fs, qt 已經 ln + z-score
%       u2 為 z-score
%
%     Joint CI 算法：
%       1. 每個深度取三個參數的真值 z_true
%       2. 取預測中心 z_mean
%       3. 用 marginal CI 寬度反推每個參數的 sigma
%       4. 用 Cs 的 correlation matrix 組合成 joint covariance
%       5. 計算 Mahalanobis distance:
%
%          D2 = (z_true - z_mean)' * inv(Sigma_joint) * (z_true - z_mean)
%
%       6. 若 D2 <= chi2inv(0.95, M)，則 joint CI 覆蓋
%
%     注意：
%       這是後處理版 joint CI。
%       若要完全 empirical joint CI，要在 prediction loop 中保留完整 posterior samples。
%% ========================================================================

JOINT_LEVEL = 0.95;

% 避免沒有 Statistics Toolbox：
% chi2inv(p,k) = 2 * gammaincinv(p,k/2)
chi2_thr = 2 * gammaincinv(JOINT_LEVEL, M/2);

% Normal 97.5% quantile
% norminv(0.975) = -sqrt(2)*erfcinv(2*0.975)
zcrit_1d = -sqrt(2) * erfcinv(2*(CI_HI/100));

if ~isfinite(zcrit_1d) || zcrit_1d <= 0
    zcrit_1d = 1.95996398454005;
end

% Cs correlation matrix
if exist('Cs','var') && all(size(Cs) == [M M])
    sdCs = sqrt(max(diag(Cs), eps));
    Ccorr = Cs ./ (sdCs * sdCs');
else
    Ccorr = eye(M);
end

Ccorr(~isfinite(Ccorr)) = 0;
Ccorr = max(min(Ccorr, 0.999999), -0.999999);
Ccorr(1:M+1:end) = 1;
Ccorr = (Ccorr + Ccorr') / 2;

% 修正為 SPD
[Vcorr, Dcorr] = eig(Ccorr);
dcorr = max(diag(Dcorr), 1e-8);
Ccorr = Vcorr * diag(dcorr) * Vcorr';
Ccorr = (Ccorr + Ccorr') / 2;

fprintf('\n%s\n', repmat('=',1,125));
fprintf('  Joint Conditional Prediction %.0f%% CI Summary\n', 100*JOINT_LEVEL);
fprintf('  Joint space: normalized/log-transformed Z-space\n');
fprintf('  Threshold: chi2inv(%.2f,%d) = %.4f\n', JOINT_LEVEL, M, chi2_thr);
fprintf('  Cross-parameter correlation: Corr(Cs)\n');
fprintf('%s\n', repmat('-',1,125));

fprintf('%-8s | %12s | %12s | %10s | %10s | %12s | %12s\n', ...
    'Hole', 'In/Total', 'Coverage', 'Target', 'Mean D2', 'P95 D2', 'Max D2');

fprintf('%s\n', repmat('-',1,125));

D2_pool = [];
inside_pool = [];

for group_i = 1:num_groups

    h_id = test_holes(group_i);
    h_name = sprintf('H%d', h_id);

    Z_true = STORE_Z_y_true{group_i};
    Z_mean = STORE_Z_mean{group_i};
    Z_lo   = STORE_Z_lo{group_i};
    Z_hi   = STORE_Z_hi{group_i};

    D2_hole = nan(size(Z_true,1),1);

    for iz = 1:size(Z_true,1)

        yt = Z_true(iz,:)';
        mu = Z_mean(iz,:)';
        lo = Z_lo(iz,:)';
        hi = Z_hi(iz,:)';

        % 用 marginal CI 寬度反推 sigma
        sig = (hi - lo) / (2*zcrit_1d);

        valid_joint = all(isfinite(yt)) && ...
                      all(isfinite(mu)) && ...
                      all(isfinite(sig)) && ...
                      all(sig > 0);

        if ~valid_joint
            continue;
        end

        Sigma_joint = diag(sig) * Ccorr * diag(sig);
        Sigma_joint = (Sigma_joint + Sigma_joint')/2 + 1e-10*eye(M);

        d = yt - mu;
        D2_hole(iz) = d' * (Sigma_joint \ d);
    end

    valid_D2 = isfinite(D2_hole);

    n_valid  = nnz(valid_D2);
    n_inside = nnz(D2_hole(valid_D2) <= chi2_thr);

    if n_valid > 0
        cov_joint = n_inside / n_valid;
        mean_D2   = mean(D2_hole(valid_D2));
        p95_D2    = prctile(D2_hole(valid_D2), 95);
        max_D2    = max(D2_hole(valid_D2));
    else
        cov_joint = NaN;
        mean_D2   = NaN;
        p95_D2    = NaN;
        max_D2    = NaN;
    end

    D2_pool = [D2_pool; D2_hole(valid_D2)];
    inside_pool = [inside_pool; D2_hole(valid_D2) <= chi2_thr];

    MET_PRINT.joint_by_hole(group_i).hole     = h_id;
    MET_PRINT.joint_by_hole(group_i).n_inside = n_inside;
    MET_PRINT.joint_by_hole(group_i).n_valid  = n_valid;
    MET_PRINT.joint_by_hole(group_i).coverage = cov_joint;
    MET_PRINT.joint_by_hole(group_i).mean_D2  = mean_D2;
    MET_PRINT.joint_by_hole(group_i).p95_D2   = p95_D2;
    MET_PRINT.joint_by_hole(group_i).max_D2   = max_D2;

    fprintf('%-8s | %5d/%-6d | %10.2f%% | %9.0f%% | %10.4f | %12.4f | %12.4f\n', ...
        h_name, n_inside, n_valid, 100*cov_joint, ...
        100*JOINT_LEVEL, mean_D2, p95_D2, max_D2);
end

fprintf('%s\n', repmat('-',1,125));

n_valid_pool  = numel(D2_pool);
n_inside_pool = nnz(inside_pool);

if n_valid_pool > 0
    cov_pool_joint = n_inside_pool / n_valid_pool;
    mean_D2_pool   = mean(D2_pool);
    p95_D2_pool    = prctile(D2_pool, 95);
    max_D2_pool    = max(D2_pool);
else
    cov_pool_joint = NaN;
    mean_D2_pool   = NaN;
    p95_D2_pool    = NaN;
    max_D2_pool    = NaN;
end

MET_PRINT.joint_pooled.n_inside = n_inside_pool;
MET_PRINT.joint_pooled.n_valid  = n_valid_pool;
MET_PRINT.joint_pooled.coverage = cov_pool_joint;
MET_PRINT.joint_pooled.mean_D2  = mean_D2_pool;
MET_PRINT.joint_pooled.p95_D2   = p95_D2_pool;
MET_PRINT.joint_pooled.max_D2   = max_D2_pool;
MET_PRINT.joint_pooled.chi2_thr = chi2_thr;

fprintf('%-8s | %5d/%-6d | %10.2f%% | %9.0f%% | %10.4f | %12.4f | %12.4f\n', ...
    'ALL', n_inside_pool, n_valid_pool, 100*cov_pool_joint, ...
    100*JOINT_LEVEL, mean_D2_pool, p95_D2_pool, max_D2_pool);

fprintf('%s\n', repmat('=',1,125));
fprintf('  Done printing marginal and joint metrics. No files were saved.\n');
fprintf('%s\n', repmat('=',1,125));


%% ========================================================================
% (10) VALIDATION PANEL PLOT (single model: MGPR)
%      一次畫 fs / qt / u2 三個參數
%      每個參數一張 figure
%      每個 hole 一個 subplot
%      顯示：
%        - Median
%        - 95% CI
%        - One CRF realization
%        - Validation data
%% ========================================================================

% ===== 要畫哪些參數 =====
params_to_plot = param_name;   % 會自動畫目前使用的所有參數，例如 {'fs','qt','u2'}

% ===== 顏色設定 =====
col_med  = [0.35 0.35 0.35];   % 灰色實線：median
col_ci   = [0.55 0.55 0.55];   % 灰色虛線：95% CI
col_crf  = [0.15 0.75 0.15];            % 綠色：one CRF realization
col_valf = [1 1 0];            % 黃色圓點填色
col_vale = [0.25 0.25 0.25];   % 黃色圓點邊框

% ===== 圖面配置 =====
left_margin  = 0.035;
right_margin = 0.02;
bot_margin   = 0.14;
top_margin   = 0.13;
h_gap        = 0.015;

z_plot = ExportMGPR.z(:);

for pp = 1:numel(params_to_plot)

    param_to_plot = params_to_plot{pp};

    % ===== 找參數位置 =====
    p = find(strcmp(param_name, param_to_plot), 1);

    if isempty(p)
        error('找不到參數 %s', param_to_plot);
    end

    % ===== x 軸標題 =====
    switch lower(param_to_plot)
        case 'fs'
            xlab_txt = 'f_s (kPa)';
        case 'qt'
            xlab_txt = 'q_t (kPa)';
        case 'u2'
            xlab_txt = '\Deltau (kPa)';
        otherwise
            xlab_txt = param_to_plot;
    end

    % ===== 自動抓 x 軸範圍 =====
    x_all_plot = [];

    for group_i = 1:num_groups
        h_id  = test_holes(group_i);
        h_tag = sprintf('H%d', h_id);

        x_all_plot = [x_all_plot; ...
            ExportMGPR.(h_tag).(param_to_plot).actual(:); ...
            ExportMGPR.(h_tag).(param_to_plot).median(:); ...
            ExportMGPR.(h_tag).(param_to_plot).low95(:); ...
            ExportMGPR.(h_tag).(param_to_plot).high95(:); ...
            ExportMGPR.(h_tag).(param_to_plot).realization(:)];
    end

    x_all_plot = x_all_plot(isfinite(x_all_plot));

    if isempty(x_all_plot)
        xlim_plot = [0 1];
    else
        xmin0 = min(x_all_plot);
        xmax0 = max(x_all_plot);
        dx = xmax0 - xmin0;

        if dx < 1e-12
            dx = max(abs(xmax0),1);
        end

        xlim_plot = [xmin0 - 0.08*dx, xmax0 + 0.08*dx];

        % fs / qt 不讓 x 軸從負值開始
        if strcmpi(param_to_plot,'fs') || strcmpi(param_to_plot,'qt')
            xlim_plot(1) = max(0, xlim_plot(1));
        end
    end

    % ===== 建立 figure =====
    fig_val = figure('Color','w', ...
        'Name', sprintf('Validation_panels_%s_MGPR', param_to_plot), ...
        'Position', [50 150 220*num_groups 420]);

    panel_w = (1 - left_margin - right_margin - (num_groups-1)*h_gap) / num_groups;
    panel_h = 1 - bot_margin - top_margin;

    for group_i = 1:num_groups

        h_id  = test_holes(group_i);
        h_tag = sprintf('H%d', h_id);

        xx = left_margin + (group_i-1)*(panel_w + h_gap);
        ax = axes('Position', [xx, bot_margin, panel_w, panel_h]); %#ok<LAXES>
        hold(ax,'on'); box(ax,'on'); grid(ax,'on');

        % ===== 取資料 =====
        y_act = ExportMGPR.(h_tag).(param_to_plot).actual(:);
        y_med = ExportMGPR.(h_tag).(param_to_plot).median(:);
        y_lo  = ExportMGPR.(h_tag).(param_to_plot).low95(:);
        y_hi  = ExportMGPR.(h_tag).(param_to_plot).high95(:);
        y_crf = ExportMGPR.(h_tag).(param_to_plot).realization(:);

        % ===== 95% CI =====
        h1 = plot(ax, y_lo, z_plot, '--', ...
            'Color', col_ci, ...
            'LineWidth', 1.4);

        plot(ax, y_hi, z_plot, '--', ...
            'Color', col_ci, ...
            'LineWidth', 1.4, ...
            'HandleVisibility','off');

        % ===== median =====
        h3 = plot(ax, y_med, z_plot, '-', ...
            'Color', col_med, ...
            'LineWidth', 1.8);

        % ===== one CRF realization =====
        h4 = plot(ax, y_crf, z_plot, '-', ...
            'Color', col_crf, ...
            'LineWidth', 1.2);

        % ===== validation data =====
        h5 = plot(ax, y_act, z_plot, 'o', ...
            'MarkerSize', 3.5, ...
            'MarkerFaceColor', col_valf, ...
            'MarkerEdgeColor', col_vale, ...
            'LineStyle', 'none');

        % ===== 軸設定 =====
        set(ax, ...
            'YDir','reverse', ...
            'FontSize',9, ...
            'LineWidth',1.0, ...
            'XColor','k', ...
            'YColor','k', ...
            'GridAlpha',0.25, ...
            'TickDir','in');

        ylim(ax, [min(z_plot), max(z_plot)]);
        xlim(ax, xlim_plot);

        title(ax, sprintf('Validation #%d', group_i), ...
            'FontSize', 10, ...
            'FontWeight','bold');

        xlabel(ax, xlab_txt, ...
            'FontSize', 10, ...
            'Interpreter','tex');

        if group_i == 1
            ylabel(ax, 'Depth (m)', 'FontSize', 10);

            legend(ax, [h3 h1 h4 h5], ...
                {'Median (MGPR)', '95% CI (MGPR)', 'One CRF realization', 'Validation data'}, ...
                'Location','northwest', ...
                'FontSize',8);
        else
            ax.YTickLabel = [];
        end
    end

    sgtitle(sprintf('Validation results for %s', xlab_txt), ...
        'FontSize', 12, ...
        'FontWeight','bold');

    fprintf('Validation panel figure generated for parameter: %s\n', param_to_plot);
end
%% ========================================================================
% (11) VALIDATION PANEL PLOT: DATA + MULTIPLE CRF + MEDIAN + 95% CI
%      一次畫 fs / qt / u2 三個參數
%      每個參數一張 figure
%      每個 validation hole 一個 subplot
%      顯示：
%        - Validation data
%        - N_CRF_PLOT CRF realizations
%        - Median
%        - 95% CI
%% ========================================================================

params_to_plot = param_name;

col_crf  = [0.15 0.75 0.15];   % 綠色：multiple CRF realizations
col_med  = [1.00 0.00 1.00];   % 紫色：median
col_ci   = [1.00 0.00 1.00];   % 紫色：95% CI
col_valf = [1 1 0];            % 黃色圓點填色
col_vale = [0.25 0.25 0.25];   % 黃色圓點邊框

left_margin  = 0.035;
right_margin = 0.02;
bot_margin   = 0.14;
top_margin   = 0.13;
h_gap        = 0.015;

z_plot = ExportMGPR.z(:);

for pp = 1:numel(params_to_plot)

    param_to_plot = params_to_plot{pp};

    p = find(strcmp(param_name, param_to_plot), 1);
    if isempty(p)
        error('找不到參數 %s', param_to_plot);
    end

    switch lower(param_to_plot)
        case 'fs'
            xlab_txt = 'f_s (kPa)';
        case 'qt'
            xlab_txt = 'q_t (kPa)';
        case 'u2'
            xlab_txt = '\Deltau (kPa)';
        otherwise
            xlab_txt = param_to_plot;
    end

    % ===== 自動抓 x 軸範圍：validation data + CRF + median + CI =====
    x_all_plot = [];

    for group_i = 1:num_groups
        h_id  = test_holes(group_i);
        h_tag = sprintf('H%d', h_id);

        x_all_plot = [x_all_plot; ...
            ExportMGPR.(h_tag).(param_to_plot).actual(:); ...
            ExportMGPR.(h_tag).(param_to_plot).median(:); ...
            ExportMGPR.(h_tag).(param_to_plot).low95(:); ...
            ExportMGPR.(h_tag).(param_to_plot).high95(:)]; %#ok<AGROW>

        if isfield(ExportMGPR.(h_tag).(param_to_plot), 'CRF_samples20')
            x_all_plot = [x_all_plot; ...
                ExportMGPR.(h_tag).(param_to_plot).CRF_samples20(:)]; %#ok<AGROW>
        else
            x_all_plot = [x_all_plot; ...
                ExportMGPR.(h_tag).(param_to_plot).realization(:)]; %#ok<AGROW>
        end
    end

    x_all_plot = x_all_plot(isfinite(x_all_plot));

    if isempty(x_all_plot)
        xlim_plot = [0 1];
    else
        xmin0 = min(x_all_plot);
        xmax0 = max(x_all_plot);
        dx = xmax0 - xmin0;

        if dx < 1e-12
            dx = max(abs(xmax0),1);
        end

        xlim_plot = [xmin0 - 0.08*dx, xmax0 + 0.08*dx];

        % fs / qt 不讓 x 軸從負值開始
        if strcmpi(param_to_plot,'fs') || strcmpi(param_to_plot,'qt')
            xlim_plot(1) = max(0, xlim_plot(1));
        end
    end

    fig_crf = figure('Color','w', ...
        'Name', sprintf('Validation_data_plus_%dCRF_medianCI_%s_MGPR', N_CRF_PLOT, param_to_plot), ...
        'Position', [70 120 220*num_groups 420]);

    panel_w = (1 - left_margin - right_margin - (num_groups-1)*h_gap) / num_groups;
    panel_h = 1 - bot_margin - top_margin;

    for group_i = 1:num_groups

        h_id  = test_holes(group_i);
        h_tag = sprintf('H%d', h_id);

        xx = left_margin + (group_i-1)*(panel_w + h_gap);
        ax = axes('Position', [xx, bot_margin, panel_w, panel_h]); %#ok<LAXES>
        hold(ax,'on'); box(ax,'on'); grid(ax,'on');

        y_act = ExportMGPR.(h_tag).(param_to_plot).actual(:);
        y_med = ExportMGPR.(h_tag).(param_to_plot).median(:);
        y_lo  = ExportMGPR.(h_tag).(param_to_plot).low95(:);
        y_hi  = ExportMGPR.(h_tag).(param_to_plot).high95(:);

        if isfield(ExportMGPR.(h_tag).(param_to_plot), 'CRF_samples20')
            y_crfN = ExportMGPR.(h_tag).(param_to_plot).CRF_samples20;
        else
            y_crfN = ExportMGPR.(h_tag).(param_to_plot).realization(:);
        end

        if isvector(y_crfN)
            y_crfN = y_crfN(:);
        end

        % ===== 95% CI：紫色虛線 =====
        h_ci = plot(ax, y_lo, z_plot, '--', ...
            'Color', col_ci, ...
            'LineWidth', 1.3);

        plot(ax, y_hi, z_plot, '--', ...
            'Color', col_ci, ...
            'LineWidth', 1.3, ...
            'HandleVisibility','off');

        % ===== Multiple CRF realizations：綠色 =====
        h_crf = [];
        n_crf_here = size(y_crfN, 2);

        for icrf = 1:n_crf_here
            if icrf == 1
                h_crf = plot(ax, y_crfN(:,icrf), z_plot, '-', ...
                    'Color', col_crf, ...
                    'LineWidth', 0.8);
            else
                plot(ax, y_crfN(:,icrf), z_plot, '-', ...
                    'Color', col_crf, ...
                    'LineWidth', 0.8, ...
                    'HandleVisibility','off');
            end
        end

        % ===== Median：紫色實線 =====
        h_med = plot(ax, y_med, z_plot, '-', ...
            'Color', col_med, ...
            'LineWidth', 1.9);

        % ===== validation data =====
        h_val = plot(ax, y_act, z_plot, 'o', ...
            'MarkerSize', 3.5, ...
            'MarkerFaceColor', col_valf, ...
            'MarkerEdgeColor', col_vale, ...
            'LineStyle', 'none');

        set(ax, ...
            'YDir','reverse', ...
            'FontSize',9, ...
            'LineWidth',1.0, ...
            'XColor','k', ...
            'YColor','k', ...
            'GridAlpha',0.25, ...
            'TickDir','in');

        ylim(ax, [min(z_plot), max(z_plot)]);
        xlim(ax, xlim_plot);

        title(ax, sprintf('Validation #%d', group_i), ...
            'FontSize', 10, ...
            'FontWeight','bold');

        xlabel(ax, xlab_txt, ...
            'FontSize', 10, ...
            'Interpreter','tex');

        if group_i == 1
            ylabel(ax, 'Depth (m)', 'FontSize', 10);

            legend(ax, [h_med h_ci h_crf h_val], ...
                {'Median (multi t-GPR)', ...
                 '95% CI (multi t-GPR)', ...
                 sprintf('%d CRF realizations', n_crf_here), ...
                 'Validation data'}, ...
                'Location','northwest', ...
                'FontSize',8);
        else
            ax.YTickLabel = [];
        end
    end

    % sgtitle(sprintf('Validation data, median, 95%% CI and %d CRF realizations for %s', ...
    %     N_CRF_PLOT, xlab_txt), ...
    %     'FontSize', 12, ...
    %     'FontWeight','bold');

    fprintf('Data + median + 95%% CI + %d CRF figure generated for parameter: %s\n', ...
        N_CRF_PLOT, param_to_plot);
end
%% ========================================================================
% Helper functions
%% ========================================================================

function [z, map] = zscore_forward(x)
    z = nan(size(x));

    mask = isfinite(x);
    v = x(mask);

    mu = mean(v);
    sd = max(std(v), 1e-12);

    z(mask) = (v - mu) / sd;

    map.mu = mu;
    map.sd = sd;
end


function Z = apply_zscore_transform(raw_mat, mp, do_log)
    Z = raw_mat;

    if do_log
        Z(Z <= 0) = NaN;
        Z = log(Z);
    end

    mask = isfinite(Z);

    Z(mask)  = (Z(mask) - mp.mu) / mp.sd;
    Z(~mask) = NaN;
end


function Cs = estimate_Cs_from_cov(t_mat, M)
    if isvector(t_mat)
        t_mat = reshape(t_mat, numel(t_mat)/M, M);
    end

    C = zeros(M,M);

    for i = 1:M
        xi = t_mat(:,i);

        for j = i:M
            xj = t_mat(:,j);

            mask = isfinite(xi) & isfinite(xj);

            if nnz(mask) < 30
                cij = 0;
            else
                mi = mean(xi(mask));
                mj = mean(xj(mask));

                cij = mean((xi(mask) - mi) .* (xj(mask) - mj));
            end

            C(i,j) = cij;
            C(j,i) = cij;
        end
    end

    Cs = (C + C') / 2;
end


function A = makeSPD(A, eps0)
    A = (A + A') / 2;

    [V,D] = eig(A);

    d = diag(D);
    d = max(d, eps0);

    A = V * diag(d) * V';
    A = (A + A') / 2;
end


function [rmse, r2] = compute_rmse_r2(y_true, y_pred)
    mask = isfinite(y_true) & isfinite(y_pred);

    if nnz(mask) < 5
        rmse = NaN;
        r2   = NaN;
        return;
    end

    e = y_true(mask) - y_pred(mask);

    rmse = sqrt(mean(e.^2));

    ss_res = sum(e.^2);
    ss_tot = sum((y_true(mask) - mean(y_true(mask))).^2);

    if ss_tot < 1e-12
        r2 = NaN;
    else
        r2 = 1 - ss_res / ss_tot;
    end
end
