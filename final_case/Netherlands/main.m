% Netherlands standalone MGPR main program.
% Keep this file with the data MAT files and the original MGPR helper functions.
clc; clear; close all;
timer_total = tic;

%% ===================== USER SETTINGS =====================
T_mcmc   = 10;                  % TMCMC posterior samples (increase for final runs)
Scenario = 1;

param_name_all = {'fs','qt','u2'};
log_ind_all    = [1 1 0];       % fs/log, qt/log, u2/linear
use_param      = [1 1 1 ];       % [1 1 1] = all three params

zmin_keep = 6.0;
zmax_keep = 10.0;
dz_grid   = 0.04;

DO_TMCMC_EACH_FOLD = true;      % false = quick plotting/debug mode only
T_mcmc_fast = 80;
tmcmc_beta = 0.5;

CI_LO = 2.5;
CI_HI = 97.5;

% ===== Two TMCMC correlation matrices: Z space + original space =====
PAIR_SCATTER_RNG_SEED = 27; % diagnostic only; main RNG is restored
SAVE_PAIR_SCATTER = 0;            % 1 = save both matrix figures; always display

% ===== Output switches: default is display only, no files are written =====
SAVE_FIGURES = 0;                 % 1 = save map, hyperparameter and profile figures
SAVE_RESULTS = 0;                 % 1 = save the MGPR result MAT file

% ===== Training/validation split =====
red_idx = [86 84 94 91 88 81 79];
save_dir = fullfile(pwd,'Leendert_de_Boerspolder_MGPR');
if (SAVE_FIGURES || SAVE_RESULTS || SAVE_PAIR_SCATTER) && ~exist(save_dir,'dir')
    mkdir(save_dir);
end

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

red_idx   = unique(red_idx(:)', 'stable');

assert(numel(red_idx)==7);

dist_blue = hypot(X_all - 50, Y_all - 10);
[blue_dist, blue_idx] = min(dist_blue);
fprintf('Blue outlier: hole=%d, coord=(%.3f,%.3f), dist=%.3f\n', ...
    blue_idx, X_all(blue_idx), Y_all(blue_idx), blue_dist);
assert(~ismember(blue_idx, red_idx));

usable_idx = setdiff(all_idx, blue_idx, 'stable');
extra_exclude = [12, 80, 82, 83, 85, 89, 92 17 25 40 42 75 69];   % 異常孔
usable_idx = setdiff(usable_idx, extra_exclude, 'stable');
train_idx  = setdiff(usable_idx, red_idx, 'stable');
test_holes = red_idx;



fprintf('Training holes (gray) = %d\n', numel(train_idx));
fprintf('Red validation holes  = %d\n', numel(test_holes));

%% ========================================================================
% (1) BUILD COMMON LATTICE
% col 3=z, col 5=fs, col 6=u2, col 9=qt
%% ========================================================================
z_grid = (zmin_keep:dz_grid:zmax_keep)';
nz0 = numel(z_grid);

fs_lat = nan(nz0, nh);
qt_lat = nan(nz0, nh);
u2_lat = nan(nz0, nh);

for h = 1:nh
    A = CPTR{h};
    if isempty(A) || ~isnumeric(A) || size(A,2) < 9, continue; end

    z0  = double(A(:,3));
    fs0 = double(A(:,5));
    u20 = double(A(:,6));
    qt0 = double(A(:,9));

    z0=z0(:); fs0=fs0(:); u20=u20(:); qt0=qt0(:);

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
gamma_w = 9.81/1000;
u0_grid = gamma_w * z_grid;
u2_lat  = u2_lat - u0_grid; 
% positivity for log params
if log_ind_all(1)>0.5, fs_lat(fs_lat<=0) = NaN; end
if log_ind_all(2)>0.5, qt_lat(qt_lat<=0) = NaN; end
% u2 is linear, no positivity constraint

fprintf('Initial lattice: nz=%d, z in [%.3f, %.3f]\n', nz0, min(z_grid), max(z_grid));

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
fprintf('Complete rows in %d training holes = %d / %d\n', ...
    numel(train_idx), numel(good_rows), nz0);

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
% (3) PLOT TRAIN/TEST MAP
%% ========================================================================
figure('Color','w','Name','Paper-style Train/Test map');
scatter(X_all(usable_idx), Y_all(usable_idx), 40, [0.82 0.82 0.82], 'filled'); hold on;
scatter(X_all(train_idx),  Y_all(train_idx),  55, [0.70 0.70 0.70], 'filled');
scatter(X_all(test_holes), Y_all(test_holes), 90, 'r', 'filled');
scatter(X_all(blue_idx),   Y_all(blue_idx),  100, 'c', 'filled');
for k = 1:numel(test_holes)
    h = test_holes(k);
    text(X_all(h)+0.15, Y_all(h)-0.28, sprintf('%d',h), ...
        'Color',[0.75 0 0],'FontSize',10,'FontWeight','bold');
end
text(X_all(blue_idx)+0.15, Y_all(blue_idx)+0.08, sprintf('%d',blue_idx), 'Color','k','FontSize',10,'FontWeight','bold');
grid on; axis equal;
xlabel('x (m)'); ylabel('y (m)');
title(sprintf('gray %d train | red %d validation | blue outlier', ...
    numel(train_idx), numel(test_holes)));
legend('Usable','Training','Validation','Blue outlier','Location','best');
if SAVE_FIGURES
    exportgraphics(gcf, fullfile(save_dir,'Train_Test_Map.png'),'Resolution',300);
end

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

% Plotting only: transform every loaded hole with training-derived maps.
% This does not feed validation information back into model fitting.
Zdat_all_plot = cell(1,M);
t_all_plot = nan(nz*nh,M);
for p = 1:M
    Zdat_all_plot{p} = apply_zscore_transform( ...
        raw_all{p}, Z_map{p}, log_ind(p)>0.5);
    t_all_plot(:,p) = reshape(Zdat_all_plot{p}, [], 1);
end

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
fprintf('>> Starting TMCMC on %d training holes...\n', nh_tr);
Cs = estimate_Cs_from_cov(y.t, M);
Cs = makeSPD(Cs, 1e-8);

[phi_t_Cs, D] = eig(Cs);
[~, eigsort] = sort(diag(D), 'descend');
phi_t_Cs = phi_t_Cs(:, eigsort);

inv_Cs = inv(Cs);
L_Cs   = chol(Cs, 'lower');
y.eig_thresh = 0.999;

% TMCMC bounds: [b, residual dz, residual dh, nu, a, trend dz, trend dh]
x_low = [-log(10),   log(0.01), log(0.1), log(0.1), -log(3),   log(max(temp_z(:))/10),  log(max(temp_h(:))/10)];
x_up  = [-log(0.01), log(10),  log(100),  log(3),   -log(0.1), log(max(temp_z(:))*10), log(max(temp_h(:))*10)];

ln_S = [];
timer_tmcmc = tic;
if DO_TMCMC_EACH_FOLD
    [x, ln_S, ~, ~,~] = iTMCMC_fun_mod1('GP_Matern_3D', y, x_low, x_up, T_mcmc, tmcmc_beta, Cs);
else
    Tuse = T_mcmc_fast;
    xmid = (x_low + x_up)/2;
    x = repmat(xmid, Tuse, 1) + 0.05*randn(Tuse, numel(xmid));
end
TIME_TMCMC = toc(timer_tmcmc);

Ns = size(x,1);
disp('>> TMCMC Done.');

bhp    = 1./exp(x(:,1));
sofv   = exp(x(:,2));
sofh   = exp(x(:,3));
nuv    = exp(x(:,4));
nuh    = nuv;
ahp    = 1./exp(x(:,5));
sofv_t = exp(x(:,6));
sofh_t = exp(x(:,7));

%% ========================================================================
% (6.1) TMCMC HYPERPARAMETER POSTERIOR (paper-style 1 x 4)
%% ========================================================================
fig_hyper = figure('Color','w','Name','TMCMC hyperparameters', ...
    'Position',[60 100 1500 390]);
tiledlayout(fig_hyper,1,4,'TileSpacing','compact','Padding','compact');

marker_yellow = [1.00 0.84 0.00];

% Residual correlation scales.
ax = nexttile; hold(ax,'on'); box(ax,'on'); grid(ax,'on');
scatter(ax,sofv,sofh,28,marker_yellow,'filled', ...
    'MarkerEdgeColor','k','LineWidth',0.9);
set(ax,'XScale','log','YScale','log','FontSize',12,'LineWidth',1.1, ...
    'TickDir','in','Layer','top','GridAlpha',0.25);
set_log_decade_axes(ax,sofv,sofh);
xlabel(ax,'$\delta_z^{(\epsilon)}$ (m)','Interpreter','latex');
ylabel(ax,'$\delta_h^{(\epsilon)}$ (m)','Interpreter','latex');

% Shared Matérn smoothness.
ax = nexttile; box(ax,'on'); grid(ax,'on');
n_bins = max(5,min(40,round(sqrt(Ns))));
histogram(ax,nuv,n_bins,'FaceColor','k','EdgeColor','k','FaceAlpha',0.95);
set(ax,'FontSize',12,'LineWidth',1.1,'TickDir','in','Layer','top','GridAlpha',0.25);
xlabel(ax,'$\nu^{(\epsilon)}$','Interpreter','latex');
ylabel(ax,'Frequency');

% Trend correlation scales.
ax = nexttile; hold(ax,'on'); box(ax,'on'); grid(ax,'on');
scatter(ax,sofv_t,sofh_t,28,marker_yellow,'filled', ...
    'MarkerEdgeColor','k','LineWidth',0.9);
set(ax,'XScale','log','YScale','log','FontSize',12,'LineWidth',1.1, ...
    'TickDir','in','Layer','top','GridAlpha',0.25);
set_log_decade_axes(ax,sofv_t,sofh_t);
xlabel(ax,'$\delta_z^{(t)}$ (m)','Interpreter','latex');
ylabel(ax,'$\delta_h^{(t)}$ (m)','Interpreter','latex');

% Residual and trend variance multipliers.
ax = nexttile; hold(ax,'on'); box(ax,'on'); grid(ax,'on');
scatter(ax,bhp,ahp,28,marker_yellow,'filled', ...
    'MarkerEdgeColor','k','LineWidth',0.9);
set(ax,'XScale','log','YScale','log','FontSize',12,'LineWidth',1.1, ...
    'TickDir','in','Layer','top','GridAlpha',0.25);
set_log_decade_axes(ax,bhp,ahp);
xlabel(ax,'$\beta$','Interpreter','latex');
ylabel(ax,'$\alpha$','Interpreter','latex');

if SAVE_FIGURES
    exportgraphics(fig_hyper,fullfile(save_dir,'TMCMC_hyperparameters.png'), ...
        'Resolution',300);
    savefig(fig_hyper,fullfile(save_dir,'TMCMC_hyperparameters.fig'));
end

%% ========================================================================
% (6.5) TWO TMCMC CORRELATION MATRICES
%
% For each valid TMCMC draw k:
%   Sigma_total(k) = (ahp(k)+bhp(k))*Cs
%   z_k ~ N(mu_Z,Sigma_total(k))
%
% Figure 1: Z/normal space; Figure 2: inverse-transformed original space.
% Black/gray = all-site observations; red = one joint sample per TMCMC draw.
%% ========================================================================
[AB_CS_SCATTER,AB_CS_SCATTER_FIGURES] = plot_tmcmc_scatter_both_spaces( ...
    t_tr,t_all_plot,raw_all,Z_map,param_name,log_ind,ahp,bhp,Cs, ...
    PAIR_SCATTER_RNG_SEED,SAVE_PAIR_SCATTER,save_dir); %#ok<NASGU>

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

%% ========================================================================
% (7) PREDICTION LOOP
%% ========================================================================
timer_crf = tic;
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
    t_pred_sample_raw = nan(nz_ele, M);
    t_pred_sample_Z   = nan(nz_ele, M);

    for p = 1:M
        Z_samp = squeeze(t_pred_model_samp(:,p,:));
        valid_col = find(all(isfinite(Z_samp),1), 1, 'first');
        if isempty(valid_col), valid_col = 1; end
        t_pred_sample_Z(:,p) = Z_samp(:, valid_col);

        raw_samp = Z_samp * Z_map{p}.sd + Z_map{p}.mu;
        if log_ind(p)>0.5, raw_samp = exp(raw_samp); end

        t_pred_sample_raw(:,p) = raw_samp(:, valid_col);
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

    fprintf('Finished predicting RED hole %d.\n', hold_i);
end
TIME_CRF = toc(timer_crf);

%% ========================================================================
% (8) FINAL VALIDATION PROFILES (one figure per parameter)
% Black = median of all CRF realizations, magenta = posterior 95% CI,
% green = one CRF realization, and yellow = validation observations.
%% ========================================================================
x_lims_all = {[0 20], [0 700], [-100 200]};  % fs, qt, Delta-u in kPa
col_mgpr = [1.00 0.00 1.00];
col_med  = [0.00 0.00 0.00];
col_crf  = [0.10 0.62 0.24];
col_obs  = [1.00 0.90 0.00];

fprintf('\nValidation panel order: ');
for group_i = 1:num_groups
    fprintf('#%d=H%d',group_i,test_holes(group_i));
    if group_i < num_groups, fprintf(', '); end
end
fprintf('\n');

for p = 1:M
    pname_key = param_name{p};
    x_lim_cur = x_lims_all{active_idx(p)};
    switch pname_key
        case 'fs', x_label = 'f_s (kPa)';
        case 'qt', x_label = 'q_t (kPa)';
        case 'u2', x_label = '\Deltau (kPa)';
        otherwise, x_label = pname_key;
    end

    fig_profile = figure('Color','w','Name',sprintf('Validation - %s',pname_key), ...
        'Position',[30 100 max(1500,225*num_groups) 430]);
    tl = tiledlayout(fig_profile,1,num_groups, ...
        'TileSpacing','compact','Padding','compact');

    for group_i = 1:num_groups
        ax = nexttile(tl); hold(ax,'on'); box(ax,'on'); grid(ax,'on');
        set(ax,'YDir','reverse','FontSize',10,'LineWidth',1.1, ...
            'TickDir','in','Layer','top','GridAlpha',0.28);
        xlim(ax,x_lim_cur);
        ylim(ax,[min(z_ele),max(z_ele)]);

        % Draw one stored conditional-random-field realization.
        h_crf = plot(ax,STORE_raw_sample{group_i}(:,p)*1000,z_ele,'-', ...
            'Color',col_crf,'LineWidth',1.0);

        % Posterior 95% interval from all conditional CRF realizations.
        h_ci = plot(ax,STORE_raw_lo{group_i}(:,p)*1000,z_ele,'--', ...
            'Color',col_mgpr,'LineWidth',1.5);
        plot(ax,STORE_raw_hi{group_i}(:,p)*1000,z_ele,'--', ...
            'Color',col_mgpr,'LineWidth',1.5,'HandleVisibility','off');

        % Pointwise median of all conditional CRF realizations.
        % Draw it after the realization and CI so the black line remains visible.
        h_med = plot(ax,STORE_raw_mean{group_i}(:,p)*1000,z_ele,'-', ...
            'Color',col_med,'LineWidth',2.2);

        % Validation observations are shown as yellow points, not a line.
        x_obs = STORE_raw_y_true{group_i}(:,p)*1000;
        valid_obs = isfinite(x_obs) & isfinite(z_ele);
        h_obs = plot(ax,x_obs(valid_obs),z_ele(valid_obs),'o', ...
            'LineStyle','none','MarkerSize',4.5,'MarkerFaceColor',col_obs, ...
            'MarkerEdgeColor','k','LineWidth',0.7);

        title(ax,sprintf('Validation #%d',group_i), ...
            'FontSize',11,'FontWeight','bold');
        xlabel(ax,x_label,'FontSize',10);
        if group_i == 1
            ylabel(ax,'Depth (m)','FontSize',10);
            legend(ax,[h_med,h_ci,h_crf,h_obs], ...
                {'Median of CRF realizations','95% CI (MGPR)', ...
                 '1 CRF realization','Validation data'}, ...
                'Location','best','FontSize',8,'Box','on');
        else
            ax.YTickLabel = [];
        end
    end

    if SAVE_FIGURES
        exportgraphics(fig_profile, ...
            fullfile(save_dir,sprintf('Validation_profiles_%s.png',pname_key)), ...
            'Resolution',300);
        savefig(fig_profile, ...
            fullfile(save_dir,sprintf('Validation_profiles_%s.fig',pname_key)));
    end
end

% Optional one-parameter results are loaded only for the later coverage table.
DO_COMPARE_1P = 0;
S_single = struct('fs',[],'qt',[],'u2',[]);
if DO_COMPARE_1P
    single_files = struct('fs','ver_Bounds_fs.mat', ...
        'qt','ver_Bounds_qt.mat','u2','ver_Bounds_u2.mat');
    fn_list = fieldnames(single_files);
    for fi = 1:numel(fn_list)
        fname = fn_list{fi};
        if exist(single_files.(fname),'file')
            S_single.(fname) = load(single_files.(fname));
        end
    end
end
%% ========================================================================
% (9) PRINT METRICS SUMMARY
%% ========================================================================
fprintf('\n================ Metrics Summary ================\n');
for p = 1:M
    fprintf('\nParameter: %s\n', param_name{p});
    fprintf('  Hole  | RMSE_raw |  R2_raw  | RMSE_Z  |  R2_Z\n');
    fprintf('  ------|----------|----------|---------|--------\n');
    for g = 1:num_groups
        fprintf('  %4d  | %8.4f | %8.4f | %7.4f | %7.4f\n', ...
            test_holes(g), ...
            metrics.RMSE_raw(g,p), metrics.R2_raw(g,p), ...
            metrics.RMSE_Z(g,p),   metrics.R2_Z(g,p));
    end
    fprintf('  Mean  | %8.4f | %8.4f | %7.4f | %7.4f\n', ...
        mean(metrics.RMSE_raw(:,p),'omitnan'), mean(metrics.R2_raw(:,p),'omitnan'), ...
        mean(metrics.RMSE_Z(:,p),'omitnan'),   mean(metrics.R2_Z(:,p),'omitnan'));
end

%% ========================================================================
% (10) SAVE RESULTS
%% ========================================================================
% save(fullfile(save_dir,'PaperLike_metrics_Train86_Red8.mat'), ...
%     'metrics','param_name','log_ind','T_mcmc','DO_TMCMC_EACH_FOLD', ...
%     'train_idx','test_holes','red_idx','blue_idx','usable_idx','z_grid');

% 另存多參數預測結果供後續比對
ExportMulti = struct();
ExportMulti.z = z_ele;
ExportMulti.param_name = param_name;
for group_i = 1:num_groups
    h_tag = sprintf('H%d', test_holes(group_i));
    for p = 1:M
        pn = param_name{p};
        ExportMulti.(h_tag).(pn).median = STORE_raw_mean{group_i}(:,p);
        ExportMulti.(h_tag).(pn).low95  = STORE_raw_lo{group_i}(:,p);
        ExportMulti.(h_tag).(pn).high95 = STORE_raw_hi{group_i}(:,p);
        ExportMulti.(h_tag).(pn).actual = STORE_raw_y_true{group_i}(:,p);
    end
end
% save(fullfile(save_dir,'const_Bounds_Multi3Para.mat'), '-struct', 'ExportMulti');

% fprintf('\nDONE. Results saved to: %s\n', save_dir);

%%
%% ========================================================================
% (10b) 95% CI COVERAGE RATE COMPARISON: Single vs Multi-parameter
%% ========================================================================
fprintf('\n================ 95%% CI Coverage Rate ================\n');
fprintf('%-6s | %-5s | %15s | %15s\n', 'Param','Hole','Single Coverage','Multi Coverage');
fprintf('%s\n', repmat('-',1,55));

for p = 1:M
    pname_cur = param_name{p};
    has_single = isfield(S_single, pname_cur) && ~isempty(S_single.(pname_cur));
    S_cur = [];
    if has_single, S_cur = S_single.(pname_cur); end

    n_covered_single = 0; n_total_single = 0;
    n_covered_multi  = 0; n_total_multi  = 0;

    for group_i = 1:num_groups
        h_tag   = sprintf('H%d', test_holes(group_i));
        y_true  = STORE_raw_y_true{group_i}(:, p);   % actual raw values

        % --- Multi-parameter coverage ---
        lo_m = STORE_raw_lo{group_i}(:, p);
        hi_m = STORE_raw_hi{group_i}(:, p);
        mask_m = isfinite(y_true) & isfinite(lo_m) & isfinite(hi_m);
        n_covered_multi = n_covered_multi + nnz(y_true(mask_m) >= lo_m(mask_m) & ...
                                                y_true(mask_m) <= hi_m(mask_m));
        n_total_multi   = n_total_multi   + nnz(mask_m);

        % --- Single-parameter coverage (需要插值到同一 z_grid) ---
        cov_s_str = '   N/A  ';
        if has_single && isfield(S_cur, h_tag) && isfield(S_cur, 'z')
            z_s   = S_cur.z(:);
            lo_s  = S_cur.(h_tag).low95(:);
            hi_s  = S_cur.(h_tag).high95(:);

            % 插值到 z_ele (與 y_true 同長度)
            lo_s_interp = interp1(z_s, lo_s, z_ele, 'linear', NaN);
            hi_s_interp = interp1(z_s, hi_s, z_ele, 'linear', NaN);

            mask_s = isfinite(y_true) & isfinite(lo_s_interp) & isfinite(hi_s_interp);
            n_in_s = nnz(y_true(mask_s) >= lo_s_interp(mask_s) & ...
                         y_true(mask_s) <= hi_s_interp(mask_s));
            n_tot_s = nnz(mask_s);
            n_covered_single = n_covered_single + n_in_s;
            n_total_single   = n_total_single   + n_tot_s;

            if n_tot_s > 0
                cov_s_str = sprintf('%6.2f%% (%3d/%3d)', 100*n_in_s/n_tot_s, n_in_s, n_tot_s);
            end
        end

        % per-hole multi coverage string
        if nnz(mask_m) > 0
            n_in_m   = nnz(y_true(mask_m) >= lo_m(mask_m) & y_true(mask_m) <= hi_m(mask_m));
            n_tot_m  = nnz(mask_m);
            cov_m_str = sprintf('%6.2f%% (%3d/%3d)', 100*n_in_m/n_tot_m, n_in_m, n_tot_m);
        else
            cov_m_str = '   N/A  ';
        end

        fprintf('%-6s | %5d | %15s | %15s\n', pname_cur, test_holes(group_i), cov_s_str, cov_m_str);
    end

    % --- Overall summary for this parameter ---
    if n_total_single > 0
        overall_s = sprintf('%6.2f%% (%d/%d)', 100*n_covered_single/n_total_single, n_covered_single, n_total_single);
    else
        overall_s = '   N/A  ';
    end
    if n_total_multi > 0
        overall_m = sprintf('%6.2f%% (%d/%d)', 100*n_covered_multi/n_total_multi, n_covered_multi, n_total_multi);
    else
        overall_m = '   N/A  ';
    end
    fprintf('%s\n', repmat('-',1,55));
    fprintf('%-6s | %5s | %15s | %15s  <-- Overall\n', pname_cur, 'All', overall_s, overall_m);
    fprintf('%s\n', repmat('-',1,55));
end

%% ========================================================================
% (10c) Joint Ellipse Coverage（用完整樣本直接算 Mahalanobis）
% ========================================================================
if M >= 2
    chi2_thresh = chi2inv(0.95, M);  % M維 95% → chi2(M)
    
    fprintf('\n%s\n', repmat('=',1,60));
    fprintf('  Joint %d-param 95%% Ellipse Coverage\n', M);
    fprintf('%s\n', repmat('-',1,60));
    fprintf('%-8s | %15s | %12s\n', 'Hole', 'In-ellipse', 'Coverage');
    fprintf('%s\n', repmat('-',1,60));
    
    n_in_total  = 0;
    n_tot_total = 0;
    
    % 需要重跑 prediction loop 來取得完整樣本
    % 這次不存圖，只算 coverage
    for group_i = 1:num_groups
        hold_i = test_holes(group_i);
        fprintf('Computing ellipse for hole %d...\n', hold_i);
        
        raw_ho_g = cell(1,M);
        for p = 1:M
            raw_ho_g{p} = raw_all{p}(:, hold_i);
        end
        
        X_ele = X_all(hold_i);
        Y_ele = Y_all(hold_i);
        temp_x_e = abs([X_tr;X_ele] - [X_tr;X_ele]');
        temp_y_e = abs([Y_tr;Y_ele] - [Y_tr;Y_ele]');
        y.temp_h_ele = sqrt(temp_x_e.^2 + temp_y_e.^2);
        
        % 重新收集完整樣本（Z-score 空間）
        t_samp_full = nan(nz_ele, M, Ns);
        reshape_data = reshape(y.t, nz*nh_tr, M);
        
        for k = 1:Ns
            [y.phiz, y.phih, y.phiz_ele, y.phih_ele, ~, ln_alpha] = ...
                GP_matrices_Step3(ahp(k), sofv_t(k), sofh_t(k), y, Cs);
            A_diag = diag(exp(ln_alpha));
            
            R_h = Matern_R(nuh(k), sofh(k), temp_h) + 1e-6*eye(nh_tr);
            R_z = Matern_R(nuv(k), sofv(k), temp_z) + 1e-6*eye(nz);
            inv_Rh = R_h \ eye(nh_tr);
            inv_Rz = R_z \ eye(nz);
            
            temp_term = kron(y.phih'*inv_Rh, y.phiz'*inv_Rz) * reshape_data * inv_Cs' * phi_t_Cs;
            if any(~isfinite(temp_term(:))), continue; end
            
            K = A_diag + (1/bhp(k)) * kron(phi_t_Cs'*inv_Cs*phi_t_Cs, ...
                kron(y.phih'*inv_Rh*y.phih, y.phiz'*inv_Rz*y.phiz));
            K = (K+K')/2 + 1e-10*eye(size(K));
            try, L_K = chol(K,'lower');
            catch, continue; end
            
            rhs = (1/bhp(k)) * temp_term(:);
            muW = L_K' \ (L_K \ rhs);
            w   = muW + L_K' \ (L_K \ randn(size(muW)));
            if any(~isfinite(w)), continue; end
            
            dd_train = kronmult2({phi_t_Cs, y.phih, y.phiz}, w);
            t_d = reshape_data(:) - dd_train;
            
            [E_X, L_h, L_z] = vertical_dense_stats(sofv(k), sofh(k), nuv(k), nuh(k), ...
                X_ele, Y_ele, z_ele, X_tr, Y_tr, z_grid, t_d, M);
            
            noise_pred = E_X + sqrt(bhp(k)) * kronmult2({L_Cs, L_h, L_z}, ...
                randn(numel(Y_ele)*numel(z_ele)*M, 1));
            d_pred = kronmult2({phi_t_Cs, y.phih_ele, y.phiz_ele}, w);
            t_pred = d_pred(:) + noise_pred(:);
            t_samp_full(:,:,k) = reshape(t_pred, nz_ele, M);
        end
        
        % 實測值轉 Z-score
        act_Z = nan(nz_ele, M);
        for p = 1:M
            v = raw_ho_g{p}(:);
            if log_ind(p)>0.5, v(v<=0)=NaN; v=log(v); end
            act_Z(:,p) = (v - Z_map{p}.mu) / Z_map{p}.sd;
        end
        
        % 對每個深度計算 Mahalanobis distance
        n_in  = 0;
        n_tot = 0;
        
        for zi = 1:nz_ele
            samp_zi = squeeze(t_samp_full(zi, :, :))';  % Ns × M
            valid = all(isfinite(samp_zi), 2);
            if nnz(valid) < M+1, continue; end
            
            mu_zi  = mean(samp_zi(valid,:), 1)';       % M × 1
            Cov_zi = cov(samp_zi(valid,:));             % M × M
            Cov_zi = (Cov_zi+Cov_zi')/2;
            [V_zi,D_zi] = eig(Cov_zi);
            D_zi = diag(max(diag(D_zi),1e-12));
            Cov_zi = V_zi*D_zi*V_zi';
            
            act_zi = act_Z(zi,:)';  % M × 1
            if any(~isfinite(act_zi)), continue; end
            
            try
                diff_zi = act_zi - mu_zi;
                maha2   = diff_zi' / Cov_zi * diff_zi;
                n_tot   = n_tot + 1;
                if maha2 <= chi2_thresh
                    n_in = n_in + 1;
                end
            catch
                continue
            end
        end
        
        n_in_total  = n_in_total  + n_in;
        n_tot_total = n_tot_total + n_tot;
        
        fprintf('%-8d | %6d / %6d | %10.1f%%\n', ...
            hold_i, n_in, n_tot, 100*n_in/max(n_tot,1));
    end
    
    fprintf('%s\n', repmat('-',1,60));
    fprintf('%-8s | %6d / %6d | %10.1f%%  <-- Overall\n', ...
        'All', n_in_total, n_tot_total, ...
        100*n_in_total/max(n_tot_total,1));
    fprintf('%s\n', repmat('=',1,60));
end


%% ========================================================================
% LEGACY MGPR/MSBL COMPARISON (disabled)
% The requested final output is now produced by Section (8).  Set this to
% true only when MSBL.mat is available and the old 3 x 7 comparison is needed.
%% ========================================================================
PLOT_LEGACY_MSBL_COMPARISON = false;
if PLOT_LEGACY_MSBL_COMPARISON
S_msbl = load('MSBL.mat');

test_holes_order = [86, 84, 94, 91, 88, 81, 79];
n_holes = numel(test_holes_order);

param_keys    = {'qt',          'fs',        'u2'};
param_xlims   = {[0 700],       [0 15],      [-100 200]};
param_xlabels = {'q_t (kPa)',   'f_s (kPa)', '\Deltau (kPa)'};

col_msbl = [0.45 0.45 0.45];      % MSBL 灰色
col_gpr  = [1.00 0.00 1.00];      % GPR 粉紅 / magenta
col_act  = [1.00 0.90 0.00];      % Actual data 黃色點

lw_med = 1.5;
lw_CI  = 1.2;
ms_act = 3.5;

fig = figure('Color','w', ...
    'Position',[40 40 250*n_holes 900], ...
    'Name','GPR_vs_MSBL_3x7');

left_margin  = 0.055;
right_margin = 0.030;
top_margin   = 0.055;
bot_margin   = 0.075;

h_gap = 0.018;
v_gap = 0.055;

n_row = 3;
n_col = n_holes;

tile_w = (1 - left_margin - right_margin - h_gap*(n_col-1)) / n_col;
tile_h = (1 - top_margin - bot_margin - v_gap*(n_row-1)) / n_row;

for pp = 1:n_row

    pkey   = param_keys{pp};
    xlim_p = param_xlims{pp};
    xlab_p = param_xlabels{pp};

    p_gpr = find(strcmp(param_name, pkey));

    if isempty(p_gpr)
        warning('找不到參數 %s，請確認 param_name 裡的名稱。', pkey);
        continue;
    end

    for hi = 1:n_col

        h_hole   = test_holes_order(hi);
        h_tag    = sprintf('H%d', h_hole);
        g_idx    = find(test_holes == h_hole);

        left_pos = left_margin + (hi-1)*(tile_w + h_gap);
        bot_pos  = bot_margin + (n_row-pp)*(tile_h + v_gap);

        ax = axes('Position',[left_pos bot_pos tile_w tile_h]);
        hold(ax,'on'); 
        box(ax,'on'); 
        grid(ax,'on');

        set(ax, ...
            'YDir','reverse', ...
            'FontSize',9, ...
            'LineWidth',1.0, ...
            'XColor','k', ...
            'YColor','k', ...
            'TickDir','in', ...
            'Layer','top', ...
            'GridAlpha',0.25);

        xlim(ax, xlim_p);

        if exist('z_ele','var')
            z_gpr = z_ele(:);
            ylim(ax, [min(z_gpr), max(z_gpr)]);
        elseif isfield(S_msbl,'z')
            z_gpr = S_msbl.z(:);
            ylim(ax, [min(S_msbl.z), max(S_msbl.z)]);
        else
            error('找不到 z_ele 或 S_msbl.z，無法設定深度軸。');
        end

        % ===================== MSBL：灰色 median + 灰色 CI =====================
        if isfield(S_msbl, h_tag) && isfield(S_msbl.(h_tag), pkey)

            D = S_msbl.(h_tag).(pkey);
            z_msbl = S_msbl.z(:);

            if isfield(D,'low95') && isfield(D,'high95')
                plot(ax, D.low95 * 1000, z_msbl, '--', ...
                    'Color', col_msbl, ...
                    'LineWidth', lw_CI);

                plot(ax, D.high95 * 1000, z_msbl, '--', ...
                    'Color', col_msbl, ...
                    'LineWidth', lw_CI, ...
                    'HandleVisibility','off');
            end

            if isfield(D,'median')
                plot(ax, D.median * 1000, z_msbl, '-', ...
                    'Color', col_msbl, ...
                    'LineWidth', lw_med);
            end
        end

        % ===================== GPR：粉紅 median + 粉紅 CI =====================
        if ~isempty(g_idx)

            plot(ax, STORE_raw_lo{g_idx}(:,p_gpr) * 1000, z_gpr, '--', ...
                'Color', col_gpr, ...
                'LineWidth', lw_CI);

            plot(ax, STORE_raw_hi{g_idx}(:,p_gpr) * 1000, z_gpr, '--', ...
                'Color', col_gpr, ...
                'LineWidth', lw_CI, ...
                'HandleVisibility','off');

            plot(ax, STORE_raw_mean{g_idx}(:,p_gpr) * 1000, z_gpr, '-', ...
                'Color', col_gpr, ...
                'LineWidth', lw_med);

            % ===================== Actual data：黃色點 =====================
            x_act = STORE_raw_y_true{g_idx}(:,p_gpr) * 1000;
            z_act = z_gpr;

            valid_act = isfinite(x_act) & isfinite(z_act);

            plot(ax, x_act(valid_act), z_act(valid_act), 'o', ...
                'MarkerFaceColor', col_act, ...
                'MarkerEdgeColor', 'k', ...
                'MarkerSize', ms_act, ...
                'LineWidth', 0.8, ...
                'LineStyle', 'none');
        end

        % ===================== 軸標籤 =====================
        if hi == 1
            ylabel(ax, 'Depth (m)', 'FontSize',10);
        else
            ylabel(ax, '');
        end

        if pp == n_row
            xlabel(ax, xlab_p, 'FontSize',10);
        else
            xlabel(ax, xlab_p, 'FontSize',10);
        end

        % 每一 row 左側標參數
    
        % 每一 column 上方標 hole number
        % if pp == 1
        %     title(ax, sprintf('H%d', h_hole), ...
        %         'FontSize',10, ...
        %         'FontWeight','normal');
        % end
        % ===================== 右下角 (a)(b)(c)... 標記 =====================
        text(ax, 0.96, 0.04, sprintf('(%s)', char('a' + hi - 1)), ...
            'Units','normalized', ...
            'FontSize',10, ...
            'FontWeight','bold', ...
            'HorizontalAlignment','right', ...
            'VerticalAlignment','bottom', ...
            'Color','k');
    end
end

% ===================== Legend =====================
legend_ax = axes('Position',[0.80 0.89 0.17 0.075]);
hold(legend_ax,'on'); 
axis(legend_ax,'off');

h1 = plot(legend_ax, nan, nan, '-', ...
    'Color', col_msbl, ...
    'LineWidth', lw_med);

h2 = plot(legend_ax, nan, nan, '--', ...
    'Color', col_msbl, ...
    'LineWidth', lw_CI);

h3 = plot(legend_ax, nan, nan, '-', ...
    'Color', col_gpr, ...
    'LineWidth', lw_med);

h4 = plot(legend_ax, nan, nan, '--', ...
    'Color', col_gpr, ...
    'LineWidth', lw_CI);

h5 = plot(legend_ax, nan, nan, 'o', ...
    'MarkerFaceColor', col_act, ...
    'MarkerEdgeColor', 'k', ...
    'MarkerSize', ms_act, ...
    'LineWidth', 0.8, ...
    'LineStyle', 'none');

lgd = legend(legend_ax, [h1 h2 h3 h4 h5], ...
    {'Median (MSBL)', ...
     '95% CI (MSBL)', ...
     'Median (MGPR)', ...
     '95% CI (MGPR)', ...
     'Actual data'}, ...
    'Location','northwest', ...
    'FontSize',8, ...
    'Box','on');

lgd.ItemTokenSize = [16 8];

if ~exist('save_dir','var') || isempty(save_dir)
    save_dir = pwd;
end

%saveas(fig, fullfile(save_dir, 'Compare_GPR_MSBL_3rows_7holes_yellow_points.png'));
%savefig(fig, fullfile(save_dir, 'Compare_GPR_MSBL_3rows_7holes_yellow_points.fig'));

fprintf('[完成] 已輸出 Compare_GPR_MSBL_3rows_7holes_yellow_points.png / .fig\n');
end
%% ========================================================================
%% 儲存 MGPR 結果，之後可直接畫圖比較
MGPR = struct();

MGPR.model_name = 'MGPR';
MGPR.z = z_ele(:);
MGPR.test_holes = test_holes(:);
MGPR.param_name = param_name;
MGPR.log_ind = log_ind;
MGPR.active_idx = active_idx;

% 若有座標資料，也一起存
MGPR.X_test = X_all(test_holes);
MGPR.Y_test = Y_all(test_holes);

% 儲存超參數與模型資訊，方便之後追蹤
MGPR.Cs = Cs;
MGPR.ln_S = ln_S;
MGPR.metrics = metrics;
MGPR.ab_Cs_scatter = AB_CS_SCATTER;

MGPR.hyperparameters.bhp    = bhp;
MGPR.hyperparameters.sofv   = sofv;
MGPR.hyperparameters.sofh   = sofh;
MGPR.hyperparameters.nuv    = nuv;
MGPR.hyperparameters.nuh    = nuh;
MGPR.hyperparameters.ahp    = ahp;
MGPR.hyperparameters.sofv_t = sofv_t;
MGPR.hyperparameters.sofh_t = sofh_t;

% 儲存每一個 validation hole 的結果
for group_i = 1:numel(test_holes)

    h_id  = test_holes(group_i);
    h_tag = sprintf('H%d', h_id);

    MGPR.(h_tag).hole_id = h_id;
    MGPR.(h_tag).x = X_all(h_id);
    MGPR.(h_tag).y = Y_all(h_id);

    for p = 1:numel(param_name)

        pn = param_name{p};

        MGPR.(h_tag).(pn).median = STORE_raw_mean{group_i}(:,p);
        MGPR.(h_tag).(pn).low95  = STORE_raw_lo{group_i}(:,p);
        MGPR.(h_tag).(pn).high95 = STORE_raw_hi{group_i}(:,p);
        MGPR.(h_tag).(pn).actual = STORE_raw_y_true{group_i}(:,p);

        if exist('STORE_raw_sample','var') && numel(STORE_raw_sample) >= group_i ...
                && ~isempty(STORE_raw_sample{group_i})
            MGPR.(h_tag).(pn).sample = STORE_raw_sample{group_i}(:,p);
        end

        % Z-space 結果也一起存，之後若要算 coverage 或 ellipse 會用到
        if exist('STORE_Z_mean','var') && numel(STORE_Z_mean) >= group_i
            MGPR.(h_tag).(pn).Z_median = STORE_Z_mean{group_i}(:,p);
            MGPR.(h_tag).(pn).Z_low95  = STORE_Z_lo{group_i}(:,p);
            MGPR.(h_tag).(pn).Z_high95 = STORE_Z_hi{group_i}(:,p);
            MGPR.(h_tag).(pn).Z_actual = STORE_Z_y_true{group_i}(:,p);
        end
    end
end

% Timing is measured independently for TMCMC and the main CRF prediction.
TIME_TOTAL = toc(timer_total);
MGPR.timing.TMCMC_seconds = TIME_TMCMC;
MGPR.timing.CRF_seconds   = TIME_CRF;
MGPR.timing.total_seconds = TIME_TOTAL;

fprintf('\n================ Timing Summary ================\n');
fprintf('TMCMC : %.3f s\n',TIME_TMCMC);
fprintf('CRF   : %.3f s\n',TIME_CRF);
fprintf('Total : %.3f s\n',TIME_TOTAL);

if SAVE_RESULTS
    result_file = fullfile(save_dir,'MGPR_results_for_plot.mat');
    save(result_file,'MGPR','-v7.3');
    fprintf('[Saved] %s\n',result_file);
else
    fprintf('[Not saved] SAVE_RESULTS = 0 and SAVE_FIGURES = %d.\n',SAVE_FIGURES);
end

% Helper functions
%% ========================================================================

function [OUT,FIGS] = plot_tmcmc_scatter_both_spaces( ...
    t_tr,t_all_plot,raw_all,Z_map,param_name,log_ind,ahp,bhp,Cs, ...
    diagnostic_seed,save_figures,save_dir)
% Produce exactly two M-by-M figures: Z space and original physical space.
% Each valid TMCMC draw supplies one joint point with covariance
% Sigma_k=(a_k+b_k)Cs. Transform statistics still come from training data.

M = numel(param_name);
scale_all = ahp(:)+bhp(:);
valid_draw = isfinite(ahp(:)) & isfinite(bhp(:)) & ...
             isfinite(scale_all) & scale_all>0;
draw_id = find(valid_draw);
scale_use = scale_all(valid_draw);

if isempty(draw_id)
    error('TMCMC matrix failed: no finite positive a+b samples.');
end

% This diagnostic must not change random numbers used later by CRF.
rng_before_diagnostic = rng;
rng_cleanup = onCleanup(@() rng(rng_before_diagnostic)); %#ok<NASGU>
rng(diagnostic_seed,'twister');

mu_Z = mean(t_tr,1,'omitnan');
Cs_spd = makeSPD(Cs,1e-10);
L_Cs_plot = chol(Cs_spd,'lower');

E = randn(M,numel(draw_id));
E = E.*repmat(sqrt(scale_use(:)'),M,1);
Z_generated = (repmat(mu_Z(:),1,numel(draw_id))+L_Cs_plot*E)';

% Inverse transform the same joint samples used in the Z-space figure.
raw_generated = nan(numel(draw_id),M);
raw_observed = nan(size(t_all_plot,1),M);
for p = 1:M
    raw_generated(:,p) = Z_generated(:,p)*Z_map{p}.sd+Z_map{p}.mu;
    if log_ind(p)>0.5
        raw_generated(:,p) = exp(raw_generated(:,p));
    end
    raw_observed(:,p) = raw_all{p}(:);
end

% Paper order: qt, fs, u2; append any other active parameters afterwards.
preferred = {'qt','fs','u2'};
plot_order = [];
for k = 1:numel(preferred)
    id = find(strcmpi(param_name,preferred{k}),1);
    if ~isempty(id), plot_order(end+1) = id; end %#ok<AGROW>
end
plot_order = [plot_order,setdiff(1:M,plot_order,'stable')];

names_plot = param_name(plot_order);
logs_plot = log_ind(plot_order);
Z_data_plot = t_all_plot(:,plot_order);
Z_tmcmc_plot = Z_generated(:,plot_order);
raw_data_plot = raw_observed(:,plot_order);
raw_tmcmc_plot = raw_generated(:,plot_order);

% Store full correlation matrices and the generated samples for later use.
OUT = struct();
OUT.definition = 'z_k ~ N(mu_Z,(a_k+b_k)Cs), followed by inverse transform';
OUT.plot_spaces = {'model-normal Z space','inverse-transformed original space'};
OUT.parameter_order = names_plot;
OUT.rng_seed = diagnostic_seed;
OUT.draw_id = draw_id;
OUT.a_samples = ahp(valid_draw);
OUT.b_samples = bhp(valid_draw);
OUT.a_plus_b_samples = scale_use;
OUT.mean_Z = mu_Z;
OUT.Cs = Cs_spd;
OUT.generated_Z_samples = Z_generated;
OUT.generated_original_samples = raw_generated;
OUT.observed_Z_all_site = t_all_plot;
OUT.observed_original_all_site = raw_observed;
OUT.corr_data_Z = pairwise_corr_matrix_local(Z_data_plot);
OUT.corr_TMCMC_Z = pairwise_corr_matrix_local(Z_tmcmc_plot);
OUT.corr_data_original = pairwise_corr_matrix_local(raw_data_plot);
OUT.corr_TMCMC_original = pairwise_corr_matrix_local(raw_tmcmc_plot);

pair_template = struct('space','','x_parameter','','y_parameter','', ...
    'N_data',NaN,'N_TMCMC',NaN,'data_correlation',NaN, ...
    'TMCMC_sample_correlation',NaN);
OUT.pair_Z = repmat(pair_template,0,1);
OUT.pair_original = repmat(pair_template,0,1);

fprintf('\n%s\n',repmat('=',1,82));
fprintf('  TMCMC joint samples (red) vs ALL site data (gray)\n');
fprintf('  Generated joint samples: %d | Sigma_k=(a_k+b_k)Cs\n',numel(draw_id));
fprintf('%s\n',repmat('=',1,82));
fprintf('%-10s | %-9s | %12s | %12s | %10s | %10s\n', ...
    'Space','Pair','Corr(data)','Corr(TMCMC)','N data','N TMCMC');
fprintf('%s\n',repmat('-',1,82));

for i = 1:M-1
    for j = i+1:M
        pair_name = [names_plot{i} '-' names_plot{j}];

        Pz = make_pair_summary_local('Z',names_plot{i},names_plot{j}, ...
            Z_data_plot(:,[i j]),Z_tmcmc_plot(:,[i j]),pair_template);
        OUT.pair_Z(end+1,1) = Pz; %#ok<AGROW>
        fprintf('%-10s | %-9s | %12.4f | %12.4f | %10d | %10d\n', ...
            'Z',pair_name,Pz.data_correlation,Pz.TMCMC_sample_correlation, ...
            Pz.N_data,Pz.N_TMCMC);

        Pr = make_pair_summary_local('original',names_plot{i},names_plot{j}, ...
            raw_data_plot(:,[i j]),raw_tmcmc_plot(:,[i j]),pair_template);
        OUT.pair_original(end+1,1) = Pr; %#ok<AGROW>
        fprintf('%-10s | %-9s | %12.4f | %12.4f | %10d | %10d\n', ...
            'original',pair_name,Pr.data_correlation,Pr.TMCMC_sample_correlation, ...
            Pr.N_data,Pr.N_TMCMC);
    end
end
fprintf('%s\n',repmat('=',1,82));

FIGS = struct();
FIGS.Z = draw_tmcmc_matrix_local(Z_data_plot,Z_tmcmc_plot, ...
    names_plot,logs_plot,'Z');
FIGS.original = draw_tmcmc_matrix_local(raw_data_plot,raw_tmcmc_plot, ...
    names_plot,logs_plot,'original');

if save_figures
    exportgraphics(FIGS.Z,fullfile(save_dir,'TMCMC_matrix_Z_space.png'), ...
        'Resolution',300);
    savefig(FIGS.Z,fullfile(save_dir,'TMCMC_matrix_Z_space.fig'));
    exportgraphics(FIGS.original, ...
        fullfile(save_dir,'TMCMC_matrix_original_space.png'),'Resolution',300);
    savefig(FIGS.original, ...
        fullfile(save_dir,'TMCMC_matrix_original_space.fig'));
end
end


function fig = draw_tmcmc_matrix_local(data_mat,tmcmc_mat,names,log_flags,space_name)
% Draw a symmetric correlation matrix. Diagonal cells are identity lines.
n_param = numel(names);
fig = figure('Color','w','Units','pixels','Position',[80 50 900 850], ...
    'Name',sprintf('TMCMC correlation matrix - %s space',space_name));
tl = tiledlayout(fig,n_param,n_param,'TileSpacing','compact','Padding','compact');

% Use one consistent range for every occurrence of the same parameter.
axis_lims = nan(n_param,2);
for j = 1:n_param
    v = [data_mat(:,j);tmcmc_mat(:,j)];
    v = v(isfinite(v));
    if strcmpi(space_name,'Z')
        radius = max(3,ceil(max(abs(v))));
        if isempty(radius) || ~isfinite(radius), radius = 3; end
        axis_lims(j,:) = [-radius,radius];
    else
        axis_lims(j,:) = padded_limits_local(v);
    end
end

for row = 1:n_param
    for col = 1:n_param
        ax = nexttile(tl,(row-1)*n_param+col);
        hold(ax,'on'); box(ax,'on'); grid(ax,'on');
        set(ax,'FontSize',9,'LineWidth',0.9,'TickDir','in', ...
            'Layer','top','GridAlpha',0.22);
        xlim(ax,axis_lims(col,:));
        ylim(ax,axis_lims(row,:));

        if row == col
            lim_same = axis_lims(row,:);
            plot(ax,lim_same,lim_same,'k-','LineWidth',1.4);
            text(ax,0.05,0.93,'\rho = 1.000','Units','normalized', ...
                'Color',[0.90 0.05 0.05],'FontWeight','bold','FontSize',9, ...
                'VerticalAlignment','top');
        else
            XY_data = data_mat(:,[col row]);
            XY_tmcmc = tmcmc_mat(:,[col row]);
            XY_data = XY_data(all(isfinite(XY_data),2),:);
            XY_tmcmc = XY_tmcmc(all(isfinite(XY_tmcmc),2),:);

            h_data = scatter(ax,XY_data(:,1),XY_data(:,2),8, ...
                [0.20 0.20 0.20],'filled','MarkerFaceAlpha',0.13, ...
                'MarkerEdgeColor','none');
            h_tmcmc = scatter(ax,XY_tmcmc(:,1),XY_tmcmc(:,2),24, ...
                [0.92 0.05 0.05],'filled','MarkerFaceAlpha',0.78, ...
                'MarkerEdgeColor',[0.55 0 0],'LineWidth',0.5);

            rho_data = pair_corr_local(XY_data);
            rho_tmcmc = pair_corr_local(XY_tmcmc);
            text(ax,0.05,0.93,sprintf('\\rho_D = %.3f',rho_data), ...
                'Units','normalized','Color',[0.15 0.15 0.15], ...
                'FontWeight','bold','FontSize',8,'VerticalAlignment','top');
            text(ax,0.05,0.83,sprintf('\\rho_T = %.3f',rho_tmcmc), ...
                'Units','normalized','Color',[0.90 0.05 0.05], ...
                'FontWeight','bold','FontSize',8,'VerticalAlignment','top');

            if row == 1 && col == n_param
                legend(ax,[h_data,h_tmcmc], ...
                    {'All-site data','TMCMC joint samples'}, ...
                    'Location','southeast','FontSize',7,'Box','on');
            end
        end

        xlabel(ax,matrix_parameter_label_local(names{col},log_flags(col),space_name), ...
            'Interpreter','tex','FontSize',9);
        ylabel(ax,matrix_parameter_label_local(names{row},log_flags(row),space_name), ...
            'Interpreter','tex','FontSize',9);
        axis(ax,'square');
    end
end

if strcmpi(space_name,'Z')
    title(tl,'TMCMC samples vs. transformed data (Z space)', ...
        'FontWeight','bold','FontSize',14);
else
    title(tl,'TMCMC samples vs. data (original space)', ...
        'FontWeight','bold','FontSize',14);
end
end


function P = make_pair_summary_local(space,x_name,y_name,XY_data,XY_tmcmc,P)
XY_data = XY_data(all(isfinite(XY_data),2),:);
XY_tmcmc = XY_tmcmc(all(isfinite(XY_tmcmc),2),:);
P.space = space;
P.x_parameter = x_name;
P.y_parameter = y_name;
P.N_data = size(XY_data,1);
P.N_TMCMC = size(XY_tmcmc,1);
P.data_correlation = pair_corr_local(XY_data);
P.TMCMC_sample_correlation = pair_corr_local(XY_tmcmc);
end


function R = pairwise_corr_matrix_local(X)
n = size(X,2);
R = eye(n);
for i = 1:n
    for j = i+1:n
        XY = X(:,[i j]);
        XY = XY(all(isfinite(XY),2),:);
        R(i,j) = pair_corr_local(XY);
        R(j,i) = R(i,j);
    end
end
end


function lim = padded_limits_local(v)
v = v(isfinite(v));
if isempty(v)
    lim = [-1 1];
    return;
end
lo = min(v); hi = max(v);
span = hi-lo;
if span <= max(1,abs(lo))*1e-10
    span = max(1,abs(lo))*0.2;
end
lim = [lo-0.05*span,hi+0.05*span];
end


function txt = matrix_parameter_label_local(name,do_log,space_name)
if strcmpi(space_name,'Z')
    sym = parameter_symbol_local(name);
    if do_log
        txt = ['normalized ln(' sym ')'];
    else
        txt = ['normalized ' sym];
    end
else
    txt = parameter_raw_label_local(name);
end
end


function [OUT,FIGS] = plot_tmcmc_scatter_both_spaces_legacy( ...
    t_tr,t_all_plot,raw_all,Z_map,param_name,log_ind,ahp,bhp,Cs, ...
    diagnostic_seed,save_figures,save_dir)
% Generate one joint M-parameter point from every TMCMC (a,b) draw using
% Sigma_total(k)=(a_k+b_k)*Cs.  Samples are generated jointly in Z space,
% back-transformed, and compared with observations from all site holes.
% Transform maps and mu_Z are still estimated from training holes only.

pair_def = {'qt','fs'; 'fs','u2'; 'qt','u2'};
M = numel(param_name);

scale_all = ahp(:)+bhp(:);
valid_draw = isfinite(ahp(:)) & isfinite(bhp(:)) & ...
             isfinite(scale_all) & scale_all>0;
draw_id = find(valid_draw);
scale_use = scale_all(valid_draw);
a_use = ahp(valid_draw);
b_use = bhp(valid_draw);

if isempty(draw_id)
    error('Pair scatter failed: no finite positive TMCMC a+b samples.');
end

% Preserve the main program RNG.  Adding this diagnostic therefore does not
% change any random numbers subsequently used by the CRF prediction loop.
rng_before_diagnostic = rng;
rng_cleanup = onCleanup(@() rng(rng_before_diagnostic)); %#ok<NASGU>
rng(diagnostic_seed,'twister');

mu_Z = mean(t_tr,1,'omitnan');
Cs_spd = makeSPD(Cs,1e-10);
L_Cs_plot = chol(Cs_spd,'lower');

E = randn(M,numel(draw_id));
E = E.*repmat(sqrt(scale_use(:)'),M,1);
Z_generated = (repmat(mu_Z(:),1,numel(draw_id))+L_Cs_plot*E)';

% Back-transform generated samples and collect all-site raw observations.
raw_generated = nan(numel(draw_id),M);
raw_observed = nan(size(t_all_plot,1),M);
for p = 1:M
    raw_generated(:,p) = Z_generated(:,p)*Z_map{p}.sd+Z_map{p}.mu;
    if log_ind(p)>0.5
        raw_generated(:,p) = exp(raw_generated(:,p));
    end
    raw_observed(:,p) = raw_all{p}(:);
end

pair_template = struct('space','','x_parameter','','y_parameter','', ...
    'N_data',NaN,'N_TMCMC',NaN,'data_correlation',NaN, ...
    'TMCMC_sample_correlation',NaN);

OUT = struct();
OUT.definition = 'z_k ~ N(mu_Z,(a_k+b_k)Cs), followed by inverse transform';
OUT.plot_spaces = {'model-normal Z space','inverse-transformed original space'};
OUT.rng_seed = diagnostic_seed;
OUT.draw_id = draw_id;
OUT.a_samples = a_use;
OUT.b_samples = b_use;
OUT.a_plus_b_samples = scale_use;
OUT.mean_Z = mu_Z;
OUT.Cs = Cs_spd;
OUT.generated_Z_samples = Z_generated;
OUT.generated_original_samples = raw_generated;
OUT.observed_Z_all_site = t_all_plot;
OUT.observed_original_all_site = raw_observed;
OUT.pair_Z = repmat(pair_template,0,1);
OUT.pair_original = repmat(pair_template,0,1);
FIGS = struct();
FIGS.Z = cell(size(pair_def,1),1);
FIGS.original = cell(size(pair_def,1),1);

fprintf('\n%s\n',repmat('=',1,82));
fprintf('  TMCMC joint samples (red) vs ALL site data (black): Z and original spaces\n');
fprintf('  Generated joint samples: %d | Sigma_k=(a_k+b_k)Cs\n',numel(draw_id));
fprintf('%s\n',repmat('=',1,82));
fprintf('%-10s | %-9s | %12s | %12s | %10s | %10s\n', ...
    'Space','Pair','Corr(data)','Corr(TMCMC)','N data','N TMCMC');
fprintf('%s\n',repmat('-',1,82));

for q = 1:size(pair_def,1)
    x_name = pair_def{q,1};
    y_name = pair_def{q,2};
    ix = find(strcmpi(param_name,x_name),1);
    iy = find(strcmpi(param_name,y_name),1);

    if isempty(ix) || isempty(iy)
        warning('Skipping %s-%s: both parameters are not active.',x_name,y_name);
        continue;
    end

    % ================================================================
    % Figures 1-3: model-normal/Z space
    %   black = transformed observations from all site holes
    %   red   = generated samples Z_generated
    % ================================================================
    XY_data_Z = t_all_plot(:,[ix iy]);
    XY_tmcmc_Z = Z_generated(:,[ix iy]);
    keep_data_Z = all(isfinite(XY_data_Z),2);
    keep_tmcmc_Z = all(isfinite(XY_tmcmc_Z),2);
    XY_data_Z = XY_data_Z(keep_data_Z,:);
    XY_tmcmc_Z = XY_tmcmc_Z(keep_tmcmc_Z,:);

    corr_data_Z = pair_corr_local(XY_data_Z);
    corr_tmcmc_Z = pair_corr_local(XY_tmcmc_Z);

    fig_Z = figure('Color','w','Units','pixels','Position',[70 70 720 620], ...
        'Name',sprintf('Z space: %s-%s',x_name,y_name));
    ax_Z = axes(fig_Z); hold(ax_Z,'on'); box(ax_Z,'on'); grid(ax_Z,'on');

    h_data_Z = scatter(ax_Z,XY_data_Z(:,1),XY_data_Z(:,2), ...
        10,'k','filled','MarkerFaceAlpha',0.16,'MarkerEdgeAlpha',0.16);
    h_tmcmc_Z = scatter(ax_Z,XY_tmcmc_Z(:,1),XY_tmcmc_Z(:,2), ...
        28,[0.90 0.05 0.05],'filled','MarkerFaceAlpha',0.70, ...
        'MarkerEdgeColor',[0.55 0 0],'MarkerEdgeAlpha',0.75);

    xlabel(ax_Z,parameter_Z_label_local(x_name,log_ind(ix)),'Interpreter','tex');
    ylabel(ax_Z,parameter_Z_label_local(y_name,log_ind(iy)),'Interpreter','tex');
    title(ax_Z,sprintf('[Z space] %s-%s: TMCMC samples vs. transformed data', ...
        parameter_symbol_local(x_name),parameter_symbol_local(y_name)), ...
        'Interpreter','tex','FontWeight','bold');

    text(ax_Z,0.03,0.97,sprintf([ ...
        'Corr(data) = %.3f\nCorr(TMCMC) = %.3f\nN_{TMCMC} = %d'], ...
        corr_data_Z,corr_tmcmc_Z,size(XY_tmcmc_Z,1)), ...
        'Units','normalized','VerticalAlignment','top', ...
        'BackgroundColor','w','EdgeColor',[0.65 0.65 0.65], ...
        'Margin',6,'Interpreter','tex','FontSize',10);

    legend(ax_Z,[h_data_Z h_tmcmc_Z], ...
        {'Transformed all-site data','TMCMC-generated joint samples'}, ...
        'Location','best','FontSize',10);
    set(ax_Z,'FontSize',11,'LineWidth',1.0,'GridAlpha',0.22);

    P_Z = pair_template;
    P_Z.space = 'Z';
    P_Z.x_parameter = x_name;
    P_Z.y_parameter = y_name;
    P_Z.N_data = size(XY_data_Z,1);
    P_Z.N_TMCMC = size(XY_tmcmc_Z,1);
    P_Z.data_correlation = corr_data_Z;
    P_Z.TMCMC_sample_correlation = corr_tmcmc_Z;
    OUT.pair_Z(end+1,1) = P_Z; %#ok<AGROW>
    FIGS.Z{q} = fig_Z;

    fprintf('%-10s | %-9s | %12.4f | %12.4f | %10d | %10d\n', ...
        'Z',[x_name '-' y_name],corr_data_Z,corr_tmcmc_Z, ...
        size(XY_data_Z,1),size(XY_tmcmc_Z,1));

    if save_figures
        tag_Z = sprintf('TMCMC_vs_data_Z_%s_%s',x_name,y_name);
        exportgraphics(fig_Z,fullfile(save_dir,[tag_Z '.png']),'Resolution',300);
        savefig(fig_Z,fullfile(save_dir,[tag_Z '.fig']));
    end
end

% Create the three original-space figures only after all three Z-space
% figures have been created, so MATLAB figure order is Z(3) then raw(3).
for q = 1:size(pair_def,1)
    x_name = pair_def{q,1};
    y_name = pair_def{q,2};
    ix = find(strcmpi(param_name,x_name),1);
    iy = find(strcmpi(param_name,y_name),1);

    if isempty(ix) || isempty(iy)
        continue;
    end

    % ================================================================
    % Figures 4-6: inverse-transformed/original parameter space
    %   black = all-site raw data, red = inverse_transform(Z_generated)
    % ================================================================
    XY_original = raw_observed(:,[ix iy]);
    XY_tmcmc = raw_generated(:,[ix iy]);
    keep_original = all(isfinite(XY_original),2);
    keep_tmcmc = all(isfinite(XY_tmcmc),2);
    XY_original = XY_original(keep_original,:);
    XY_tmcmc = XY_tmcmc(keep_tmcmc,:);

    corr_original = pair_corr_local(XY_original);
    corr_tmcmc = pair_corr_local(XY_tmcmc);

    fig = figure('Color','w','Units','pixels','Position',[90 70 720 620], ...
        'Name',sprintf('TMCMC vs original: %s-%s',x_name,y_name));
    ax = axes(fig); hold(ax,'on'); box(ax,'on'); grid(ax,'on');

    h_original = scatter(ax,XY_original(:,1),XY_original(:,2), ...
        10,'k','filled','MarkerFaceAlpha',0.16,'MarkerEdgeAlpha',0.16);
    h_tmcmc = scatter(ax,XY_tmcmc(:,1),XY_tmcmc(:,2), ...
        28,[0.90 0.05 0.05],'filled','MarkerFaceAlpha',0.70, ...
        'MarkerEdgeColor',[0.55 0 0],'MarkerEdgeAlpha',0.75);

    xlabel(ax,parameter_raw_label_local(x_name),'Interpreter','tex');
    ylabel(ax,parameter_raw_label_local(y_name),'Interpreter','tex');
    title(ax,sprintf('[Original space] %s-%s: TMCMC samples vs. original data', ...
        parameter_symbol_local(x_name),parameter_symbol_local(y_name)), ...
        'Interpreter','tex','FontWeight','bold');

    text(ax,0.03,0.97,sprintf([ ...
        'Corr(original) = %.3f\nCorr(TMCMC) = %.3f\n' ...
        'N_{TMCMC} = %d'],corr_original,corr_tmcmc,size(XY_tmcmc,1)), ...
        'Units','normalized','VerticalAlignment','top', ...
        'BackgroundColor','w','EdgeColor',[0.65 0.65 0.65], ...
        'Margin',6,'Interpreter','tex','FontSize',10);

    legend(ax,[h_original h_tmcmc], ...
        {'Original all-site data','TMCMC-generated joint samples'}, ...
        'Location','best','FontSize',10);
    set(ax,'FontSize',11,'LineWidth',1.0,'GridAlpha',0.22);

    P = pair_template;
    P.space = 'original';
    P.x_parameter = x_name;
    P.y_parameter = y_name;
    P.N_data = size(XY_original,1);
    P.N_TMCMC = size(XY_tmcmc,1);
    P.data_correlation = corr_original;
    P.TMCMC_sample_correlation = corr_tmcmc;
    OUT.pair_original(end+1,1) = P; %#ok<AGROW>
    FIGS.original{q} = fig;

    fprintf('%-10s | %-9s | %12.4f | %12.4f | %10d | %10d\n', ...
        'original',[x_name '-' y_name],corr_original,corr_tmcmc, ...
        size(XY_original,1),size(XY_tmcmc,1));

    if save_figures
        tag = sprintf('TMCMC_vs_data_original_%s_%s',x_name,y_name);
        exportgraphics(fig,fullfile(save_dir,[tag '.png']),'Resolution',300);
        savefig(fig,fullfile(save_dir,[tag '.fig']));
    end
end

fprintf('%s\n',repmat('=',1,82));
end


function r = pair_corr_local(XY)
if size(XY,1)<3 || std(XY(:,1))<=1e-12 || std(XY(:,2))<=1e-12
    r = NaN;
else
    R = corrcoef(XY);
    r = R(1,2);
end
end


function txt = parameter_Z_label_local(name,do_log)
sym = parameter_symbol_local(name);
if do_log
    txt = ['Standardized ln(' sym ')'];
else
    txt = ['Standardized ' sym];
end
end


function txt = parameter_raw_label_local(name)
switch lower(char(name))
    case 'fs'
        txt = 'f_s (MPa)';
    case {'qt','qc'}
        txt = [parameter_symbol_local(name) ' (MPa)'];
    case {'u2','du'}
        txt = '\Delta u (MPa)';
    otherwise
        txt = [char(name) ' (original scale)'];
end
end


function sym = parameter_symbol_local(name)
switch lower(char(name))
    case 'fs'
        sym = 'f_s';
    case {'qt','qc'}
        name_char = lower(char(name));
        sym = ['q_' name_char(2)];
    case {'u2','du'}
        sym = '\Delta u';
    otherwise
        sym = char(name);
end
end

function [z, map] = zscore_forward(x)
    z = nan(size(x));
    mask = isfinite(x);
    v = x(mask);
    mu = mean(v);
    sd = max(std(v), 1e-12);
    z(mask) = (v-mu)/sd;
    map.mu = mu;
    map.sd = sd;
end

function Z = apply_zscore_transform(raw_mat, mp, do_log)
    Z = raw_mat;
    if do_log
        Z(Z<=0) = NaN;
        Z = log(Z);
    end
    mask = isfinite(Z);
    Z(mask) = (Z(mask) - mp.mu) / mp.sd;
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
            if nnz(mask)<30, cij=0;
            else
                mi=mean(xi(mask)); mj=mean(xj(mask));
                cij=mean((xi(mask)-mi).*(xj(mask)-mj));
            end
            C(i,j)=cij; C(j,i)=cij;
        end
    end
    Cs=(C+C')/2;
end

function A = makeSPD(A, eps0)
    A=(A+A')/2;
    [V,D]=eig(A);
    d=diag(D); d=max(d,eps0);
    A=V*diag(d)*V';
    A=(A+A')/2;
end

function set_log_decade_axes(ax,x,y)
% Expand narrow posterior clouds to complete powers-of-ten intervals.
% Without this, MATLAB may label a log axis as 2.68, 2.70, 2.72, which
% visually resembles a linear axis even though XScale/YScale are logarithmic.
    x = x(isfinite(x) & x>0);
    y = y(isfinite(y) & y>0);
    if isempty(x) || isempty(y), return; end

    x_exp = [floor(log10(min(x))), ceil(log10(max(x)))];
    y_exp = [floor(log10(min(y))), ceil(log10(max(y)))];
    if x_exp(1) == x_exp(2), x_exp = x_exp + [-1 1]; end
    if y_exp(1) == y_exp(2), y_exp = y_exp + [-1 1]; end

    xlim(ax,10.^x_exp);
    ylim(ax,10.^y_exp);
    xticks(ax,10.^(x_exp(1):x_exp(2)));
    yticks(ax,10.^(y_exp(1):y_exp(2)));
    ax.XMinorGrid = 'on';
    ax.YMinorGrid = 'on';
end

function [rmse, r2] = compute_rmse_r2(y_true, y_pred)
    mask = isfinite(y_true) & isfinite(y_pred);
    if nnz(mask)<5, rmse=NaN; r2=NaN; return; end
    e=y_true(mask)-y_pred(mask);
    rmse=sqrt(mean(e.^2));
    ss_res=sum(e.^2);
    ss_tot=sum((y_true(mask)-mean(y_true(mask))).^2);
    if ss_tot<1e-12, r2=NaN; else, r2=1-ss_res/ss_tot; end
end
