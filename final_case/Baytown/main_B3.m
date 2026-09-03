clear; clc; close all; tic;
rng(1, 'twister');

%% ===== User settings =====
Scenario     = 2;
T_mcmc       = 5;
SAVE_TMCMC   = false;
SAVE_FIGURES = false;

%% ===== Load model parameters and site data =====
% HBM_CLAY_8_OCR_sigv_new.mat:
%   Johnson transformation parameters used by HBM-MUSIC-3X.
%
% Cs.mat:
%   Cross-parameter covariance matrix adopted from HBM-MUSIC-3X.
%
% Cs_global.mat:
%   Global cross-parameter covariance matrix estimated from the database.
%
% integrated_geotechnical_data_0521.mat:
%   Baytown site observations, coordinates, depths, and effective stresses.

load('HBM_CLAY_8_OCR_sigv_new.mat');
load('Cs_global.mat');
load('integrated_geotechnical_data_0521.mat');

full_z    = data_structure.depth;
dep_range = find(full_z >= 1 & full_z <= 10);
z  = full_z(dep_range);
nz = length(z);

if Scenario == 1
    para_Cs    = [7, 8];
    param_name = {'B_q', 'q_{t1}'};
    log_ind    = [0, 1];
    numbb      = 8:14;
    sounding   = length(numbb);
    E = [6.401 6.401 0.000 12.802 0.000 12.802 0.000 12.802];
    N = [4.267 12.802 25.604 25.604 12.802 12.802 0.000 0.000];

elseif Scenario == 2
    para_Cs    = [1, 2, 3,4, 5, 6, 7,8];
    param_name = {'LL (%)', 'PI (%)', 'LI', '\sigma''_v/P_a','OCR', 's_u (kN/m^2)', 'Bq','q_t - \sigma_v (kN/m^2)'};
    log_ind    = [1 1 0 1 1 1 0 1];
    numbb      = 1:14;
    sounding   = 14;
    E = [6.401 6.401 6.401 6.401 6.401 6.401 6.401 6.401 0.000 12.802 0.000 12.802 0.000 12.802];
    N = [6.4 14.173 1.372 19.202 26.974 4.267 12.802 21.336 25.604 25.604 12.802 12.802 0.000 0.000];
end
%% ===== Select the cross-parameter covariance matrix =====
% Current setting: use the global covariance matrix.
% To use the HBM-MUSIC-3X covariance, load Cs.mat and replace
% Cs_global below with the variable stored in Cs.mat.
Cs = Cs_global;
Cs = Cs(para_Cs, para_Cs);
Cs = (Cs + Cs') / 2;
M  = size(Cs, 1);

%% ===== Load and preprocess observations =====
data      = data_structure.data;
all_nh_db = size(data_structure.data, 2) / 8;
cols = [];
for i = para_Cs
    cols = [cols, (i-1)*all_nh_db + numbb];
end
data = data(dep_range, cols);

param_list_all = cell(1, M);
for k = 1:M
    col_idx = (1:sounding) + (k-1)*sounding;
    param_list_all{k} = data(:, col_idx);
end
param_list_original_all = param_list_all;

for p = 1:M
    param_idx     = para_Cs(p);
    data_p        = param_list_all{p}(:);
    if log_ind(p) > 0.5; data_p = log(data_p); end
    data_p_normal = nan(size(data_p));
    valid_mask    = ~isnan(data_p);
    data_p_normal(valid_mask) = JS_2_normal(data_p(valid_mask), ...
        type(param_idx), ax(param_idx), bx(param_idx), ay(param_idx), by(param_idx));
    param_list_all{p} = reshape(data_p_normal, nz, sounding);
end

if Scenario == 1
    location_labels = arrayfun(@(x) sprintf('CPT%d', x), 1:sounding, 'UniformOutput', false);
else
    location_labels = data_structure.positions(numbb);
end

X_all = E(1:sounding)';
Y_all = N(1:sounding)';

%% ===== Remove CPT-3 observations with Bq < -0.07 =====
dist_cpt3 = sqrt((X_all - 6.401).^2 + (Y_all - 21.336).^2);
[~, idx_cpt3_global] = min(dist_cpt3);
idx_cpt3_local = find(numbb == idx_cpt3_global);

if ~isempty(idx_cpt3_local)
    if Scenario == 1
        p_Bq  = 2; p_qt1 = 1;
        Bq_col = param_list_original_all{p_Bq}(:, idx_cpt3_local);
        mask_bad = Bq_col < -0.07;
        param_list_all{p_Bq}(mask_bad, idx_cpt3_local)  = NaN;
        param_list_all{p_qt1}(mask_bad, idx_cpt3_local) = NaN;
        param_list_original_all{p_Bq}(mask_bad, idx_cpt3_local)  = NaN;
        param_list_original_all{p_qt1}(mask_bad, idx_cpt3_local) = NaN;
        fprintf('CPT-3 filter: %d depth points were set to NaN.\n', sum(mask_bad));
    elseif Scenario == 2
        bq_param_idx = 7;
        col_bq = (bq_param_idx - 1)*all_nh_db + numbb(idx_cpt3_local);
        Bq_raw = data_structure.data(dep_range, col_bq);
        mask_bad = Bq_raw < -0.07;
        p_Bq = 7; p_qt = 8;
        param_list_all{p_Bq}(mask_bad, idx_cpt3_local) = NaN;
        param_list_original_all{p_Bq}(mask_bad, idx_cpt3_local) = NaN;
        param_list_all{p_qt}(mask_bad, idx_cpt3_local) = NaN;
        param_list_original_all{p_qt}(mask_bad, idx_cpt3_local) = NaN;
        fprintf('CPT-3 filter (Scenario 2): %d depth points were set to NaN.\n', sum(mask_bad));
    end
end

%% ===== Effective vertical stress =====
sigvp_all = data_structure.sigvp(dep_range, :);

%% ===== Define training and test locations =====
E_train = [6.401  4.267;
           6.401  12.802;
           6.401  21.336;
           0.000  25.604;
           12.802 25.604;
           0.000  12.802;
           12.802 12.802;
           0.000  0.000;
           12.802 0.000;
           6.401  6.400;
           6.401  14.173;
           6.401  1.372;
           6.401  19.202;
           6.401  26.974];

E_test = [6.401  1.372];

tol = 1e-3;

train_idx = zeros(1, size(E_train,1));
for k = 1:size(E_train,1)
    dist = sqrt((X_all - E_train(k,1)).^2 + (Y_all - E_train(k,2)).^2);
    [d_min, i_min] = min(dist);
    assert(d_min < tol, 'Training location [%.3f, %.3f] was not found.', E_train(k,1), E_train(k,2));
    train_idx(k) = i_min;
end

nh_test  = size(E_test, 1);
test_idx = nan(1, nh_test);
X_test   = E_test(:, 1);
Y_test   = E_test(:, 2);
test_label_list = cell(1, nh_test);

for k = 1:nh_test
    dist = sqrt((X_all - E_test(k,1)).^2 + (Y_all - E_test(k,2)).^2);
    [d_min, i_min] = min(dist);
    if d_min < tol
        test_idx(k) = i_min;
        test_label_list{k} = location_labels{i_min};
        fprintf('Test %d: [%.3f, %.3f] -> %s (observed data available).\n', ...
            k, E_test(k,1), E_test(k,2), location_labels{i_min});
    else
        test_label_list{k} = sprintf('(%.3f,%.3f)', E_test(k,1), E_test(k,2));
        fprintf('Test %d: [%.3f, %.3f] -> prediction-only location.\n', ...
            k, E_test(k,1), E_test(k,2));
    end
end

nh_train = length(train_idx);
fprintf('Training locations (%d): %s\n', nh_train, ...
    strjoin(location_labels(train_idx), ', '));

%% ===== Prepare effective stress at each test location =====
sigvp_test = nan(nz, nh_test);
for k = 1:nh_test
    if ~isnan(test_idx(k))
        sigvp_test(:,k) = sigvp_all(:, test_idx(k));
    else
        sigvp_test(:,k) = nanmean(sigvp_all(:, train_idx), 2);
    end
end

%% ===== Preallocate prediction arrays =====
pred_median = nan(nz, nh_test, M);
pred_p025   = nan(nz, nh_test, M);
pred_p975   = nan(nz, nh_test, M);
pred_single = nan(nz, nh_test, M);
true_data   = nan(nz, nh_test, M);

for k = 1:nh_test
    if ~isnan(test_idx(k))
        for pm = 1:M
            true_data(:, k, pm) = param_list_original_all{pm}(:, test_idx(k));
        end
    end
end

%% ===== Assemble centered training data =====
X = X_all(train_idx);
Y = Y_all(train_idx);

param_list = cell(1, M);
for k = 1:M
    param_list{k} = param_list_all{k}(:, train_idx);
end

param_mean = nan(1, M);
t = zeros(nz * nh_train, M);
for p = 1:M
    tvec = reshape(param_list{p}, [], 1);
    param_mean(p) = nanmean(tvec);
    t(:, p) = tvec - param_mean(p);
end

temp_x = abs(X*ones(1,nh_train) - (X*ones(1,nh_train))');
temp_y = abs(Y*ones(1,nh_train) - (Y*ones(1,nh_train))');
temp_z = abs(z*ones(1,nz) - (z*ones(1,nz))');
temp_h = sqrt(temp_x.^2 + temp_y.^2);

%% ===== TMCMC =====
y.z = z; y.X = X; y.Y = Y;
y.t = real(t(:));
y.temp_z = temp_z; y.temp_h = temp_h;

[phi_t_Cs, omege_t_Cs] = eig(Cs);
eigvals = diag(omege_t_Cs);
[~, idx] = sort(eigvals, 'descend');
phi_t_Cs   = phi_t_Cs(:, idx);
L_Cs_fixed = chol(Cs, 'lower');
%% ===== TMCMC parameter bounds =====
% Parameter order:
% x(1): log inverse residual amplitude,  bhp = 1/exp(x(1))
% x(2): log vertical residual SOF
% x(3): log horizontal residual SOF
% x(4): log Matern smoothness parameter
% x(5): log inverse trend amplitude,     ahp = 1/exp(x(5))
% x(6): log vertical trend SOF
% x(7): log horizontal trend SOF
%
% A trend SOF much larger than the corresponding site dimension
% produces an almost constant trend in that direction.
% If both vertical and horizontal trend SOFs are very large,
% the MGPR trend approaches the t-const model.
x_low = [-log(10), log(0.01), log(0.1),  log(0.1), -log(10),  log(max(temp_z(:))/10), log(max(temp_h(:))/10)];
x_up  = [-log(0.01),log(100), log(100), log(3.0), -log(0.01), log(max(temp_z(:))*10), log(max(temp_h(:))*10)];

y.eig_thresh = 0.999;

temp_x_ele = abs([X; X_test] - [X; X_test]');
temp_y_ele = abs([Y; Y_test] - [Y; Y_test]');
temp_z_ele = abs([z; z] - [z; z]');
temp_h_ele = sqrt(temp_x_ele.^2 + temp_y_ele.^2);
y.temp_z_ele = temp_z_ele; y.temp_h_ele = temp_h_ele;

fprintf('\nStarting TMCMC...\n');
[x_mcmc, ln_S, ~, ~, ~] = iTMCMC_fun_mod1('BaytownGP_Matern_3D', y, x_low, x_up, T_mcmc, 0.5, Cs);
bhp_mcmc    = 1./exp(x_mcmc(:,1));
sofv_mcmc   = exp(x_mcmc(:,2));
sofh_mcmc   = exp(x_mcmc(:,3));
nuv_mcmc    = exp(x_mcmc(:,4));
nuh_mcmc    = nuv_mcmc;
ahp_mcmc    = 1./exp(x_mcmc(:,5));
sofv_t_mcmc = exp(x_mcmc(:,6));
sofh_t_mcmc = exp(x_mcmc(:,7));
fprintf('TMCMC completed.\n');

if SAVE_TMCMC
    save('TMCMC_B3out_0611.mat', ...
        'bhp_mcmc', 'sofv_mcmc', 'sofh_mcmc', 'nuv_mcmc', 'nuh_mcmc', ...
        'ahp_mcmc', 'sofv_t_mcmc', 'sofh_t_mcmc', ...
        'x_mcmc', 'ln_S', ...
        'x_low', 'x_up', 'T_mcmc', 'Scenario', 'para_Cs', 'param_name');
    fprintf('Saved TMCMC_B3out_0611.mat.\n');
end

%% ===== Posterior parameter summary on log-log axes =====
fprintf('Plotting the posterior parameter summary...\n');

% Use the common sample count if arrays have different lengths.
N_post = min([numel(nuv_mcmc), numel(bhp_mcmc), numel(ahp_mcmc), ...
              numel(sofv_mcmc), numel(sofh_mcmc), ...
              numel(sofv_t_mcmc), numel(sofh_t_mcmc)]);

nu_s    = nuv_mcmc(1:N_post);      % nu_h = nu_v in the current model.
bhp_s   = bhp_mcmc(1:N_post);      % Residual amplitude beta.
ahp_s   = ahp_mcmc(1:N_post);      % Trend amplitude alpha.
sofv_s  = sofv_mcmc(1:N_post);     % Vertical residual SOF.
sofh_s  = sofh_mcmc(1:N_post);     % Horizontal residual SOF.
sofvt_s = sofv_t_mcmc(1:N_post);   % Vertical trend SOF.
sofht_s = sofh_t_mcmc(1:N_post);   % Horizontal trend SOF.

fig = figure('Name', 'GP Posterior Summary', ...
    'Color', 'w', ...
    'Position', [50 50 1400 360]);

col_hist = [0.7 0.7 0.7];
col_sc   = [1.0 0.85 0.0];
ms       = 30;

%% (a) Residual SOF posterior
ax1 = subplot(1,4,1);
valid1 = isfinite(sofv_s) & isfinite(sofh_s) & sofv_s > 0 & sofh_s > 0;
scatter(ax1, sofv_s(valid1), sofh_s(valid1), ms, col_sc, 'filled', ...
    'MarkerEdgeColor', 'k', 'LineWidth', 1.2);
hold(ax1, 'on'); box(ax1, 'on'); grid(ax1, 'on');
set(ax1, 'XScale', 'log', 'YScale', 'log');
xlabel(ax1, '\delta_z^{(\epsilon)} (m)', 'FontSize', 11, 'FontWeight', 'normal');
ylabel(ax1, '\delta_h^{(\epsilon)} (m)', 'FontSize', 11, 'FontWeight', 'normal');

x1_min = min(sofv_s, [], 'omitnan'); x1_max = max(sofv_s, [], 'omitnan');
y1_min = min(sofh_s, [], 'omitnan'); y1_max = max(sofh_s, [], 'omitnan');
if isfinite(x1_min) && isfinite(x1_max) && x1_max > x1_min
    xlim(ax1, [x1_min*0.95, x1_max*1.05]);
end
if isfinite(y1_min) && isfinite(y1_max) && y1_max > y1_min
    ylim(ax1, [y1_min*0.95, y1_max*1.05]);
end

%% (b) Residual smoothness posterior
ax2 = subplot(1,4,2);
nu_pos = nu_s(isfinite(nu_s) & nu_s > 0);
if isempty(nu_pos)
    error('No positive finite nu samples are available for the log-log plot.');
elseif max(nu_pos) > min(nu_pos)
    nu_edges = logspace(log10(min(nu_pos)), log10(max(nu_pos)), 31);
    histogram(ax2, nu_pos, 'BinEdges', nu_edges, ...
        'FaceColor', col_hist, 'EdgeColor', 'w');
else
    histogram(ax2, nu_pos, 1, 'FaceColor', col_hist, 'EdgeColor', 'w');
end
hold(ax2, 'on'); box(ax2, 'on'); grid(ax2, 'on');
set(ax2, 'XScale', 'log', 'YScale', 'log');
xlabel(ax2, '\nu^{(\epsilon)}', 'FontSize', 11, 'FontWeight', 'normal');
ylabel(ax2, 'Frequency', 'FontSize', 11, 'FontWeight', 'normal');

%% (c) Trend SOF posterior
ax3 = subplot(1,4,3);
valid3 = isfinite(sofvt_s) & isfinite(sofht_s) & sofvt_s > 0 & sofht_s > 0;
scatter(ax3, sofvt_s(valid3), sofht_s(valid3), ms, col_sc, 'filled', ...
    'MarkerEdgeColor', 'k', 'LineWidth', 1.2);
hold(ax3, 'on'); box(ax3, 'on'); grid(ax3, 'on');
set(ax3, 'XScale', 'log', 'YScale', 'log');
xlabel(ax3, '\delta_z^{(t)} (m)', 'FontSize', 11, 'FontWeight', 'normal');
ylabel(ax3, '\delta_h^{(t)} (m)', 'FontSize', 11, 'FontWeight', 'normal');

x3_min = min(sofvt_s, [], 'omitnan'); x3_max = max(sofvt_s, [], 'omitnan');
y3_min = min(sofht_s, [], 'omitnan'); y3_max = max(sofht_s, [], 'omitnan');
if isfinite(x3_min) && isfinite(x3_max) && x3_max > x3_min
    xlim(ax3, [x3_min*0.95, x3_max*1.05]);
end
if isfinite(y3_min) && isfinite(y3_max) && y3_max > y3_min
    ylim(ax3, [y3_min*0.95, y3_max*1.05]);
end

%% (d) Residual and trend amplitudes
ax4 = subplot(1,4,4);
valid4 = isfinite(bhp_s) & isfinite(ahp_s) & bhp_s > 0 & ahp_s > 0;
scatter(ax4, bhp_s(valid4), ahp_s(valid4), ms, col_sc, 'filled', ...
    'MarkerEdgeColor', 'k', 'LineWidth', 1.2);
hold(ax4, 'on'); box(ax4, 'on'); grid(ax4, 'on');
set(ax4, 'XScale', 'log', 'YScale', 'log');
xlabel(ax4, '\beta', 'FontSize', 11, 'FontWeight', 'normal');
ylabel(ax4, '\alpha', 'FontSize', 11, 'FontWeight', 'normal');

x4_min = min(bhp_s, [], 'omitnan'); x4_max = max(bhp_s, [], 'omitnan');
y4_min = min(ahp_s, [], 'omitnan'); y4_max = max(ahp_s, [], 'omitnan');
if isfinite(x4_min) && isfinite(x4_max) && x4_max > x4_min
    xlim(ax4, [x4_min*0.95, x4_max*1.05]);
end
if isfinite(y4_min) && isfinite(y4_max) && y4_max > y4_min
    ylim(ax4, [y4_min*0.95, y4_max*1.05]);
end

% Apply a consistent publication style.
for ax_now  = [ax1 ax2 ax3 ax4]
    set(ax_now , ...
        'FontSize', 10, ...
        'TickDir', 'in', ...
        'LineWidth', 1.0, ...
        'Box', 'on');
end

sgtitle(sprintf('GP Posterior Summary (Scenario %d, N = %d)', Scenario, N_post), ...
    'FontSize', 13, 'FontWeight', 'bold');

if SAVE_FIGURES
    exportgraphics(fig, 'posterior_4panel_B3.png', 'Resolution', 300);
    savefig(fig, 'posterior_4panel_B3.fig');
end


%% ===== CPU conditional random-field simulation =====

% Each TMCMC sample defines one conditional realization at the test location.

w_store           = cell(nh_test, T_mcmc);
phiz_store        = cell(nh_test, T_mcmc);
phih_store        = cell(nh_test, T_mcmc);

fprintf('\nStarting CPU conditional simulation for %d test location(s)...\n', nh_test);

jitterRh = 1e-6;
jitterRz = 1e-6;
jitterP  = 1e-11;

z_ele = z;

for ti = 1:nh_test

    X_ele = X_test(ti);
    Y_ele = Y_test(ti);

    fprintf('\n============================================================\n');
    fprintf('  Predicting %s (%d/%d)\n', test_label_list{ti}, ti, nh_test);
    fprintf('============================================================\n');

    % Distance matrices between training and current test locations.
    temp_x_ele_ti = abs([X; X_ele] - [X; X_ele]');
    temp_y_ele_ti = abs([Y; Y_ele] - [Y; Y_ele]');
    temp_z_ele_ti = abs([z; z_ele] - [z; z_ele]');
    temp_h_ele_ti = sqrt(temp_x_ele_ti.^2 + temp_y_ele_ti.^2);

    y_ti = y;
    y_ti.temp_z_ele = temp_z_ele_ti;
    y_ti.temp_h_ele = temp_h_ele_ti;

    % Preallocate realizations and stored trend quantities.
    t_ele = zeros(length(z_ele) * M, T_mcmc);

    w_store_ti           = cell(T_mcmc, 1);
    phiz_store_ti        = cell(T_mcmc, 1);
    phih_store_ti        = cell(T_mcmc, 1);
    % Per-realization CPU timings.
    dw_time_vec   = zeros(T_mcmc, 1);
    iter_time_vec = zeros(T_mcmc, 1);

    printEvery = max(1, min(20, T_mcmc));

    tic_ti_wall = tic;
    fprintf('  Running %d CRF realizations on the CPU...\n', T_mcmc);

    for i = 1:T_mcmc

        tic_iter = tic;
        dw_time_i = 0;

        % Build the trend basis for this posterior sample.
        y_local = y_ti;

        [y_phiz, y_phih, y_phiz_ele, y_phih_ele, ~, ln_alpha] = ...
            GP_matrices_Step3(ahp_mcmc(i), sofv_t_mcmc(i), sofh_t_mcmc(i), y_local, Cs);

        y_local.phiz     = y_phiz;
        y_local.phih     = y_phih;
        y_local.phiz_ele = y_phiz_ele;
        y_local.phih_ele = y_phih_ele;

        A_diag = exp(ln_alpha(:));

        % Residual covariance factors.
        R_h = Matern_R(nuh_mcmc(i), sofh_mcmc(i), temp_h);
        R_z = Matern_R(nuv_mcmc(i), sofv_mcmc(i), temp_z);

        Rh = R_h + jitterRh * eye(nh_train);
        Rz = R_z + jitterRz * eye(nz);

        Lh_R = chol(Rh, 'lower');
        Lz_R = chol(Rz, 'lower');

        % Arrange the training data for Kronecker operations.
        reshape_data = reshape(y_local.t, nz, nh_train * M);

        % Fill missing non-lattice observations before conditioning.
        if Scenario == 2

            tic_dw = tic;

            reshape_data = DW_sampler_new2(reshape_data, X, Y, z, ...
                sofv_mcmc(i), sofh_mcmc(i), nuv_mcmc(i), nuh_mcmc(i), ...
                bhp_mcmc(i), A_diag, y_local, Cs, ...
                sofv_t_mcmc(i), sofh_t_mcmc(i), M);

            dw_time_i = toc(tic_dw);
        end

        reshape_vec = reshape(reshape_data, [], 1);

        % Posterior distribution of the trend weights.
        AA = (Lh_R' \ (Lh_R \ y_phih)).';
        BB = (Lz_R' \ (Lz_R \ y_phiz)).';
        CC = (L_Cs_fixed' \ (L_Cs_fixed \ phi_t_Cs)).';

        temp_vec = kronmult2({CC, AA, BB}, reshape_vec);

        bhp = bhp_mcmc(i);

        P = spdiags(A_diag, 0, length(A_diag), length(A_diag)) + ...
            (1 / bhp) * kron(CC * phi_t_Cs, kron(AA * y_phih, BB * y_phiz));

        P = (P + P') / 2 + jitterP * speye(size(P,1));

        try
            R_chol = chol(P, 'lower');
        catch
            Pf = full(P);
            Pf = (Pf + Pf') / 2 + jitterP * eye(size(Pf,1));
            R_chol = chol(Pf, 'lower');
        end

        mu = (1 / bhp) * (R_chol' \ (R_chol \ temp_vec));

        w = mu + (R_chol' \ (R_chol \ randn(size(mu))));

        % Conditional residual at the test location.
        w_row = w.';

        dd_temp = kronmult2({phi_t_Cs, y_phih, y_phiz}, w);
        t_diff  = reshape_vec - dd_temp;

        [E_X, L_h, L_z] = vertical_dense_stats( ...
            sofv_mcmc(i), sofh_mcmc(i), ...
            nuv_mcmc(i), nuh_mcmc(i), ...
            X_ele, Y_ele, z_ele, X, Y, z, t_diff, M);

        noise = E_X + sqrt(bhp) * kronmult2({L_Cs_fixed, L_h, L_z}, ...
            randn(length(z_ele) * M, 1));

        % Trend at the test location.
        d_ele = kronmult2({phi_t_Cs, y_phih_ele, y_phiz_ele}, reshape(w_row.', [], M));

        % Combine trend and residual components.
        t_ele(:, i) = d_ele(:) + noise(:);

        % Store quantities required by the trend plots.
        w_store_ti{i}           = w;
        phiz_store_ti{i}        = y_phiz;
        phih_store_ti{i}        = y_phih;

        % Record CPU time and print lightweight progress information.
        iter_time_i = toc(tic_iter);

        dw_time_vec(i)   = dw_time_i;
        iter_time_vec(i) = iter_time_i;

        if mod(i, printEvery) == 0 || i == T_mcmc
            fprintf('    Completed %d/%d realizations.\n', i, T_mcmc);
        end

    end

    wall_ti = toc(tic_ti_wall);

    % CPU timing summary.
    valid_dw = dw_time_vec(dw_time_vec > 0);
    valid_it = iter_time_vec(iter_time_vec > 0);

    fprintf('\n------------------------------------------------------------\n');
    fprintf('  CPU timing summary for %s\n', test_label_list{ti});
    fprintf('------------------------------------------------------------\n');
    fprintf('  total CPU wall time           = %.2f s\n', wall_ti);
    fprintf('  completed samples             = %d / %d\n', numel(valid_it), T_mcmc);

    if ~isempty(valid_dw)
        fprintf('  mean DW time per sample        = %.3f s\n', mean(valid_dw));
        fprintf('  median DW time per sample      = %.3f s\n', median(valid_dw));
        fprintf('  min / max DW time              = %.3f / %.3f s\n', min(valid_dw), max(valid_dw));
    end

    if ~isempty(valid_it)
        fprintf('  mean total iter time/sample    = %.3f s\n', mean(valid_it));
        fprintf('  median total iter time/sample  = %.3f s\n', median(valid_it));
        fprintf('  min / max iter time            = %.3f / %.3f s\n', min(valid_it), max(valid_it));

        fprintf('  effective wall time / sample   = %.3f s/sample\n', wall_ti / numel(valid_it));
    end

    fprintf('------------------------------------------------------------\n\n');

    % Copy the current location's trend quantities to the main stores.
    for i = 1:T_mcmc
        w_store{ti, i}           = w_store_ti{i};
        phiz_store{ti, i}        = phiz_store_ti{i};
        phih_store{ti, i}        = phih_store_ti{i};
    end

    % Transform conditional realizations back to physical space.
    t_ele_original = zeros(size(t_ele));

    for i = 1:T_mcmc

        t_ele_reshaped = reshape(t_ele(:, i), nz, M);

        for p = 1:M

            param_idx_p = para_Cs(p);

            data_normal = t_ele_reshaped(:, p) + param_mean(p);

            tmp = JS_2_original(data_normal, ...
                type(param_idx_p), ...
                ax(param_idx_p), bx(param_idx_p), ...
                ay(param_idx_p), by(param_idx_p));

            if log_ind(p) > 0.5
                t_ele_reshaped(:, p) = exp(tmp);
            else
                t_ele_reshaped(:, p) = tmp;
            end
        end

        t_ele_original(:, i) = t_ele_reshaped(:);
    end

    % Summarize the conditional ensemble.
    for p = 1:M

        row_idx = (1:nz) + (p-1) * nz;

        p_samp = real(t_ele_original(row_idx, :));

        pred_median(:, ti, p) = prctile(p_samp, 50,   2);
        pred_p025(:,   ti, p) = prctile(p_samp, 2.5,  2);
        pred_p975(:,   ti, p) = prctile(p_samp, 97.5, 2);
        pred_single(:, ti, p) = p_samp(:, 1);
    end

    fprintf('  Completed test location %s.\n', test_label_list{ti});

end  % for ti

fprintf('\n===== CPU conditional simulation completed =====\n');

%% ===== Plot GPR and HBM comparison =====
fprintf('Plotting the GPR and HBM comparison...\n');

is_Bq = false(1, M);
for p = 1:M
    name_tmp = lower(param_name{p});
    name_tmp = erase(name_tmp, {'_', ' ', '{', '}', '\'});
    is_Bq(p) = contains(name_tmp, 'bq');
end

plot_param_idx = [1 2 3 5 6 8];
M_plot = numel(plot_param_idx);
fprintf('Number of plotted parameters = %d; skipped parameters: ', M_plot);
disp(param_name(is_Bq));

fig_hbm = openfig('HBM_only_B3.fig', 'invisible');
drawnow;
all_ax_hbm  = findall(fig_hbm, 'Type', 'axes');
pos_hbm     = cell2mat(get(all_ax_hbm, 'Position'));
[~, sidx_h] = sortrows([-pos_hbm(:,2), pos_hbm(:,1)]);
all_ax_hbm  = all_ax_hbm(sidx_h);
is_real_hbm = arrayfun(@(a) a.Position(3)>0.05 && a.Position(4)>0.05, all_ax_hbm);
real_ax_hbm = all_ax_hbm(is_real_hbm);

HBM_line_data = cell(numel(real_ax_hbm), 1);
for pp = 1:numel(real_ax_hbm)
    lines_p = findobj(real_ax_hbm(pp), 'Type','line');
    line_store = struct();
    for li = 1:numel(lines_p)
        line_store(li).XData     = get(lines_p(li),'XData');
        line_store(li).YData     = get(lines_p(li),'YData');
        line_store(li).LineStyle = get(lines_p(li),'LineStyle');
        line_store(li).Color     = get(lines_p(li),'Color');
        line_store(li).LineWidth = get(lines_p(li),'LineWidth');
    end
    HBM_line_data{pp} = line_store;
end
close(fig_hbm);

col_HBM_med = [0.0  0.0  0.0];
col_HBM_CI  = [0.0  0.0  0.0];
col_GPR_med = [0.85 0.0  0.85];
col_GPR_CI  = [0.85 0.0  0.85];
col_CRF     = [0.0  0.65 0.0];
col_data    = [1.0  0.9  0.0];
col_sand    = [0.75 0.75 0.75];

lw_HBM_med = 2.0; lw_HBM_CI = 1.1;
lw_GPR_med = 2.0; lw_GPR_CI = 1.1;
lw_CRF     = 1.1;

if M == 8
    x_lims_all = {[0,150],[0,100],[-0.75,1.25],[-0.75,1.25],[1,100],[10,1000],[],[100,10000]};
    use_log_all        = [false,false,false,false,true,true,false,true];
    multiply_sigvp_all = [false,false,false,false,false,true,false,true];
else
    x_lims_all = cell(1, M);
    use_log_all        = false(1, M);
    multiply_sigvp_all = false(1, M);
    for p = 1:M
        if log_ind(p) > 0.5, use_log_all(p) = true; end
    end
end

fig_W = 145*M_plot + 260;
fig_H = 660;
left_margin    = 0.06;
right_boundary = 0.75;
bot            = 0.09;
top_margin     = 0.04;
ax_height      = 1 - bot - top_margin;
h_gap          = 0.006;
tile_w         = (right_boundary - left_margin - h_gap*(M_plot-1)) / M_plot;

for ti = 1:nh_test
    svp = sigvp_test(:, ti);
    fig = figure('Name', sprintf('GPR vs HBM - %s', test_label_list{ti}), ...
        'Position',[50,50,fig_W,fig_H], 'Color','w');

    hl = gobjects(6,1);
    hl_hbm_leg = gobjects(2,1);
    hbm_leg_done = false;

    for pp = 1:M_plot
        p = plot_param_idx(pp);
        lp   = left_margin + (pp-1)*(tile_w + h_gap);
        ax_h = axes('Position',[lp, bot, tile_w, ax_height]); %#ok<LAXES>
        hold(ax_h,'on'); box(ax_h,'on');
        set(ax_h,'YDir','reverse','FontSize',9, ...
            'XGrid','on','YGrid','on','GridAlpha',0.3,'Layer','bottom');
        ylim(ax_h,[min(z) max(z)]);
        if use_log_all(p), set(ax_h,'XScale','log'); end
        if ~isempty(x_lims_all{p}), xlim(ax_h, x_lims_all{p}); end
        xlabel(ax_h, param_name{p}, 'FontSize',9, 'FontWeight','bold');
        if pp == 1
            ylabel(ax_h,'Depth (m)','FontSize',9);
        else
            set(ax_h,'YTickLabel',{});
        end

        if pp <= numel(HBM_line_data)
            ld = HBM_line_data{pp};
            hv = gobjects(numel(ld),1);
            for li = 1:numel(ld)
                hv(li) = plot(ax_h, ld(li).XData, ld(li).YData, ...
                    'LineStyle', ld(li).LineStyle, 'Color', ld(li).Color, ...
                    'LineWidth', ld(li).LineWidth, 'HandleVisibility','off');
            end
            if ~hbm_leg_done && numel(ld) > 0
                is_dash  = arrayfun(@(h) strcmp(get(h,'LineStyle'),'--'), hv);
                is_solid = arrayfun(@(h) strcmp(get(h,'LineStyle'),'-'),  hv);
                if any(is_dash)
                    set(hv(find(is_dash,1)),'HandleVisibility','on');
                    hl_hbm_leg(1) = hv(find(is_dash,1));
                end
                if any(is_solid)
                    set(hv(find(is_solid,1)),'HandleVisibility','on');
                    hl_hbm_leg(2) = hv(find(is_solid,1));
                end
                hbm_leg_done = true;
            end
        end

        obs_val  = true_data(:, ti, p);
        pred_med = pred_median(:, ti, p);
        pred_lo  = pred_p025(:,  ti, p);
        pred_hi  = pred_p975(:,  ti, p);
        pred_crf = pred_single(:, ti, p);

        if multiply_sigvp_all(p)
            obs_val  = obs_val  .* svp; pred_med = pred_med .* svp;
            pred_lo  = pred_lo  .* svp; pred_hi  = pred_hi  .* svp;
            pred_crf = pred_crf .* svp;
        end
        if use_log_all(p)
            pred_lo(pred_lo<=0)=NaN; pred_hi(pred_hi<=0)=NaN;
            pred_med(pred_med<=0)=NaN; pred_crf(pred_crf<=0)=NaN;
            obs_val(obs_val<=0)=NaN;
        end

        hl(3) = plot(ax_h, pred_lo,  z, '--', 'Color',col_GPR_CI,  'LineWidth',lw_GPR_CI);
                plot(ax_h, pred_hi,  z, '--', 'Color',col_GPR_CI,  'LineWidth',lw_GPR_CI, 'HandleVisibility','off');
        hl(4) = plot(ax_h, pred_med, z, '-',  'Color',col_GPR_med, 'LineWidth',lw_GPR_med);
        hl(5) = plot(ax_h, pred_crf, z, '-',  'Color',col_CRF,     'LineWidth',lw_CRF);

        xl = xlim(ax_h);
        patch(ax_h, [xl(1) xl(2) xl(2) xl(1)], [3.4 3.4 4.66 4.66], ...
            col_sand, 'FaceAlpha',1.0,'EdgeColor','none','HandleVisibility','off');

        valid = isfinite(obs_val) & isfinite(z);
        if any(valid)
            hl(6) = scatter(ax_h, obs_val(valid), z(valid), 55, 'o', ...
                'MarkerFaceColor',col_data,'MarkerEdgeColor','k','LineWidth',1.1);
        else
            hl(6) = scatter(ax_h, NaN, NaN, 55, 'o', ...
                'MarkerFaceColor',col_data,'MarkerEdgeColor','k');
        end
    end

    hl(1) = hl_hbm_leg(1); hl(2) = hl_hbm_leg(2);
    valid_hl = arrayfun(@(h) isgraphics(h), hl);
    if all(valid_hl)
        ax_leg = axes('Position', ...
            [right_boundary+0.01, bot+ax_height*0.25, ...
             1-right_boundary-0.02, ax_height*0.55], 'Visible','off');
        lg = legend(ax_leg, hl, ...
            {'95% CI (HBM-MUSIC-3X)','Median (HBM-MUSIC-3X)', ...
             '95% CI (GPR-MUSIC-3X)','Median (GPR-MUSIC-3X)', ...
             'GPR CRF sample','Actual data'}, ...
            'Location','northwest','FontSize',8,'Box','on');
        lg.Position(1) = right_boundary + 0.01;
        lg.ItemTokenSize = [18 8]; lg.AutoUpdate = 'off';
    else
        warning('Some legend handles are invalid; the legend was skipped.');
    end

    if SAVE_FIGURES
        exportgraphics(fig, sprintf('GPR_vs_HBM_%s.png', test_label_list{ti}), ...
            'Resolution', 300);
        savefig(fig, sprintf('MGPR_vs_HBM_0610%s.fig', test_label_list{ti}));
    end
end
fprintf('Comparison plots completed.\n');


