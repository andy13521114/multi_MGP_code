%% MUSIC-X Taipei: t-const MGPR vs MGPR
%  功能：
%  1) 同時跑 t-const 與 MGPR
%  2) 記錄 TMCMC time、CRF generation time
%  3) 輸出兩模型 log evidence
%  4) 輸出 MAE、RMSE_point、RMSE_BV = sqrt(mean(bias^2 + predictive variance))、95% CI coverage、CI width
%  5) 畫 paper-style 比較圖：灰色=t-const，紫色=MGPR，綠色=one MGPR CRF realization，黃色=observed data
%
%  注意：本主程式沿用你原本的函式：
%  iTMCMC_fun_mod1, BaytownGP_Matern_3D, GP_matrices_Step3,
%  Matern_R, DW_sampler_new2, vertical_dense_stats, kronmult2,
%  JS_2_normal_gb, JS_2_original_gb

clear; clc; close all;

tic_total = tic;
rng(10,'twister');
rand('state',10); %#ok<RAND>
randn('state',10); %#ok<RAND>

%% ============================================================
%  0. 使用者設定
%% ============================================================
T_mcmc = 1000;       % 正式建議 1000；快速測試可改 10 或 50
tmcmc_beta = 0.5;

SAVE_OUTPUT = true;
PLOT_ONE_CRF = true;

% 原始10維順序: 1=LL,2=PI,3=LI,4=sv,5=sp,6=su,7=St,8=Bq,9=qt1,10=qtu
use_param = [1 1 1 1 1 1 0 0 1 0];
%              LL PI LI sv sp su St Bq qt1 qtu

Pa = 101.3;   % kN/m^2 = kPa

%% ============================================================
%  1. 基礎設定與載入
%% ============================================================
load CLAY_10_7490_para_rho.mat;   % needs: type, ax, bx, ay, by, rho

keep_idx = find(use_param);
M = length(keep_idx);
fprintf('\n============================================================\n');
fprintf('使用 %d 個參數，index: %s\n', M, num2str(keep_idx));
fprintf('============================================================\n');

log_ind_all  = [1 1 0 1 1 1 1 0 1 1];
param_name_all = {'LL (%)','PI (%)','LI (%)','\sigma''_v/P_a','\sigma''_p/P_a', ...
                  's_u/\sigma''_v','S_t','B_q','q_{t1}','q_{tu}'};

log_ind    = log_ind_all(keep_idx);
param_name = param_name_all(keep_idx); %#ok<NASGU>

type_6 = type(keep_idx);
ax_6   = ax(keep_idx);
bx_6   = bx(keep_idx);
ay_6   = ay(keep_idx);
by_6   = by(keep_idx);

%% ===== 讀取/整理 Cs =====
S_Cs = load('Cs_site_Md.mat');
if isfield(S_Cs,'Cs')
    Cs = S_Cs.Cs;
else
    fn = fieldnames(S_Cs);
    Cs = S_Cs.(fn{1});
end

% 如果 Cs 是10維，截取成 keep_idx；如果已經是M維，直接用
if size(Cs,1) == numel(log_ind_all)
    Cs = Cs(keep_idx, keep_idx);
elseif size(Cs,1) ~= M
    error('Cs dimension mismatch: size(Cs,1)=%d, M=%d. 請確認 Cs_site_Md.mat。', size(Cs,1), M);
end

% 標準化成 correlation matrix，並確保正定
D  = diag(1 ./ sqrt(diag(Cs)));
Cs = D * Cs * D;
Cs = (Cs + Cs')/2;
min_eig_Cs = min(eig(Cs));
if min_eig_Cs <= 0
    Cs = Cs + (-min_eig_Cs + 1e-8) * eye(size(Cs));
    fprintf('警告：Cs 非正定，已加入 jitter 修正。\n');
end

[phi_t_Cs, omega_t_Cs] = eig(Cs);
eigvals = diag(omega_t_Cs);
[~, idx_sort] = sort(eigvals, 'descend');
phi_t_Cs = phi_t_Cs(:, idx_sort);
L_Cs_fixed = chol(Cs, 'lower');

%% 找各參數的新 index
idx_LL = find(keep_idx==1, 1);
idx_PI = find(keep_idx==2, 1);
idx_LI = find(keep_idx==3, 1);
idx_sv = find(keep_idx==4, 1);   % sigma'_v / Pa
idx_sp = find(keep_idx==5, 1);   % sigma'_p / Pa
idx_su = find(keep_idx==6, 1);   % su / sigma'_v

if any(cellfun(@isempty,{idx_LL,idx_PI,idx_LI,idx_sv,idx_sp,idx_su}))
    error('keep_idx 必須包含 LL, PI, LI, sv, sp, su，否則無法畫五欄比較圖。');
end

%% ============================================================
%  2. 讀取台北廠址 Excel 資料
%% ============================================================
ttt = xlsread('Taipei_case_table.xlsx');

% 沿用你原本的資料篩選：刪掉偶數列
ttt([2:2:276],:) = [];

z  = ttt(:,1);
z  = z(:);
nz = length(z);

y_site_full       = ttt(:,11:19);
y_site_full(:,10) = nan;        % 補第10個 qtu 欄位，讓 keep_idx 可對應10維順序

y_site   = y_site_full(:, keep_idx);
y_actual = y_site;              % 原始物理空間資料
qt_actual = ttt(:,2); %#ok<NASGU>

% log 轉換
for i = 1:M
    if log_ind(i) == 1
        y_site(:,i) = log(y_site(:,i));
    end
end

% Johnson 轉 normal space
x_site = nan(nz, M);
for i = 1:M
    valid = ~isnan(y_site(:,i));
    x_site(valid,i) = JS_2_normal_gb(y_site(valid,i), ...
        type_6(i), ax_6(i), bx_6(i), ay_6(i), by_6(i));
end

%% ============================================================
%  3. 組 y struct 與距離矩陣
%% ============================================================
param_mean = nan(1, M);
t_mat = zeros(nz, M);
for p = 1:M
    param_mean(p) = nanmean(x_site(:,p));
    t_mat(:,p)    = x_site(:,p) - param_mean(p);
end

y = struct();
y.t = t_mat(:);
y.z = z;
y.X = 0;
y.Y = 0;
X_test = 0;
Y_test = 0;

temp_h = 0;
temp_z = abs(z*ones(1,nz) - (z*ones(1,nz))');
y.temp_z = temp_z;
y.temp_h = temp_h;
y.eig_thresh = 0.999;

% prediction grid = same vertical profile
all_X = [y.X; X_test];
all_Y = [y.Y; Y_test];
all_z = [z; z];
temp_x_ele = abs(all_X - all_X');
temp_y_ele = abs(all_Y - all_Y');
y.temp_h_ele = sqrt(temp_x_ele.^2 + temp_y_ele.^2);
y.temp_z_ele = abs(all_z*ones(1,length(all_z)) - (all_z*ones(1,length(all_z)))');

%% ============================================================
%  4. 兩個模型的 hyperparameter bounds
%% ============================================================
% 7個參數順序：
% x1 -> b = 1/exp(x1)
% x2 -> residual SOF_v
% x3 -> residual SOF_h
% x4 -> residual nu_v, nu_h=nu_v
% x5 -> a = 1/exp(x5)
% x6 -> trend SOF_v
% x7 -> trend SOF_h

z_range = max(temp_z(:));
if z_range <= 0
    z_range = max(z) - min(z);
end
if z_range <= 0
    z_range = 1;
end

% ===== MGPR：trend SOF 由資料估 =====
x_low_mgpr = [-log(10),   log(0.2), log(1),     log(0.3), -log(10),   log(z_range/10), log(1)];
x_up_mgpr  = [-log(0.01), log(10),  log(1.001), log(3),   -log(0.01), log(z_range*10), log(1.001)];

% ===== t-const：把 trend SOF_v 固定在超大值，使 trend kernel 近似常數 =====
% 這等價於 t(z) 幾乎不隨深度變化，只剩 constant trend。
CONST_TREND_SOF_Z = max(1e5, z_range*1e4);
x_low_tconst = [-log(10),   log(0.2), log(1),     log(0.3), -log(10),   log(CONST_TREND_SOF_Z*0.999), log(1)];
x_up_tconst  = [-log(0.01), log(10),  log(1.001), log(3),   -log(0.01), log(CONST_TREND_SOF_Z*1.001), log(1.001)];

%% ============================================================
%  5. 只跑 MGPR
%% ============================================================
fprintf('\n============================================================\n');
fprintf('開始模型：MGPR only\n');
fprintf('============================================================\n');

Results = struct();

Results.MGPR = run_one_model( ...
    'MGPR', y, x_low_mgpr, x_up_mgpr, ...
    T_mcmc, tmcmc_beta, Cs, ...
    param_mean, type_6, ax_6, bx_6, ay_6, by_6, log_ind, ...
    z, M, phi_t_Cs, L_Cs_fixed, X_test, Y_test);

Results.MGPR.Plot = build_plot_result( ...
    Results.MGPR, y_actual, z, keep_idx, Pa);

%% ============================================================
%  6. 指標：MAE / RMSE / RMSE_BV / CI coverage / CI width
%% ============================================================
Metrics_MGPR = calc_metrics_table('MGPR', Results.MGPR.Plot);

TimingEvidence = table( ...
    {'MGPR'}, ...
    Results.MGPR.time_tmcmc_sec, ...
    Results.MGPR.time_crf_sec, ...
    Results.MGPR.logEvidence, ...
    'VariableNames', {'Model','TMCMC_sec','CRF_sec','LogEvidence'});

fprintf('\n================ Timing and model evidence ================\n');
disp(TimingEvidence);

fprintf('\n================ MGPR metrics ================\n');
disp(Metrics_MGPR);

%% ============================================================
%  7. Paper-style MGPR only figure
%% ============================================================
fig_cmp = plot_mgpr_only(Results.MGPR.Plot, PLOT_ONE_CRF);

%% ============================================================
%  8. 儲存輸出
%% ============================================================
if SAVE_OUTPUT
    out_prefix = sprintf('Taipei_MGPR_only_T%d', T_mcmc);

    save([out_prefix '.mat'], 'Results', 'Metrics_MGPR', 'TimingEvidence', ...
        'keep_idx', 'use_param', 'Cs', 'z', 'y_actual', 'T_mcmc', '-v7.3');

    try
        writetable(TimingEvidence, [out_prefix '_metrics.xlsx'], 'Sheet', 'TimingEvidence');
        writetable(Metrics_MGPR,   [out_prefix '_metrics.xlsx'], 'Sheet', 'Metrics');
    catch ME
        warning('writetable to xlsx failed: %s', ME.message);
        writetable(TimingEvidence, [out_prefix '_TimingEvidence.csv']);
        writetable(Metrics_MGPR,   [out_prefix '_Metrics.csv']);
    end

    try
        exportgraphics(fig_cmp, [out_prefix '.png'], 'Resolution', 300);
    catch
        print(fig_cmp, [out_prefix '.png'], '-dpng', '-r300');
    end
    savefig(fig_cmp, [out_prefix '.fig']);

    fprintf('\n已輸出：\n');
    fprintf('  %s.mat\n', out_prefix);
    fprintf('  %s_metrics.xlsx 或 csv\n', out_prefix);
    fprintf('  %s.png / %s.fig\n', out_prefix, out_prefix);
end

fprintf('\n總時間 = %.2f sec = %.2f min\n', toc(tic_total), toc(tic_total)/60);

%% ========================================================================
%  Local functions
%% ========================================================================

function R = run_one_model(model_label, y, x_low, x_up, T_mcmc, tmcmc_beta, Cs, ...
    param_mean, type_6, ax_6, bx_6, ay_6, by_6, log_ind, ...
    z, M, phi_t_Cs, L_Cs_fixed, X_test, Y_test)

    nz = length(z);

    %% ===== TMCMC =====
    fprintf('[%s] 開始 TMCMC...\n', model_label);
    tic_tmcmc = tic;
    [x_mcmc, ln_S, ~, ~, ~] = iTMCMC_fun_mod1('GP_Matern_3D', y, x_low, x_up, T_mcmc, tmcmc_beta, Cs);
    time_tmcmc_sec = toc(tic_tmcmc);
    fprintf('[%s] TMCMC 完成，time = %.2f sec = %.2f min\n', model_label, time_tmcmc_sec, time_tmcmc_sec/60);

    % model evidence：若 ln_S 是每個stage的increment，sum(ln_S) 即 log evidence；若本來是scalar也不影響
    if isempty(ln_S)
        logEvidence = NaN;
    else
        logEvidence = sum(ln_S(:));
    end

    bhp_mcmc    = 1./exp(x_mcmc(:,1));
    sofv_mcmc   = exp(x_mcmc(:,2));
    sofh_mcmc   = exp(x_mcmc(:,3));
    nuv_mcmc    = exp(x_mcmc(:,4));
    nuh_mcmc    = nuv_mcmc;
    ahp_mcmc    = 1./exp(x_mcmc(:,5));
    sofv_t_mcmc = exp(x_mcmc(:,6));
    sofh_t_mcmc = exp(x_mcmc(:,7));

    %% ===== CRF conditional simulation =====
    fprintf('[%s] 開始生成 CRF...\n', model_label);
    tic_crf = tic;

    jitterRh = 1e-6;
    jitterRz = 1e-6;
    jitterP  = 1e-11;

    nh_train = numel(y.X);
    Npost = size(x_mcmc, 1);

    t_ele     = zeros(nz * M, Npost);
    trend_ele = zeros(nz * M, Npost);

    for i = 1:Npost
        [y_phiz, y_phih, y_phiz_ele, y_phih_ele, ~, ln_alpha] = ...
            GP_matrices_Step3(ahp_mcmc(i), sofv_t_mcmc(i), sofh_t_mcmc(i), y, Cs);

        y.phiz     = y_phiz;
        y.phih     = y_phih;
        y.phiz_ele = y_phiz_ele;
        y.phih_ele = y_phih_ele;

        A_diag = exp(ln_alpha(:));

        R_h = Matern_R(nuh_mcmc(i), sofh_mcmc(i), y.temp_h);
        R_z = Matern_R(nuv_mcmc(i), sofv_mcmc(i), y.temp_z);

        Rh = R_h + jitterRh * eye(nh_train);
        Rz = R_z + jitterRz * eye(nz);

        Lh_R = chol(Rh, 'lower');
        Lz_R = chol(Rz, 'lower');

        reshape_data = reshape(y.t, nz, nh_train*M);
        reshape_data = DW_sampler_new2(reshape_data, y.X, y.Y, z, ...
            sofv_mcmc(i), sofh_mcmc(i), nuv_mcmc(i), nuh_mcmc(i), ...
            bhp_mcmc(i), A_diag, y, Cs, sofv_t_mcmc(i), sofh_t_mcmc(i), M);

        reshape_vec = reshape(reshape_data, [], 1);

        AA = (Lh_R' \ (Lh_R \ y_phih)).';
        BB = (Lz_R' \ (Lz_R \ y_phiz)).';
        CC = (L_Cs_fixed' \ (L_Cs_fixed \ phi_t_Cs)).';

        temp_vec = kronmult2({CC, AA, BB}, reshape_vec);
        bhp = bhp_mcmc(i);

        P = spdiags(A_diag, 0, length(A_diag), length(A_diag)) + ...
            (1/bhp) * kron(CC*phi_t_Cs, kron(AA*y_phih, BB*y_phiz));
        P = (P + P')/2 + jitterP * speye(size(P,1));

        try
            Rchol = chol(P, 'lower');
        catch
            Pf = full(P);
            Pf = (Pf + Pf')/2 + jitterP * eye(size(Pf,1));
            Rchol = chol(Pf, 'lower');
        end

        mu = (1/bhp) * (Rchol' \ (Rchol \ temp_vec));
        w  = mu + (Rchol' \ (Rchol \ randn(size(mu))));
        w_row = w.';

        d_ele  = kronmult2({phi_t_Cs, y_phih_ele, y_phiz_ele}, reshape(w_row.', [], M));
        t_diff = reshape_vec - kronmult2({phi_t_Cs, y_phih, y_phiz}, w);

        [E_X, L_h, L_z] = vertical_dense_stats(sofv_mcmc(i), sofh_mcmc(i), ...
            nuv_mcmc(i), nuh_mcmc(i), X_test, Y_test, z, y.X, y.Y, z, t_diff, M);

        noise = E_X + sqrt(bhp) * kronmult2({L_Cs_fixed, L_h, L_z}, randn(nz*M, 1));

        t_ele(:,i)     = d_ele(:) + noise(:);
        trend_ele(:,i) = d_ele(:);
    end

    %% ===== inverse transform to physical space =====
    t_ele_original = inverse_to_original(t_ele, nz, M, param_mean, type_6, ax_6, bx_6, ay_6, by_6, log_ind);
    trend_original = inverse_to_original(trend_ele, nz, M, param_mean, type_6, ax_6, bx_6, ay_6, by_6, log_ind);

    pred_mean   = nan(nz, M);
    pred_median = nan(nz, M);
    pred_p025   = nan(nz, M);
    pred_p975   = nan(nz, M);
    pred_single = nan(nz, M);

    trend_median = nan(nz, M);
    trend_p025   = nan(nz, M);
    trend_p975   = nan(nz, M);

    for p = 1:M
        row_idx = (1:nz) + (p-1)*nz;

        p_samp = t_ele_original(row_idx, :);
        pred_mean(:,p)   = nanmean_row(p_samp);
        pred_median(:,p) = prctile(p_samp, 50,   2);
        pred_p025(:,p)   = prctile(p_samp, 2.5,  2);
        pred_p975(:,p)   = prctile(p_samp, 97.5, 2);
        pred_single(:,p) = p_samp(:,1);

        tr_samp = trend_original(row_idx, :);
        trend_median(:,p) = prctile(tr_samp, 50,   2);
        trend_p025(:,p)   = prctile(tr_samp, 2.5,  2);
        trend_p975(:,p)   = prctile(tr_samp, 97.5, 2);
    end

    time_crf_sec = toc(tic_crf);
    fprintf('[%s] CRF 完成，time = %.2f sec = %.2f min\n', model_label, time_crf_sec, time_crf_sec/60);
    fprintf('[%s] log evidence = %.6g\n', model_label, logEvidence);

    %% ===== store =====
    R = struct();
    R.model_label = model_label;
    R.x_mcmc = x_mcmc;
    R.ln_S = ln_S;
    R.logEvidence = logEvidence;
    R.time_tmcmc_sec = time_tmcmc_sec;
    R.time_crf_sec   = time_crf_sec;

    R.bhp_mcmc = bhp_mcmc;
    R.sofv_mcmc = sofv_mcmc;
    R.sofh_mcmc = sofh_mcmc;
    R.nuv_mcmc = nuv_mcmc;
    R.nuh_mcmc = nuh_mcmc;
    R.ahp_mcmc = ahp_mcmc;
    R.sofv_t_mcmc = sofv_t_mcmc;
    R.sofh_t_mcmc = sofh_t_mcmc;

    R.t_ele = t_ele;
    R.trend_ele = trend_ele;
    R.t_ele_original = t_ele_original;
    R.trend_original = trend_original;

    R.pred_mean = pred_mean;
    R.pred_median = pred_median;
    R.pred_p025 = pred_p025;
    R.pred_p975 = pred_p975;
    R.pred_single = pred_single;

    R.trend_median = trend_median;
    R.trend_p025 = trend_p025;
    R.trend_p975 = trend_p975;
end

function X_original = inverse_to_original(X_normal_centered, nz, M, param_mean, type_6, ax_6, bx_6, ay_6, by_6, log_ind)
    Npost = size(X_normal_centered, 2);
    X_original = zeros(size(X_normal_centered));

    for i = 1:Npost
        X_reshaped = reshape(X_normal_centered(:,i), nz, M);
        for p = 1:M
            data_normal = X_reshaped(:,p) + param_mean(p);
            tmp = JS_2_original_gb(data_normal, type_6(p), ax_6(p), bx_6(p), ay_6(p), by_6(p));
            if log_ind(p) == 1
                X_reshaped(:,p) = exp(tmp);
            else
                X_reshaped(:,p) = tmp;
            end
        end
        X_original(:,i) = X_reshaped(:);
    end
end

function Plot = build_plot_result(R, y_actual, z, keep_idx, Pa)
    nz = length(z);

    idx_LL = find(keep_idx == 1, 1);
    idx_PI = find(keep_idx == 2, 1);
    idx_LI = find(keep_idx == 3, 1);
    idx_sv = find(keep_idx == 4, 1);
    idx_sp = find(keep_idx == 5, 1);
    idx_su = find(keep_idx == 6, 1);

    Plot = struct();
    Plot.depth_m = z(:);

    % ----- direct samples -----
    s_LL = get_param_samples(R.t_ele_original, nz, idx_LL);
    s_PI = get_param_samples(R.t_ele_original, nz, idx_PI);
    s_LI = get_param_samples(R.t_ele_original, nz, idx_LI);
    s_sv = get_param_samples(R.t_ele_original, nz, idx_sv);
    s_sp = get_param_samples(R.t_ele_original, nz, idx_sp);
    s_su_ratio = get_param_samples(R.t_ele_original, nz, idx_su);

    % ----- derived samples -----
    s_sigma_p = s_sp * Pa;
    s_su = s_sv * Pa .* s_su_ratio;

    Plot.samples.LL = s_LL;
    Plot.samples.PI = s_PI;
    Plot.samples.LI = s_LI;
    Plot.samples.sigma_p_eff = s_sigma_p;
    Plot.samples.su = s_su;

    Plot.real.LL = y_actual(:,idx_LL);
    Plot.real.PI = y_actual(:,idx_PI);
    Plot.real.LI = y_actual(:,idx_LI);
    Plot.real.sigma_p_eff = y_actual(:,idx_sp) * Pa;
    Plot.real.su = y_actual(:,idx_sv) * Pa .* y_actual(:,idx_su);

    flds = {'LL','PI','LI','sigma_p_eff','su'};
    for k = 1:numel(flds)
        f = flds{k};
        S = Plot.samples.(f);
        Plot.mean.(f)   = nanmean_row(S);
        Plot.median.(f) = prctile(S, 50,   2);
        Plot.CI_low.(f) = prctile(S, 2.5,  2);
        Plot.CI_up.(f)  = prctile(S, 97.5, 2);
        Plot.one.(f)    = S(:,1);
        Plot.var.(f)    = nanvar_row(S);
    end
end

function S = get_param_samples(X_original, nz, idx_param)
    row_idx = (1:nz) + (idx_param-1)*nz;
    S = X_original(row_idx, :);
end

function T = calc_metrics_table(model_name, Plot)
    fields = {'LL','PI','LI','sigma_p_eff','su'};
    names  = {'LL (%)','PI (%)','LI (%)','sigma''_p (kN/m^2)','s_u (kN/m^2)'};

    n = numel(fields);
    Model = cell(n,1);
    Parameter = cell(n,1);
    N = nan(n,1);
    Bias = nan(n,1);
    MAE = nan(n,1);
    RMSE_point = nan(n,1);
    RMSE_BV = nan(n,1);
    Coverage95_percent = nan(n,1);
    Mean_CI_width = nan(n,1);

    for k = 1:n
        f = fields{k};
        obs = Plot.real.(f)(:);
        med = Plot.median.(f)(:);
        mu  = Plot.mean.(f)(:);
        lo  = Plot.CI_low.(f)(:);
        hi  = Plot.CI_up.(f)(:);
        pv  = Plot.var.(f)(:);

        valid = isfinite(obs) & isfinite(med) & isfinite(mu) & isfinite(lo) & isfinite(hi) & isfinite(pv);

        err_med = med(valid) - obs(valid);
        err_mu  = mu(valid)  - obs(valid);
        pv_v    = pv(valid);

        Model{k} = model_name;
        Parameter{k} = names{k};
        N(k) = sum(valid);

        if N(k) > 0
            Bias(k) = mean(err_med);
            MAE(k) = mean(abs(err_med));
            RMSE_point(k) = sqrt(mean(err_med.^2));

            % 你要的 bias^2 + variance 版本：
            % RMSE_BV = sqrt( mean( (predictive mean - observed)^2 + predictive variance ) )
            RMSE_BV(k) = sqrt(mean(err_mu.^2 + pv_v));

            Coverage95_percent(k) = 100 * mean(obs(valid) >= lo(valid) & obs(valid) <= hi(valid));
            Mean_CI_width(k) = mean(hi(valid) - lo(valid));
        end
    end

    T = table(Model, Parameter, N, Bias, MAE, RMSE_point, RMSE_BV, Coverage95_percent, Mean_CI_width);
end

function fig = plot_mgpr_only(MG, PLOT_ONE_CRF)
    field_name = {'LL','PI','LI','sigma_p_eff','su'};

    x_label = { ...
        'LL (%)', ...
        'PI (%)', ...
        'LI (%)', ...
        '\sigma''_p (kN/m^2)', ...
        's_u (kN/m^2)'};

    x_lim = { ...
        [20 50], ...
        [0 30], ...
        [0 2.5], ...
        [1e1 1e3], ...
        [20 120]};

    use_log_x = [false false false true false];
    panel_label = {'(a)','(b)','(c)','(d)','(e)'};

    col_mgpr = [1.00 0.00 1.00];
    col_crf  = [0.00 0.65 0.00];
    col_data = [1.00 0.90 0.00];

    lw_med = 1.6;
    lw_CI  = 1.2;
    lw_crf = 1.0;

    fig = figure('Name','MGPR only', 'Color','w', 'Position',[80 80 1180 500]);

    left_margin = 0.055;
    right_blank = 0.205;
    bottom_pos  = 0.16;
    height_pos  = 0.76;
    gap         = 0.040;
    n_col       = 5;

    tile_width = (1 - left_margin - right_blank - gap*(n_col-1)) / n_col;
    depth = MG.depth_m(:);

    for k = 1:n_col
        fname = field_name{k};
        left_pos = left_margin + (k-1)*(tile_width + gap);

        ax = axes('Position',[left_pos bottom_pos tile_width height_pos]); %#ok<LAXES>
        hold(ax,'on'); box(ax,'on');

        if use_log_x(k)
            plot_profile_log(ax, MG.median.(fname), depth, '-',  col_mgpr, lw_med);
            plot_profile_log(ax, MG.CI_low.(fname), depth, '--', col_mgpr, lw_CI);
            plot_profile_log(ax, MG.CI_up.(fname),  depth, '--', col_mgpr, lw_CI);

            if PLOT_ONE_CRF
                plot_profile_log(ax, MG.one.(fname), depth, '-', col_crf, lw_crf);
            end

            x_real = MG.real.(fname);
            valid_real = isfinite(x_real) & isfinite(depth) & x_real > 0;
            semilogx(ax, x_real(valid_real), depth(valid_real), 'o', ...
                'MarkerSize', 5.5, ...
                'MarkerFaceColor', col_data, ...
                'MarkerEdgeColor','k', ...
                'LineWidth',0.8, ...
                'LineStyle','none');

            set(ax,'XScale','log');
        else
            plot(ax, MG.median.(fname), depth, '-',  'Color', col_mgpr, 'LineWidth', lw_med);
            plot(ax, MG.CI_low.(fname), depth, '--', 'Color', col_mgpr, 'LineWidth', lw_CI);
            plot(ax, MG.CI_up.(fname),  depth, '--', 'Color', col_mgpr, 'LineWidth', lw_CI);

            if PLOT_ONE_CRF
                plot(ax, MG.one.(fname), depth, '-', 'Color', col_crf, 'LineWidth', lw_crf);
            end

            x_real = MG.real.(fname);
            valid_real = isfinite(x_real) & isfinite(depth);
            plot(ax, x_real(valid_real), depth(valid_real), 'o', ...
                'MarkerSize', 5.5, ...
                'MarkerFaceColor', col_data, ...
                'MarkerEdgeColor','k', ...
                'LineWidth',0.8, ...
                'LineStyle','none');
        end

        set(ax,'YDir','reverse');
        ylim(ax, [12 27]);
        xlim(ax, x_lim{k});

        xlabel(ax, x_label{k}, 'FontSize', 13);
        ylabel(ax, 'Depth (m)', 'FontSize', 12);

        set(ax, ...
            'FontSize', 10, ...
            'LineWidth', 0.9, ...
            'TickDir','in', ...
            'Box','on', ...
            'Layer','top');

        grid(ax,'on');
        ax.GridAlpha = 0.18;

        text(ax, 0.78, 0.06, panel_label{k}, ...
            'Units','normalized', 'FontSize',11, 'FontWeight','bold');
    end

    legend_ax = axes('Position',[0.815 0.62 0.15 0.20], 'Visible','off'); %#ok<LAXES>
    hold(legend_ax, 'on');

    h1 = plot(legend_ax, nan, nan, '-',  'Color', col_mgpr, 'LineWidth', lw_med);
    h2 = plot(legend_ax, nan, nan, '--', 'Color', col_mgpr, 'LineWidth', lw_CI);

    if PLOT_ONE_CRF
        h3 = plot(legend_ax, nan, nan, '-',  'Color', col_crf, 'LineWidth', lw_crf);
        h4 = plot(legend_ax, nan, nan, 'o', ...
            'MarkerFaceColor', col_data, ...
            'MarkerEdgeColor','k', ...
            'MarkerSize',5, ...
            'LineStyle','none');

        lgd = legend(legend_ax, [h1 h2 h3 h4], ...
            {'Median (MGPR)', ...
             '95% CI (MGPR)', ...
             'One CRF (MGPR)', ...
             'Observed data'}, ...
            'Location','northwest', 'FontSize',8, 'Box','on');
    else
        h3 = plot(legend_ax, nan, nan, 'o', ...
            'MarkerFaceColor', col_data, ...
            'MarkerEdgeColor','k', ...
            'MarkerSize',5, ...
            'LineStyle','none');

        lgd = legend(legend_ax, [h1 h2 h3], ...
            {'Median (MGPR)', ...
             '95% CI (MGPR)', ...
             'Observed data'}, ...
            'Location','northwest', 'FontSize',8, 'Box','on');
    end

    lgd.ItemTokenSize = [14 7];
    lgd.AutoUpdate = 'off';
end

function fig = plot_tconst_vs_mgpr(TC, MG, PLOT_ONE_CRF)
    field_name = {'LL','PI','LI','sigma_p_eff','su'};

    x_label = { ...
        'LL (%)', ...
        'PI (%)', ...
        'LI (%)', ...
        '\sigma''_p (kN/m^2)', ...
        's_u (kN/m^2)'};

    x_lim = { ...
        [20 50], ...
        [0 30], ...
        [0 2.5], ...
        [1e1 1e3], ...
        [20 120]};

    use_log_x = [false false false true false];
    panel_label = {'(a)','(b)','(c)','(d)','(e)'};

    col_tconst = [0.45 0.45 0.45];
    col_mgpr   = [1.00 0.00 1.00];
    col_crf    = [0.00 0.65 0.00];
    col_data   = [1.00 0.90 0.00];

    lw_med = 1.6;
    lw_CI  = 1.2;
    lw_crf = 1.0;

    fig = figure('Name','t-const vs MGPR', 'Color','w', 'Position',[80 80 1280 500]);

    left_margin = 0.055;
    right_blank = 0.225;   % legend 區域，已縮小，不像原本那麼寬
    bottom_pos  = 0.16;
    height_pos  = 0.76;
    gap         = 0.040;
    n_col       = 5;

    tile_width = (1 - left_margin - right_blank - gap*(n_col-1)) / n_col;

    for k = 1:n_col
        fname = field_name{k};
        left_pos = left_margin + (k-1)*(tile_width + gap);

        ax = axes('Position',[left_pos bottom_pos tile_width height_pos]); %#ok<LAXES>
        hold(ax,'on'); box(ax,'on');

        depth = TC.depth_m(:);

        if use_log_x(k)
            % ----- t-const -----
            plot_profile_log(ax, TC.median.(fname), depth, '-',  col_tconst, lw_med);
            plot_profile_log(ax, TC.CI_low.(fname), depth, '--', col_tconst, lw_CI);
            plot_profile_log(ax, TC.CI_up.(fname),  depth, '--', col_tconst, lw_CI);

            % ----- MGPR -----
            plot_profile_log(ax, MG.median.(fname), depth, '-',  col_mgpr, lw_med);
            plot_profile_log(ax, MG.CI_low.(fname), depth, '--', col_mgpr, lw_CI);
            plot_profile_log(ax, MG.CI_up.(fname),  depth, '--', col_mgpr, lw_CI);

            % ----- one MGPR CRF realization -----
            if PLOT_ONE_CRF
                plot_profile_log(ax, MG.one.(fname), depth, '-', col_crf, lw_crf);
            end

            % ----- observed -----
            x_real = MG.real.(fname);
            valid_real = isfinite(x_real) & x_real > 0;
            semilogx(ax, x_real(valid_real), depth(valid_real), 'o', ...
                'MarkerSize', 5.5, 'MarkerFaceColor', col_data, ...
                'MarkerEdgeColor','k', 'LineWidth',0.8, 'LineStyle','none');

            set(ax,'XScale','log');
        else
            % ----- t-const -----
            plot(ax, TC.median.(fname), depth, '-',  'Color', col_tconst, 'LineWidth', lw_med);
            plot(ax, TC.CI_low.(fname), depth, '--', 'Color', col_tconst, 'LineWidth', lw_CI);
            plot(ax, TC.CI_up.(fname),  depth, '--', 'Color', col_tconst, 'LineWidth', lw_CI);

            % ----- MGPR -----
            plot(ax, MG.median.(fname), depth, '-',  'Color', col_mgpr, 'LineWidth', lw_med);
            plot(ax, MG.CI_low.(fname), depth, '--', 'Color', col_mgpr, 'LineWidth', lw_CI);
            plot(ax, MG.CI_up.(fname),  depth, '--', 'Color', col_mgpr, 'LineWidth', lw_CI);

            % ----- one MGPR CRF realization -----
            if PLOT_ONE_CRF
                plot(ax, MG.one.(fname), depth, '-', 'Color', col_crf, 'LineWidth', lw_crf);
            end

            % ----- observed -----
            x_real = MG.real.(fname);
            valid_real = isfinite(x_real);
            plot(ax, x_real(valid_real), depth(valid_real), 'o', ...
                'MarkerSize', 5.5, 'MarkerFaceColor', col_data, ...
                'MarkerEdgeColor','k', 'LineWidth',0.8, 'LineStyle','none');
        end

        set(ax,'YDir','reverse');
        ylim(ax, [12 27]);
        xlim(ax, x_lim{k});

        xlabel(ax, x_label{k}, 'FontSize', 13);
        ylabel(ax, 'Depth (m)', 'FontSize', 12);

        set(ax, ...
            'FontSize', 10, ...
            'LineWidth', 0.9, ...
            'TickDir','in', ...
            'Box','on', ...
            'Layer','top');

        grid(ax,'on');
        ax.GridAlpha = 0.18;

        text(ax, 0.78, 0.06, panel_label{k}, ...
            'Units','normalized', 'FontSize',11, 'FontWeight','bold');
    end

    % ===== compact legend：右側小盒子 =====
    legend_ax = axes('Position',[0.805 0.60 0.155 0.24], 'Visible','off'); %#ok<LAXES>
    hold(legend_ax, 'on');

    h1 = plot(legend_ax, nan, nan, '-',  'Color', col_tconst, 'LineWidth', lw_med);
    h2 = plot(legend_ax, nan, nan, '--', 'Color', col_tconst, 'LineWidth', lw_CI);
    h3 = plot(legend_ax, nan, nan, '-',  'Color', col_mgpr,   'LineWidth', lw_med);
    h4 = plot(legend_ax, nan, nan, '--', 'Color', col_mgpr,   'LineWidth', lw_CI);

    if PLOT_ONE_CRF
        h5 = plot(legend_ax, nan, nan, '-',  'Color', col_crf, 'LineWidth', lw_crf);
        h6 = plot(legend_ax, nan, nan, 'o', 'MarkerFaceColor', col_data, ...
            'MarkerEdgeColor','k', 'MarkerSize',5, 'LineStyle','none');

        lgd = legend(legend_ax, [h1 h2 h3 h4 h5 h6], ...
            {'Median (t-const)', ...
             '95% CI (t-const)', ...
             'Median (MGPR)', ...
             '95% CI (MGPR)', ...
             'One CRF (MGPR)', ...
             'Observed data'}, ...
            'Location','northwest', 'FontSize',8, 'Box','on');
    else
        h5 = plot(legend_ax, nan, nan, 'o', 'MarkerFaceColor', col_data, ...
            'MarkerEdgeColor','k', 'MarkerSize',5, 'LineStyle','none');

        lgd = legend(legend_ax, [h1 h2 h3 h4 h5], ...
            {'Median (t-const)', ...
             '95% CI (t-const)', ...
             'Median (multi t-GPR)', ...
             '95% CI (multi t-GPR)', ...
             'Observed data'}, ...
            'Location','northwest', 'FontSize',8, 'Box','on');
    end

    lgd.ItemTokenSize = [14 7];
    lgd.AutoUpdate = 'off';
end

function plot_profile_log(ax, x, z, lineStyle, color, lw)
    x = x(:);
    z = z(:);
    valid = isfinite(x) & isfinite(z) & x > 0;
    semilogx(ax, x(valid), z(valid), lineStyle, 'Color', color, 'LineWidth', lw);
end

function m = nanmean_row(A)
    m = nan(size(A,1),1);
    for i = 1:size(A,1)
        x = A(i,:);
        x = x(isfinite(x));
        if ~isempty(x)
            m(i) = mean(x);
        end
    end
end

function v = nanvar_row(A)
    v = nan(size(A,1),1);
    for i = 1:size(A,1)
        x = A(i,:);
        x = x(isfinite(x));
        if numel(x) >= 2
            v(i) = var(x, 0);
        elseif numel(x) == 1
            v(i) = 0;
        end
    end
end
