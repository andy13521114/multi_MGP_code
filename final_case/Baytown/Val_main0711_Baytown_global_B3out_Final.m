clear all; clc; close all;
tic_total_all = tic;
rand('state',1); randn('state',1);
%% ===== 載入 Johnson 參數 與 基礎設定 =====
load('HBM_CLAY_8_OCR_sigv_new.mat');
load('Cs_global.mat');
%load('Cs.mat');
load('integrated_geotechnical_data_0521.mat');

Scenario = 2;
T_mcmc   =3;

%% ===== SAVE CONTROL =====
% 0 = do not save; 1 = save only data .mat and result .fig
DO_SAVE_OUTPUT = 1;
SAVE_DATA_MAT  = DO_SAVE_OUTPUT;
SAVE_RESULT_FIG = DO_SAVE_OUTPUT;

% Output files are saved in the current MATLAB folder (pwd)
output_dir = pwd;
output_mat_name = fullfile(output_dir, 'GPR_MUSIC_3X_B3out_const_timing_metrics.mat');
fig_prefix = 'GPR_vs_HBM_const';

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

Cs = Cs_global;
Cs = Cs(para_Cs, para_Cs);
Cs = (Cs + Cs') / 2;
M  = size(Cs, 1);

%% ===== 載入資料並前處理 =====
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

%% ===== 過濾 CPT-3 Bq < -0.07 =====
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
        fprintf('CPT-3 過濾：共 %d 個深度點 Bq < 0.07，已設為 NaN\n', sum(mask_bad));
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
        fprintf('CPT-3 過濾 (Scenario 2)：共 %d 個深度點\n', sum(mask_bad));
    end
end

%% ===== σ'v 矩陣 =====
sigvp_all = data_structure.sigvp(dep_range, :);

%% ===== 訓練孔與測試孔 =====
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
          % 6.401  1.372;
           6.401  19.202;
           6.401  26.974];

E_test = [6.401  1.372];

tol = 1e-3;

train_idx = zeros(1, size(E_train,1));
for k = 1:size(E_train,1)
    dist = sqrt((X_all - E_train(k,1)).^2 + (Y_all - E_train(k,2)).^2);
    [d_min, i_min] = min(dist);
    assert(d_min < tol, '找不到訓練孔座標 [%.3f, %.3f]！', E_train(k,1), E_train(k,2));
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
        fprintf('測試孔 %d: [%.3f, %.3f] → %s（有真實資料）\n', k, E_test(k,1), E_test(k,2), location_labels{i_min});
    else
        test_label_list{k} = sprintf('(%.3f,%.3f)', E_test(k,1), E_test(k,2));
        fprintf('測試孔 %d: [%.3f, %.3f] → 純預測點\n', k, E_test(k,1), E_test(k,2));
    end
end

nh_train = length(train_idx);
fprintf('訓練孔 (%d 孔): %s\n', nh_train, strjoin(location_labels(train_idx), ', '));

%% ===== 為每個測試孔準備 σ'v =====
sigvp_test = nan(nz, nh_test);
for k = 1:nh_test
    if ~isnan(test_idx(k))
        sigvp_test(:,k) = sigvp_all(:, test_idx(k));
    else
        sigvp_test(:,k) = nanmean(sigvp_all(:, train_idx), 2);
    end
end

%% ===== 準備預測矩陣 =====
pred_median = nan(nz, nh_test, M);
pred_p025   = nan(nz, nh_test, M);
pred_p975   = nan(nz, nh_test, M);
pred_single = nan(nz, nh_test, M);
pred_samples = nan(nz, nh_test, M, T_mcmc);   % Store all posterior samples for metrics
true_data   = nan(nz, nh_test, M);

for k = 1:nh_test
    if ~isnan(test_idx(k))
        for pm = 1:M
            true_data(:, k, pm) = param_list_original_all{pm}(:, test_idx(k));
        end
    end
end

%% ===== 訓練資料整理 =====
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
   % t(:, p) = tvec ;
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

x_low = [-log(10), log(0.01), log(0.1),  log(0.1), -log(10),  log(max(temp_z(:))/10), log(max(temp_h(:))/10)];
x_up  = [-log(0.01),log(100), log(100), log(3.0), -log(0.01), log(max(temp_z(:))*10), log(max(temp_h(:))*10)];
% x_low = [-log(10), log(0.01), log(0.1),  log(0.1), -log(10),  log(99999), log(max(temp_h(:))/10)];
% x_up  = [-log(0.01),log(100), log(100), log(3.0), -log(0.0001), log(100000), log(max(temp_h(:))*10)];

y.eig_thresh = 0.999;

temp_x_ele = abs([X; X_test] - [X; X_test]');
temp_y_ele = abs([Y; Y_test] - [Y; Y_test]');
temp_z_ele = abs([z; z] - [z; z]');
temp_h_ele = sqrt(temp_x_ele.^2 + temp_y_ele.^2);
y.temp_z_ele = temp_z_ele; y.temp_h_ele = temp_h_ele;

fprintf('\n開始 TMCMC...\n');
%load('TMCMC_B3out_const_0611.mat')
tic_tmcmc = tic;
[x_mcmc, ln_S, ~, ~, ~] = iTMCMC_fun_mod1('BaytownGP_Matern_3D', y, x_low, x_up, T_mcmc, 0.5, Cs);
time_TMCMC_sec = toc(tic_tmcmc);
bhp_mcmc    = 1./exp(x_mcmc(:,1));
sofv_mcmc   = exp(x_mcmc(:,2));
sofh_mcmc   = exp(x_mcmc(:,3));
nuv_mcmc    = exp(x_mcmc(:,4));
nuh_mcmc    = nuv_mcmc;
ahp_mcmc    = 1./exp(x_mcmc(:,5));
sofv_t_mcmc = exp(x_mcmc(:,6));
sofh_t_mcmc = exp(x_mcmc(:,7));
fprintf('TMCMC 完成。\n');
fprintf('[TIME] TMCMC elapsed = %.2f sec = %.2f min\n', time_TMCMC_sec, time_TMCMC_sec/60);
%%
if SAVE_DATA_MAT == 1
save('TMCMC_B3out_const_0611.mat', ...
    'bhp_mcmc', 'sofv_mcmc', 'sofh_mcmc', 'nuv_mcmc', 'nuh_mcmc', ...
    'ahp_mcmc', 'sofv_t_mcmc', 'sofh_t_mcmc', ...
    'x_mcmc', 'ln_S', ...
    'x_low', 'x_up', 'T_mcmc', 'Scenario', 'para_Cs', 'param_name');
fprintf('已存出 TMCMC_Bt.mat\n');
end
%% ===== 參數分布圖 =====
figure('Name', 'GP 參數後驗分布', 'Position', [160, 100, 1400, 800]);
subplot(2,4,1); loglog(nuv_mcmc, sofv_mcmc, 's', 'Color', [0.5 0.5 0.5], 'MarkerSize', 4);
xlabel('\nu_z','FontSize',12,'FontWeight','bold'); ylabel('\delta_z (m)','FontSize',12,'FontWeight','bold');
title('垂直相關參數'); xlim([0.01 10]); ylim([0.1 20]); grid on; box on;
subplot(2,4,2); loglog(nuh_mcmc, sofh_mcmc, 's', 'Color', [0.5 0.5 0.5], 'MarkerSize', 4);
xlabel('\nu_h','FontSize',12,'FontWeight','bold'); ylabel('\delta_h (m)','FontSize',12,'FontWeight','bold');
title('水平相關參數'); xlim([0.01 10]); ylim([0.1 20]); grid on; box on;
subplot(2,4,3); histogram(bhp_mcmc, 30, 'FaceColor', [0.3 0.5 0.8], 'EdgeColor', 'k');
xlabel('b','FontSize',12,'FontWeight','bold'); ylabel('Frequency'); title('b(Residual)'); grid on; box on;
subplot(2,4,4); histogram(ahp_mcmc, 30, 'FaceColor', [0.8 0.5 0.3], 'EdgeColor', 'k');
xlabel('a','FontSize',12,'FontWeight','bold'); ylabel('Frequency'); title('a(Trend)'); grid on; box on;
subplot(2,4,5); histogram(sofv_mcmc, 30, 'FaceColor', [0.5 0.8 0.5], 'EdgeColor', 'k');
xlabel('\delta_z (m)','FontSize',12,'FontWeight','bold'); ylabel('Frequency'); title('垂直 SOF'); grid on; box on;
subplot(2,4,6); histogram(sofh_mcmc, 30, 'FaceColor', [0.8 0.5 0.8], 'EdgeColor', 'k');
xlabel('\delta_h (m)','FontSize',12,'FontWeight','bold'); ylabel('Frequency'); title('水平 SOF'); grid on; box on;
subplot(2,4,7); histogram(sofv_t_mcmc, 30, 'FaceColor', [0.5 0.5 0.8], 'EdgeColor', 'k');
xlabel('\delta_{z,t} (m)','FontSize',12,'FontWeight','bold'); ylabel('Frequency'); title('Trend \delta_z'); grid on; box on;
subplot(2,4,8); histogram(sofh_t_mcmc, 30, 'FaceColor', [0.8 0.8 0.5], 'EdgeColor', 'k');
xlabel('\delta_{h,t} (m)','FontSize',12,'FontWeight','bold'); ylabel('Frequency'); title('Trend \delta_h'); grid on; box on;
sgtitle(sprintf('GP 參數後驗分布 (Scenario %d, N=%d)', Scenario, T_mcmc), 'FontSize', 14, 'FontWeight', 'bold');

% %% ===== Conditional Simulation + 同步收集 trend 的 w =====
% % ─── w_store: 存每個 MCMC 樣本的 w，供後面畫 trend 直接使用 ───
% % 大小：(n_w × T_mcmc)，n_w = M × n_phi_h × n_phi_z（由第一次迴圈決定）
% % w 維度因 eig_thresh 截斷數不同而每次可能不同，改用 cell array
% w_store           = cell(T_mcmc, 1);
% phiz_store        = cell(T_mcmc, 1);
% phih_store        = cell(T_mcmc, 1);
% reshapedata_store = cell(T_mcmc, 1);
% 
% fprintf('\n開始 Conditional Simulation（共 %d 個測試孔）...\n', nh_test);
% jitterRh = 1e-6; jitterRz = 1e-6; jitterP = 1e-11;
% z_ele = z;
% 
% for ti = 1:nh_test
%     X_ele = X_test(ti);
%     Y_ele = Y_test(ti);
%     fprintf('  預測測試孔 %s (%d/%d)...\n', test_label_list{ti}, ti, nh_test);
% 
%     temp_x_ele_ti = abs([X; X_ele] - [X; X_ele]');
%     temp_y_ele_ti = abs([Y; Y_ele] - [Y; Y_ele]');
%     temp_z_ele_ti = abs([z; z_ele] - [z; z_ele]');
%     temp_h_ele_ti = sqrt(temp_x_ele_ti.^2 + temp_y_ele_ti.^2);
%     y.temp_z_ele  = temp_z_ele_ti;
%     y.temp_h_ele  = temp_h_ele_ti;
% 
%     t_ele = zeros(length(z_ele) * M, T_mcmc);
% 
%     for i = 1:T_mcmc
%         [y_phiz, y_phih, y_phiz_ele, y_phih_ele, ~, ln_alpha] = ...
%             GP_matrices_Step3(ahp_mcmc(i), sofv_t_mcmc(i), sofh_t_mcmc(i), y, Cs);
%         y.phiz = y_phiz; y.phih = y_phih;
%         y.phiz_ele = y_phiz_ele; y.phih_ele = y_phih_ele;
% 
%         A_diag = exp(ln_alpha(:));
%         R_h = Matern_R(nuh_mcmc(i), sofh_mcmc(i), temp_h);
%         R_z = Matern_R(nuv_mcmc(i), sofv_mcmc(i), temp_z);
%         Rh  = R_h + jitterRh * eye(nh_train);
%         Rz  = R_z + jitterRz * eye(nz);
% 
%         Lh_R = chol(Rh,'lower'); Lz_R = chol(Rz,'lower');
%         reshape_data = reshape(y.t, nz, nh_train*M);
% 
%         if Scenario == 2
%             reshape_data = DW_sampler_new3(reshape_data, X, Y, z, ...
%                 sofv_mcmc(i), sofh_mcmc(i), nuv_mcmc(i), nuh_mcmc(i), ...
%                 bhp_mcmc(i), A_diag, y, Cs, sofv_t_mcmc(i), sofh_t_mcmc(i), M);
%         end
% 
%         reshape_vec = reshape(reshape_data, [], 1);
%         AA = (Lh_R' \ (Lh_R \ y_phih)).';
%         BB = (Lz_R' \ (Lz_R \ y_phiz)).';
%         CC = (L_Cs_fixed' \ (L_Cs_fixed \ phi_t_Cs)).';
%         temp_vec = kronmult2({CC, AA, BB}, reshape_vec);
% 
%         bhp = bhp_mcmc(i);
%         P   = spdiags(A_diag, 0, length(A_diag), length(A_diag)) + ...
%               (1/bhp) * kron(CC*phi_t_Cs, kron(AA*y_phih, BB*y_phiz));
%         P   = (P+P')/2 + jitterP * speye(size(P,1));
% 
%         try
%             R_chol = chol(P,'lower');
%         catch
%             Pf = full(P); Pf = (Pf+Pf')/2 + jitterP*eye(size(Pf,1));
%             R_chol = chol(Pf,'lower');
%         end
% 
%         mu = (1/bhp) * (R_chol' \ (R_chol \ temp_vec));
%         w  = mu + (R_chol' \ (R_chol \ randn(size(mu))));
% 
%         % ─── 存下 w 與對應的 phi（給 trend 段落用）───
%         w_store{i}              = w;   % cell：每次 w 維度可不同
%         phiz_store{i}           = y_phiz;
%         phih_store{i}           = y_phih;
%         reshapedata_store{i}    = reshape_data;   % DW 補全後的資料
% 
%         w_row = w.';
%         dd_temp = kronmult2({phi_t_Cs, y_phih, y_phiz}, w);
%         t_diff  = reshape_vec - dd_temp;
% 
%         [E_X, L_h, L_z] = vertical_dense_stats(sofv_mcmc(i), sofh_mcmc(i), ...
%             nuv_mcmc(i), nuh_mcmc(i), X_ele, Y_ele, z_ele, X, Y, z, t_diff, M);
% 
%         noise = E_X + sqrt(bhp) * kronmult2({L_Cs_fixed, L_h, L_z}, ...
%             randn(length(z_ele)*M, 1));
%         d_ele = kronmult2({phi_t_Cs, y_phih_ele, y_phiz_ele}, reshape(w_row.', [], M));
% 
%         t_ele(:,i) = d_ele(:) + noise(:);
%     end
% 
%     % 轉回原始尺度
%     t_ele_original = zeros(size(t_ele));
%     for i = 1:T_mcmc
%         t_ele_reshaped = reshape(t_ele(:,i), nz, M);
%         for p = 1:M
%             param_idx_p = para_Cs(p);
%             data_normal = t_ele_reshaped(:,p) + param_mean(p);
%             tmp = JS_2_original(data_normal, type(param_idx_p), ax(param_idx_p), ...
%                 bx(param_idx_p), ay(param_idx_p), by(param_idx_p));
%             if log_ind(p) > 0.5
%                 t_ele_reshaped(:,p) = exp(tmp);
%             else
%                 t_ele_reshaped(:,p) = tmp;
%             end
%         end
%         t_ele_original(:,i) = t_ele_reshaped(:);
%     end
% 
%     for p = 1:M
%         row_idx = (1:nz) + (p-1)*nz;
%         p_samp  = real(t_ele_original(row_idx, :));
%         pred_median(:, ti, p) = prctile(p_samp, 50,   2);
%         pred_p025(:,   ti, p) = prctile(p_samp, 2.5,  2);
%         pred_p975(:,   ti, p) = prctile(p_samp, 97.5, 2);
%         pred_single(:, ti, p) = p_samp(:, 1);
%     end
% end
% fprintf('\n===== Conditional Simulation 完成 =====\n');
%% 0611平行運算

% ===== Conditional Simulation + 平行運算 + 同步收集 trend 的 w =====

tic_crf_total = tic;
crf_hole_time_sec = nan(nh_test,1);

w_store           = cell(nh_test, T_mcmc);
phiz_store        = cell(nh_test, T_mcmc);
phih_store        = cell(nh_test, T_mcmc);
reshapedata_store = cell(nh_test, T_mcmc);

fprintf('\n開始 Conditional Simulation（共 %d 個測試孔）...\n', nh_test);

jitterRh = 1e-6;
jitterRz = 1e-6;
jitterP  = 1e-11;

z_ele = z;

% 你前一次單次 DW 的參考時間，之後可改成新的 benchmark
serial_DW_ref = 11.73;   % sec, 若不想比較可設成 NaN

% 建議：先控制 worker 數，不一定越多越快
% if isempty(gcp('nocreate'))
%     parpool('Processes', 8);
% end

poolobj = gcp('nocreate');
if isempty(poolobj)
    nWorkers = 0;
else
    nWorkers = poolobj.NumWorkers;
end

fprintf('目前 parpool workers = %d\n', nWorkers);

for ti = 1:nh_test

    X_ele = X_test(ti);
    Y_ele = Y_test(ti);

    fprintf('\n============================================================\n');
    fprintf('  預測測試孔 %s (%d/%d)\n', test_label_list{ti}, ti, nh_test);
    fprintf('============================================================\n');

    %% ------------------------------------------------------------
    %  建立此測試孔對應的 extended distance matrix
    %% ------------------------------------------------------------
    temp_x_ele_ti = abs([X; X_ele] - [X; X_ele]');
    temp_y_ele_ti = abs([Y; Y_ele] - [Y; Y_ele]');
    temp_z_ele_ti = abs([z; z_ele] - [z; z_ele]');
    temp_h_ele_ti = sqrt(temp_x_ele_ti.^2 + temp_y_ele_ti.^2);

    y_ti = y;
    y_ti.temp_z_ele = temp_z_ele_ti;
    y_ti.temp_h_ele = temp_h_ele_ti;

    %% ------------------------------------------------------------
    %  預先配置此測試孔的結果
    %% ------------------------------------------------------------
    t_ele = zeros(length(z_ele) * M, T_mcmc);

    w_store_ti           = cell(T_mcmc, 1);
    phiz_store_ti        = cell(T_mcmc, 1);
    phih_store_ti        = cell(T_mcmc, 1);
    reshapedata_store_ti = cell(T_mcmc, 1);

    % timing arrays
    dw_time_vec   = zeros(T_mcmc, 1);
    iter_time_vec = zeros(T_mcmc, 1);

    %% ------------------------------------------------------------
    %  parfor progress monitor
    %% ------------------------------------------------------------
    printEvery = 20;  % 每完成 20 個 sample 印一次，可改 10/50/100

    dq = parallel.pool.DataQueue;

    progress_update(struct( ...
        'cmd',        'reset', ...
        'total',      T_mcmc, ...
        'printEvery', printEvery, ...
        'label',      test_label_list{ti}, ...
        'nWorkers',   nWorkers, ...
        'serialDW',   serial_DW_ref));

    afterEach(dq, @progress_update);

    tic_ti_wall = tic;

    fprintf('  開始 parfor：T_mcmc = %d, workers = %d\n', T_mcmc, nWorkers);

    %% ------------------------------------------------------------
    %  MCMC samples 平行運算
    %% ------------------------------------------------------------
    parfor i = 1:T_mcmc

        tic_iter = tic;
        dw_time_i = 0;

        %% -----------------------------
        %  每個 worker 使用自己的 y_local
        %% -----------------------------
        y_local = y_ti;

        [y_phiz, y_phih, y_phiz_ele, y_phih_ele, ~, ln_alpha] = ...
            GP_matrices_Step3(ahp_mcmc(i), sofv_t_mcmc(i), sofh_t_mcmc(i), y_local, Cs);

        y_local.phiz     = y_phiz;
        y_local.phih     = y_phih;
        y_local.phiz_ele = y_phiz_ele;
        y_local.phih_ele = y_phih_ele;

        A_diag = exp(ln_alpha(:));

        %% -----------------------------
        %  residual covariance
        %% -----------------------------
        R_h = Matern_R(nuh_mcmc(i), sofh_mcmc(i), temp_h);
        R_z = Matern_R(nuv_mcmc(i), sofv_mcmc(i), temp_z);

        Rh = R_h + jitterRh * eye(nh_train);
        Rz = R_z + jitterRz * eye(nz);

        Lh_R = chol(Rh, 'lower');
        Lz_R = chol(Rz, 'lower');

        %% -----------------------------
        %  training data reshape
        %% -----------------------------
        reshape_data = reshape(y_local.t, nz, nh_train * M);

        %% -----------------------------
        %  DW conditional simulation
        %% -----------------------------
        if Scenario == 2

            tic_dw = tic;

            reshape_data = DW_sampler_new2(reshape_data, X, Y, z, ...
                sofv_mcmc(i), sofh_mcmc(i), nuv_mcmc(i), nuh_mcmc(i), ...
                bhp_mcmc(i), A_diag, y_local, Cs, ...
                sofv_t_mcmc(i), sofh_t_mcmc(i), M);

            dw_time_i = toc(tic_dw);
        end

        reshape_vec = reshape(reshape_data, [], 1);

        %% -----------------------------
        %  Posterior of trend weights w
        %% -----------------------------
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

        % parfor 版 randn 順序和 serial 不會 bit-by-bit 一樣
        w = mu + (R_chol' \ (R_chol \ randn(size(mu))));

        %% -----------------------------
        %  residual part at test location
        %% -----------------------------
        w_row = w.';

        dd_temp = kronmult2({phi_t_Cs, y_phih, y_phiz}, w);
        t_diff  = reshape_vec - dd_temp;

        [E_X, L_h, L_z] = vertical_dense_stats( ...
            sofv_mcmc(i), sofh_mcmc(i), ...
            nuv_mcmc(i), nuh_mcmc(i), ...
            X_ele, Y_ele, z_ele, X, Y, z, t_diff, M);

        noise = E_X + sqrt(bhp) * kronmult2({L_Cs_fixed, L_h, L_z}, ...
            randn(length(z_ele) * M, 1));

        %% -----------------------------
        %  trend part at test location
        %% -----------------------------
        d_ele = kronmult2({phi_t_Cs, y_phih_ele, y_phiz_ele}, reshape(w_row.', [], M));

        %% -----------------------------
        %  final simulated field at test location
        %% -----------------------------
        t_ele(:, i) = d_ele(:) + noise(:);

        %% -----------------------------
        %  store for trend plotting
        %% -----------------------------
        w_store_ti{i}           = w;
        phiz_store_ti{i}        = y_phiz;
        phih_store_ti{i}        = y_phih;
        reshapedata_store_ti{i} = reshape_data;

        %% -----------------------------
        %  timing + progress
        %% -----------------------------
        iter_time_i = toc(tic_iter);

        dw_time_vec(i)   = dw_time_i;
        iter_time_vec(i) = iter_time_i;

        send(dq, struct( ...
            'cmd',      'tick', ...
            'i',        i, ...
            'dwTime',   dw_time_i, ...
            'iterTime', iter_time_i));

    end  % parfor i

    wall_ti = toc(tic_ti_wall);
    crf_hole_time_sec(ti) = wall_ti;

    progress_update(struct('cmd','done'));

    %% ------------------------------------------------------------
    %  parfor timing summary
    %% ------------------------------------------------------------
    valid_dw = dw_time_vec(dw_time_vec > 0);
    valid_it = iter_time_vec(iter_time_vec > 0);

    fprintf('\n------------------------------------------------------------\n');
    fprintf('  測試孔 %s parfor timing summary\n', test_label_list{ti});
    fprintf('------------------------------------------------------------\n');
    fprintf('  wall time parfor              = %.2f s\n', wall_ti);
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

        serial_est = sum(valid_it);
        speedup_est = serial_est / wall_ti;

        fprintf('  estimated serial time          = %.2f s\n', serial_est);
        fprintf('  estimated parfor speedup       = %.2f x\n', speedup_est);
        fprintf('  effective wall time / sample   = %.3f s/sample\n', wall_ti / numel(valid_it));
    end

    if ~isnan(serial_DW_ref) && ~isempty(valid_dw)
        fprintf('  DW mean vs serial_DW_ref       = %.3f / %.3f = %.2f x\n', ...
            mean(valid_dw), serial_DW_ref, mean(valid_dw) / serial_DW_ref);
        fprintf('  說明：>1 代表平行時單次 DW 變慢，<1 代表單次 DW 反而變快。\n');
    end

    fprintf('------------------------------------------------------------\n\n');

    %% ------------------------------------------------------------
    %  parfor 結束後，寫回總 store
    %% ------------------------------------------------------------
    for i = 1:T_mcmc
        w_store{ti, i}           = w_store_ti{i};
        phiz_store{ti, i}        = phiz_store_ti{i};
        phih_store{ti, i}        = phih_store_ti{i};
        reshapedata_store{ti, i} = reshapedata_store_ti{i};
    end

    %% ------------------------------------------------------------
    %  轉回原始尺度
    %% ------------------------------------------------------------
    t_ele_original = zeros(size(t_ele));

    for i = 1:T_mcmc

        t_ele_reshaped = reshape(t_ele(:, i), nz, M);

        for p = 1:M

            param_idx_p = para_Cs(p);

            data_normal = t_ele_reshaped(:, p) + param_mean(p);
           % data_normal = t_ele_reshaped(:, p) ;
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

    %% ------------------------------------------------------------
    %  統計量
    %% ------------------------------------------------------------
    for p = 1:M

        row_idx = (1:nz) + (p-1) * nz;

        p_samp = real(t_ele_original(row_idx, :));

        pred_median(:, ti, p) = prctile(p_samp, 50,   2);
        pred_p025(:,   ti, p) = prctile(p_samp, 2.5,  2);
        pred_p975(:,   ti, p) = prctile(p_samp, 97.5, 2);
        pred_single(:, ti, p) = p_samp(:, 1);
        pred_samples(:, ti, p, :) = reshape(p_samp, nz, 1, 1, T_mcmc);
    end

    fprintf('  測試孔 %s 完成。\n', test_label_list{ti});

end  % for ti

time_CRF_sec = toc(tic_crf_total);
fprintf('\n===== Conditional Simulation 完成 =====\n');
fprintf('[TIME] CRF total elapsed = %.2f sec = %.2f min\n', time_CRF_sec, time_CRF_sec/60);
%% 0611平行運算結束
%% ===== 畫圖（GPR vs HBM 比較）=====
fprintf('繪製 GPR vs HBM 比較圖...\n');
tic_plot = tic;

is_Bq = false(1, M);
for p = 1:M
    name_tmp = lower(param_name{p});
    name_tmp = erase(name_tmp, {'_', ' ', '{', '}', '\'});
    is_Bq(p) = contains(name_tmp, 'bq');
end

plot_param_idx = [1 2 3 5 6 8];
M_plot = numel(plot_param_idx);
fprintf('畫圖參數數量 = %d，跳過參數：', M_plot);
disp(param_name(is_Bq));

fig_hbm = openfig('HBM_only_B3out.fig', 'invisible');
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
        warning('部分 legend handle 無效，略過 legend。');
    end

    if SAVE_RESULT_FIG == 1
        fig_name = fullfile(output_dir, sprintf('%s_%s.fig', fig_prefix, safe_filename(test_label_list{ti})));
        savefig(fig, fig_name);
        fprintf('[SAVE] Result FIG saved: %s\n', fig_name);
    end
end
time_plot_sec = toc(tic_plot);
fprintf('比較圖繪製完成。\n');
fprintf('[TIME] Plotting elapsed = %.2f sec = %.2f min\n', time_plot_sec, time_plot_sec/60);

% %% ===== 存檔 MGPR 預測結果 =====
% for ti = 1:nh_test
%     svp = sigvp_test(:, ti);
%     save_median = nan(nz, M_plot); save_p025 = nan(nz, M_plot);
%     save_p975   = nan(nz, M_plot); save_crf  = nan(nz, M_plot);
%     save_actual = nan(nz, M_plot); save_param_name = cell(1, M_plot);
% 
%     for pp = 1:M_plot
%         p = plot_param_idx(pp);
%         obs_val  = true_data(:, ti, p);
%         pred_med = pred_median(:, ti, p); pred_lo = pred_p025(:,ti,p);
%         pred_hi  = pred_p975(:,  ti, p); pred_crf = pred_single(:,ti,p);
%         if multiply_sigvp_all(p)
%             obs_val=obs_val.*svp; pred_med=pred_med.*svp;
%             pred_lo=pred_lo.*svp; pred_hi=pred_hi.*svp; pred_crf=pred_crf.*svp;
%         end
%         save_median(:,pp)=pred_med; save_p025(:,pp)=pred_lo;
%         save_p975(:,pp)=pred_hi;   save_crf(:,pp)=pred_crf;
%         save_actual(:,pp)=obs_val; save_param_name{pp}=param_name{p};
%     end
% 
%     fname = sprintf('MGPR_results_%s.mat', test_label_list{ti});
%     % save(fname,'save_median','save_p025','save_p975','save_crf','save_actual', ...
%     %     'save_param_name','z','plot_param_idx','multiply_sigvp_all','use_log_all');
%     fprintf('已存出：%s\n', fname);
% end


%% ========================================================================
%  GPR-MUSIC-3X Metrics: CI Coverage, MAE, RMSE_BV
%  RMSE_BV = sqrt( mean( bias_error^2 + predictive variance ) )
%
%  Notes:
%  1. MAE and RMSE_error use posterior median prediction.
%  2. RMSE_BV uses posterior samples:
%       RMSE_BV = sqrt(mean((y - predictive_mean)^2 + predictive_variance))
%  3. CI coverage uses posterior 2.5% / 97.5% interval.
%  4. In Scenario 2:
%       p = 6: s_u / sigma'_v, multiply by sigma'_v to obtain s_u
%       p = 8: qt1, multiply by sigma'_v to obtain q_t - sigma_v
%% ========================================================================

tic_metrics = tic;

fprintf('\n%s\n', repmat('=',1,135));
fprintf('  GPR-MUSIC-3X Conditional Prediction Metrics\n');
fprintf('  RMSE_BV = sqrt(mean((y - predictive mean)^2 + predictive variance))\n');
fprintf('%s\n', repmat('-',1,135));
fprintf('%-28s | %-14s | %8s | %12s | %12s | %12s | %12s | %12s | %14s\n', ...
    'Parameter', 'Test hole', 'N', 'CI cover', 'MAE', 'RMSE_err', 'RMSE_BV', 'Bias2', 'PredVar');
fprintf('%s\n', repmat('-',1,135));

GPR_METRICS = struct();

% If plot_param_idx is not defined, use all non-Bq parameters.
if ~exist('plot_param_idx','var') || isempty(plot_param_idx)
    is_Bq_metric = false(1,M);
    for p0 = 1:M
        name_tmp = lower(param_name{p0});
        name_tmp = erase(name_tmp, {'_', ' ', '{', '}', '\'});
        is_Bq_metric(p0) = contains(name_tmp, 'bq');
    end
    plot_param_idx = find(~is_Bq_metric);
end

% If multiply_sigvp_all is not defined, set a conservative default.
if ~exist('multiply_sigvp_all','var') || isempty(multiply_sigvp_all)
    multiply_sigvp_all = false(1,M);
    if M >= 8
        multiply_sigvp_all(6) = true;   % s_u / sigma'_v -> s_u
        multiply_sigvp_all(8) = true;   % qt1 -> q_t - sigma_v
    end
end

FinalMetricRows = {};
PooledMetricRows = {};

for pp = 1:numel(plot_param_idx)

    p = plot_param_idx(pp);
    pname = param_name{p};

    N_all        = nan(nh_test,1);
    cover_all    = nan(nh_test,1);
    mae_all      = nan(nh_test,1);
    rmse_err_all = nan(nh_test,1);
    rmse_bv_all  = nan(nh_test,1);
    bias2_all    = nan(nh_test,1);
    predvar_all  = nan(nh_test,1);
    width_all    = nan(nh_test,1);

    y_true_pool = [];
    y_med_pool  = [];
    y_mean_pool = [];
    y_var_pool  = [];
    lo_pool     = [];
    hi_pool     = [];

    for ti = 1:nh_test

        obs_val  = true_data(:, ti, p);
        pred_med = pred_median(:, ti, p);
        pred_lo  = pred_p025(:, ti, p);
        pred_hi  = pred_p975(:, ti, p);

        samp = squeeze(pred_samples(:, ti, p, :));   % nz x T_mcmc
        if isvector(samp)
            samp = samp(:);
        end

        if multiply_sigvp_all(p)
            svp = sigvp_test(:, ti);

            obs_val  = obs_val  .* svp;
            pred_med = pred_med .* svp;
            pred_lo  = pred_lo  .* svp;
            pred_hi  = pred_hi  .* svp;
            samp      = bsxfun(@times, samp, svp);
        end

        pred_mean = mean(samp, 2, 'omitnan');
        pred_var  = var(samp, 0, 2, 'omitnan');

        mask = isfinite(obs_val) & isfinite(pred_med) & ...
               isfinite(pred_mean) & isfinite(pred_var) & ...
               isfinite(pred_lo) & isfinite(pred_hi);

        n_valid = nnz(mask);

        if n_valid < 1
            fprintf('%-28s | %-14s | %8d | %12s | %12s | %12s | %12s | %12s | %14s\n', ...
                pname, test_label_list{ti}, n_valid, 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A');
            continue;
        elseif n_valid < 5
            warning('%s | %s only has N=%d valid points. Metrics are computed but may be unstable.', ...
                pname, test_label_list{ti}, n_valid);
        end

        y_true = obs_val(mask);

        y_med = pred_med(mask);
        err_med = y_true - y_med;

        y_mean = pred_mean(mask);
        y_var  = pred_var(mask);
        err_mean = y_true - y_mean;

        lo_use = pred_lo(mask);
        hi_use = pred_hi(mask);

        ci_cover     = 100 * mean(y_true >= lo_use & y_true <= hi_use, 'omitnan');
        mae_val      = mean(abs(err_med), 'omitnan');
        rmse_err_val = sqrt(mean(err_med.^2, 'omitnan'));

        bias2_val    = mean(err_mean.^2, 'omitnan');
        predvar_val  = mean(y_var, 'omitnan');
        rmse_bv_val  = sqrt(bias2_val + predvar_val);

        width_val = mean(hi_use - lo_use, 'omitnan');

        N_all(ti)        = n_valid;
        cover_all(ti)    = ci_cover;
        mae_all(ti)      = mae_val;
        rmse_err_all(ti) = rmse_err_val;
        rmse_bv_all(ti)  = rmse_bv_val;
        bias2_all(ti)    = bias2_val;
        predvar_all(ti)  = predvar_val;
        width_all(ti)    = width_val;

        y_true_pool = [y_true_pool; y_true(:)]; %#ok<AGROW>
        y_med_pool  = [y_med_pool;  y_med(:)];  %#ok<AGROW>
        y_mean_pool = [y_mean_pool; y_mean(:)]; %#ok<AGROW>
        y_var_pool  = [y_var_pool;  y_var(:)];  %#ok<AGROW>
        lo_pool     = [lo_pool;     lo_use(:)]; %#ok<AGROW>
        hi_pool     = [hi_pool;     hi_use(:)]; %#ok<AGROW>

        fprintf('%-28s | %-14s | %8d | %11.2f%% | %12.4g | %12.4g | %12.4g | %12.4g | %14.4g\n', ...
            pname, test_label_list{ti}, n_valid, ci_cover, mae_val, rmse_err_val, rmse_bv_val, bias2_val, predvar_val);
    end

    fprintf('%s\n', repmat('-',1,135));

    mean_N        = mean(N_all, 'omitnan');
    mean_cov      = mean(cover_all, 'omitnan');
    mean_mae      = mean(mae_all, 'omitnan');
    mean_rmse_err = mean(rmse_err_all, 'omitnan');
    mean_rmse_bv  = mean(rmse_bv_all, 'omitnan');
    mean_bias2    = mean(bias2_all, 'omitnan');
    mean_predvar  = mean(predvar_all, 'omitnan');
    mean_width    = mean(width_all, 'omitnan');

    fprintf('%-28s | %-14s | %8.1f | %11.2f%% | %12.4g | %12.4g | %12.4g | %12.4g | %14.4g  <-- Mean over holes\n', ...
        pname, 'Mean', mean_N, mean_cov, mean_mae, mean_rmse_err, mean_rmse_bv, mean_bias2, mean_predvar);

    mask_pool = isfinite(y_true_pool) & isfinite(y_med_pool) & ...
                isfinite(y_mean_pool) & isfinite(y_var_pool) & ...
                isfinite(lo_pool) & isfinite(hi_pool);

    if nnz(mask_pool) >= 1
        yt   = y_true_pool(mask_pool);
        ymed = y_med_pool(mask_pool);
        ym   = y_mean_pool(mask_pool);
        yv   = y_var_pool(mask_pool);
        yl   = lo_pool(mask_pool);
        yh   = hi_pool(mask_pool);

        err_med_pool  = yt - ymed;
        err_mean_pool = yt - ym;

        pooled_N        = nnz(mask_pool);
        pooled_cov      = 100 * mean(yt >= yl & yt <= yh, 'omitnan');
        pooled_mae      = mean(abs(err_med_pool), 'omitnan');
        pooled_rmse_err = sqrt(mean(err_med_pool.^2, 'omitnan'));
        pooled_bias2    = mean(err_mean_pool.^2, 'omitnan');
        pooled_predvar  = mean(yv, 'omitnan');
        pooled_rmse_bv  = sqrt(pooled_bias2 + pooled_predvar);
        pooled_width    = mean(yh - yl, 'omitnan');

        fprintf('%-28s | %-14s | %8d | %11.2f%% | %12.4g | %12.4g | %12.4g | %12.4g | %14.4g  <-- Pooled\n', ...
            pname, 'Pooled', pooled_N, pooled_cov, pooled_mae, pooled_rmse_err, pooled_rmse_bv, pooled_bias2, pooled_predvar);
    else
        pooled_N = NaN;
        pooled_cov = NaN;
        pooled_mae = NaN;
        pooled_rmse_err = NaN;
        pooled_rmse_bv = NaN;
        pooled_bias2 = NaN;
        pooled_predvar = NaN;
        pooled_width = NaN;
    end

    fprintf('%s\n', repmat('=',1,135));

    safe_name = matlab.lang.makeValidName(pname);

    GPR_METRICS.(safe_name).name = pname;
    GPR_METRICS.(safe_name).N = N_all;
    GPR_METRICS.(safe_name).CI_coverage_percent = cover_all;
    GPR_METRICS.(safe_name).MAE = mae_all;
    GPR_METRICS.(safe_name).RMSE_error = rmse_err_all;
    GPR_METRICS.(safe_name).RMSE_BV = rmse_bv_all;
    GPR_METRICS.(safe_name).Bias2 = bias2_all;
    GPR_METRICS.(safe_name).PredictiveVariance = predvar_all;
    GPR_METRICS.(safe_name).Mean_CI_width = width_all;

    GPR_METRICS.(safe_name).mean_over_holes.N = mean_N;
    GPR_METRICS.(safe_name).mean_over_holes.CI_coverage_percent = mean_cov;
    GPR_METRICS.(safe_name).mean_over_holes.MAE = mean_mae;
    GPR_METRICS.(safe_name).mean_over_holes.RMSE_error = mean_rmse_err;
    GPR_METRICS.(safe_name).mean_over_holes.RMSE_BV = mean_rmse_bv;
    GPR_METRICS.(safe_name).mean_over_holes.Bias2 = mean_bias2;
    GPR_METRICS.(safe_name).mean_over_holes.PredictiveVariance = mean_predvar;
    GPR_METRICS.(safe_name).mean_over_holes.Mean_CI_width = mean_width;

    GPR_METRICS.(safe_name).pooled.N = pooled_N;
    GPR_METRICS.(safe_name).pooled.CI_coverage_percent = pooled_cov;
    GPR_METRICS.(safe_name).pooled.MAE = pooled_mae;
    GPR_METRICS.(safe_name).pooled.RMSE_error = pooled_rmse_err;
    GPR_METRICS.(safe_name).pooled.RMSE_BV = pooled_rmse_bv;
    GPR_METRICS.(safe_name).pooled.Bias2 = pooled_bias2;
    GPR_METRICS.(safe_name).pooled.PredictiveVariance = pooled_predvar;
    GPR_METRICS.(safe_name).pooled.Mean_CI_width = pooled_width;

    FinalMetricRows(end+1,:) = {pname, mean_N, mean_cov, mean_mae, mean_rmse_err, mean_rmse_bv, mean_bias2, mean_predvar, mean_width}; %#ok<SAGROW>
    PooledMetricRows(end+1,:) = {pname, pooled_N, pooled_cov, pooled_mae, pooled_rmse_err, pooled_rmse_bv, pooled_bias2, pooled_predvar, pooled_width}; %#ok<SAGROW>
end

FinalMetricTable = cell2table(FinalMetricRows, 'VariableNames', ...
    {'Parameter','Mean_N','Mean_CI_percent','Mean_MAE','Mean_RMSE_error','Mean_RMSE_BV','Mean_Bias2','Mean_PredictiveVariance','Mean_CI_width'});

PooledMetricTable = cell2table(PooledMetricRows, 'VariableNames', ...
    {'Parameter','Pooled_N','Pooled_CI_percent','Pooled_MAE','Pooled_RMSE_error','Pooled_RMSE_BV','Pooled_Bias2','Pooled_PredictiveVariance','Pooled_CI_width'});

fprintf('\n%s\n', repmat('=',1,135));
fprintf('==== Final metric table: mean over holes ====\n');
disp(FinalMetricTable);
fprintf('==== Pooled metric table: pooled over all valid depths/holes ====\n');
disp(PooledMetricTable);

time_metrics_sec = toc(tic_metrics);
fprintf('[TIME] Metrics elapsed = %.2f sec = %.2f min\n', time_metrics_sec, time_metrics_sec/60);


%% ===== 畫出每個訓練孔的 trend（直接使用已存的 w，不重跑 DW）=====
fprintf('\n計算所有訓練孔的 trend（使用已存 w）...\n');
tic_trend = tic;

trend_train_all = zeros(nz, nh_train, M, T_mcmc);

for i = 1:T_mcmc
    % nh_test is usually 1 here. Use the first test-hole store for trend reconstruction.
    y_phiz_i = phiz_store{1, i};
    y_phih_i = phih_store{1, i};
    w_i      = w_store{1, i};   % 從 cell 取出，維度與對應 phi 一致

    % trend = phi_t_Cs ⊗ phih ⊗ phiz × w
    d_train_i        = kronmult2({phi_t_Cs, y_phih_i, y_phiz_i}, reshape(w_i.', [], M));
    d_train_reshaped = reshape(d_train_i, nz, nh_train, M);

    for k = 1:nh_train
        for p = 1:M
            param_idx_p = para_Cs(p);
            tmp = JS_2_original(d_train_reshaped(:,k,p) + param_mean(p), ...
                type(param_idx_p), ax(param_idx_p), bx(param_idx_p), ...
                ay(param_idx_p), by(param_idx_p));
            if log_ind(p) > 0.5
                trend_train_all(:,k,p,i) = exp(tmp);
            else
                trend_train_all(:,k,p,i) = tmp;
            end
        end
    end
end

trend_med = median(trend_train_all, 4);
trend_lo  = prctile(trend_train_all, 2.5,  4);
trend_hi  = prctile(trend_train_all, 97.5, 4);

% ── 畫圖 ──
col_trend_med = [0.2 0.4 0.8];
col_trend_CI  = [0.6 0.7 0.9];
col_sand_t    = [0.75 0.75 0.75];
fig_W_t = 145*M_plot + 260;
fig_H_t = 660;

for k = 1:nh_train
    svp_k = sigvp_all(:, train_idx(k));

    fig_t = figure('Name', sprintf('Trend - %s', location_labels{train_idx(k)}), ...
        'Position', [50,50,fig_W_t,fig_H_t], 'Color','w');

    for pp = 1:M_plot
        p = plot_param_idx(pp);
        lp   = left_margin + (pp-1)*(tile_w + h_gap);
        ax_t = axes('Position',[lp, bot, tile_w, ax_height]); %#ok<LAXES>
        hold(ax_t,'on'); box(ax_t,'on');
        set(ax_t,'YDir','reverse','FontSize',9, ...
            'XGrid','on','YGrid','on','GridAlpha',0.3,'Layer','bottom');
        ylim(ax_t,[min(z) max(z)]);
        if use_log_all(p), set(ax_t,'XScale','log'); end
        if ~isempty(x_lims_all{p}), xlim(ax_t, x_lims_all{p}); end
        xlabel(ax_t, param_name{p}, 'FontSize',9, 'FontWeight','bold');
        if pp == 1
            ylabel(ax_t,'Depth (m)','FontSize',9);
        else
            set(ax_t,'YTickLabel',{});
        end

        t_med = trend_med(:,k,p); t_lo = trend_lo(:,k,p); t_hi = trend_hi(:,k,p);
        if multiply_sigvp_all(p)
            t_med=t_med.*svp_k; t_lo=t_lo.*svp_k; t_hi=t_hi.*svp_k;
        end
        if use_log_all(p)
            t_lo(t_lo<=0)=NaN; t_hi(t_hi<=0)=NaN; t_med(t_med<=0)=NaN;
        end

        xl = xlim(ax_t);
        patch(ax_t,[xl(1) xl(2) xl(2) xl(1)],[3.4 3.4 4.66 4.66], ...
            col_sand_t,'FaceAlpha',1.0,'EdgeColor','none','HandleVisibility','off');

        plot(ax_t,t_lo, z,'--','Color',col_trend_CI, 'LineWidth',1.1,'HandleVisibility','off');
        plot(ax_t,t_hi, z,'--','Color',col_trend_CI, 'LineWidth',1.1,'HandleVisibility','off');
        plot(ax_t,t_med,z,'-', 'Color',col_trend_med,'LineWidth',2.0,'HandleVisibility','off');

        obs_k = param_list_original_all{p}(:, train_idx(k));
        if multiply_sigvp_all(p), obs_k = obs_k .* svp_k; end
        if use_log_all(p), obs_k(obs_k<=0) = NaN; end
        valid = isfinite(obs_k) & isfinite(z);
        if any(valid)
            scatter(ax_t,obs_k(valid),z(valid),45,'o', ...
                'MarkerFaceColor',col_data,'MarkerEdgeColor','k','LineWidth',1.0);
        end
    end

    % ── Legend：在 figure 上建立獨立 axes，用純 Line handle 避免 scatter 失效 ──
    ax_leg_t = axes('Parent', fig_t, ...
        'Position',[right_boundary+0.01, bot+ax_height*0.35, ...
        1-right_boundary-0.02, ax_height*0.4], ...
        'Visible','off', 'HitTest','off');
    hold(ax_leg_t, 'on');
    h_t_med  = plot(ax_leg_t, NaN, NaN, '-',  'Color', col_trend_med, 'LineWidth', 2.0);
    h_t_CI   = plot(ax_leg_t, NaN, NaN, '--', 'Color', col_trend_CI,  'LineWidth', 1.1);
    % 用帶 marker 的 line 代替 scatter，確保 handle 類型一致
    h_t_data = plot(ax_leg_t, NaN, NaN, 'o', ...
        'MarkerFaceColor', col_data, 'MarkerEdgeColor', 'k', ...
        'MarkerSize', 7, 'LineStyle', 'none');
    lg_t = legend(ax_leg_t, [h_t_med, h_t_CI, h_t_data], ...
        {'Median trend', '95% CI trend', 'Actual data'}, ...
        'Location', 'northwest', 'FontSize', 8, 'Box', 'on');
    lg_t.AutoUpdate = 'off';

    sgtitle(fig_t, sprintf('Trend (\\phi w) - %s', location_labels{train_idx(k)}), ...
        'FontSize',11,'FontWeight','bold');

    if SAVE_RESULT_FIG == 1
        trend_fig_name = fullfile(output_dir, sprintf('Trend_%s.fig', safe_filename(location_labels{train_idx(k)})));
        savefig(fig_t, trend_fig_name);
        fprintf('  [SAVE] Trend FIG saved: %s\n', trend_fig_name);
    end
end
time_trend_sec = toc(tic_trend);
fprintf('Trend 圖繪製完成。\n');
fprintf('[TIME] Trend plotting elapsed = %.2f sec = %.2f min\n', time_trend_sec, time_trend_sec/60);

%% ===== Timing summary and final MAT save =====
time_total_sec = toc(tic_total_all);

TimingSummary = table(time_TMCMC_sec, time_CRF_sec, time_plot_sec, time_metrics_sec, time_trend_sec, time_total_sec, ...
    time_TMCMC_sec/60, time_CRF_sec/60, time_plot_sec/60, time_metrics_sec/60, time_trend_sec/60, time_total_sec/60, ...
    'VariableNames', {'TMCMC_sec','CRF_sec','Plot_sec','Metrics_sec','Trend_sec','Total_sec', ...
                      'TMCMC_min','CRF_min','Plot_min','Metrics_min','Trend_min','Total_min'});

fprintf('\n%s\n', repmat('=',1,100));
fprintf('==== Timing summary ====\n');
disp(TimingSummary);
fprintf('[TIME] TMCMC   = %.2f sec = %.2f min\n', time_TMCMC_sec, time_TMCMC_sec/60);
fprintf('[TIME] CRF     = %.2f sec = %.2f min\n', time_CRF_sec, time_CRF_sec/60);
fprintf('[TIME] Plot    = %.2f sec = %.2f min\n', time_plot_sec, time_plot_sec/60);
fprintf('[TIME] Metrics = %.2f sec = %.2f min\n', time_metrics_sec, time_metrics_sec/60);
fprintf('[TIME] Trend   = %.2f sec = %.2f min\n', time_trend_sec, time_trend_sec/60);
fprintf('[TIME] Total   = %.2f sec = %.2f min\n', time_total_sec, time_total_sec/60);

fprintf('\n說明：\n');
fprintf('  CI coverage 使用 2.5%% / 97.5%% posterior percentile band。\n');
fprintf('  MAE 與 RMSE_error 使用 posterior median prediction。\n');
fprintf('  RMSE_BV = sqrt(mean((y - predictive mean)^2 + predictive variance))。\n');
fprintf('  Bias2 = mean((y - predictive mean)^2)，PredictiveVariance = mean(predictive variance)。\n');
fprintf('  s_u 與 q_t - sigma_v 已乘上 sigma''_v 轉為實際單位。\n');

if SAVE_DATA_MAT == 1
    save(output_mat_name, ...
        'GPR_METRICS', 'FinalMetricTable', 'PooledMetricTable', 'TimingSummary', ...
        'time_TMCMC_sec', 'time_CRF_sec', 'crf_hole_time_sec', 'time_plot_sec', ...
        'time_metrics_sec', 'time_trend_sec', 'time_total_sec', ...
        'pred_median', 'pred_p025', 'pred_p975', 'pred_single', 'pred_samples', 'true_data', ...
        'trend_med', 'trend_lo', 'trend_hi', ...
        'bhp_mcmc', 'sofv_mcmc', 'sofh_mcmc', 'nuv_mcmc', 'nuh_mcmc', ...
        'ahp_mcmc', 'sofv_t_mcmc', 'sofh_t_mcmc', 'x_mcmc', 'ln_S', ...
        'x_low', 'x_up', 'T_mcmc', 'Scenario', 'para_Cs', 'param_name', ...
        'plot_param_idx', 'test_label_list', 'z', 'sigvp_test', 'param_mean', '-v7.3');
    fprintf('\n[SAVE] Data MAT saved: %s\n', output_mat_name);
else
    fprintf('\n[SAVE] SAVE_DATA_MAT = 0, no MAT file saved.\n');
end
fprintf('\n全部完成。\n');

%%
function out = safe_filename(in)
    out = char(string(in));
    out = regexprep(out, '[^a-zA-Z0-9_\-\.]', '_');
    if isempty(out)
        out = 'unnamed';
    end
end

%%
function progress_update(msg)
% progress_update
% 用於 parfor DataQueue 顯示進度。
%
% 使用方式：
%   progress_update(struct('cmd','reset',...))
%   afterEach(dq, @progress_update)
%   send(dq, struct('cmd','tick',...))
%   progress_update(struct('cmd','done'))

    persistent count total printEvery label tStart dwSum iterSum dwN iterN nWorkers serialDW

    switch msg.cmd

        case 'reset'
            count      = 0;
            total      = msg.total;
            printEvery = msg.printEvery;
            label      = msg.label;
            nWorkers   = msg.nWorkers;
            serialDW   = msg.serialDW;

            tStart  = tic;
            dwSum   = 0;
            iterSum = 0;
            dwN     = 0;
            iterN   = 0;

            fprintf('  [progress] %s reset: total = %d, printEvery = %d, workers = %d\n', ...
                label, total, printEvery, nWorkers);

        case 'tick'
            count = count + 1;

            if isfield(msg, 'dwTime') && msg.dwTime > 0
                dwSum = dwSum + msg.dwTime;
                dwN = dwN + 1;
            end

            if isfield(msg, 'iterTime') && msg.iterTime > 0
                iterSum = iterSum + msg.iterTime;
                iterN = iterN + 1;
            end

            if mod(count, printEvery) == 0 || count == total

                elapsed = toc(tStart);
                pct = 100 * count / total;

                rate = count / elapsed;  % completed samples per wall second

                if count > 0
                    eta = (total - count) / max(rate, eps);
                else
                    eta = NaN;
                end

                if dwN > 0
                    meanDW = dwSum / dwN;
                else
                    meanDW = NaN;
                end

                if iterN > 0
                    meanIter = iterSum / iterN;
                    serialEst = meanIter * total;
                    speedupNow = serialEst / elapsed;
                else
                    meanIter = NaN;
                    serialEst = NaN;
                    speedupNow = NaN;
                end

                if ~isnan(serialDW) && ~isnan(meanDW)
                    dwRatio = meanDW / serialDW;
                else
                    dwRatio = NaN;
                end

                fprintf(['  [progress] %s | done %d/%d (%.1f%%) | elapsed %.1f s | ETA %.1f s | ', ...
                         'meanDW %.2f s | meanIter %.2f s | speedup %.2fx | DWratio %.2fx | last i=%d\n'], ...
                    label, count, total, pct, elapsed, eta, ...
                    meanDW, meanIter, speedupNow, dwRatio, msg.i);
            end

        case 'done'
            elapsed = toc(tStart);

            fprintf('  [progress] %s done: %d/%d completed, elapsed = %.1f s\n', ...
                label, count, total, elapsed);

        otherwise
            warning('progress_update: unknown cmd = %s', msg.cmd);
    end
end