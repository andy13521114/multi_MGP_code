clear all; clc; close all; tic;
%% ===== 載入 Johnson 參數 與 基礎設定 =====
load('HBM_CLAY_8_OCR_sigv_new.mat');
%load('Cs.mat');
load('Cs_global.mat');
load('integrated_geotechnical_data_0521.mat');

Scenario = 2;
T_mcmc   = 3;

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
    % ★ 8 個參數中選 6 個：LL PI LI OCR su/σ'v qt1
    % 第5個是 su/σ'v（無因次），第6個是 qt1（無因次）
    % 畫圖時：su/σ'v × σ'v → su (kN/m2)；qt1 × σ'v → qt-σv (kN/m2)
    para_Cs    = [1, 2, 3,4, 5, 6,7,8];
    param_name = {'LL (%)', 'PI (%)', 'LI', '\sigma''_v/P_a', 'OCR', 's_u (kN/m^2)', 'Bq','q_t - \sigma_v (kN/m^2)'};
    log_ind    = [1 1 0 1 1 1 0 1];
    numbb      = 1:14;
    sounding   = 14;
    E = [6.401 6.401 6.401 6.401 6.401 6.401 6.401 6.401 0.000 12.802 0.000 12.802 0.000 12.802];
    N = [6.4 14.173 1.372 19.202 26.974 4.267 12.802 21.336 25.604 25.604 12.802 12.802 0.000 0.000];
end
Cs =Cs_global ;
Cs = Cs(para_Cs, para_Cs);
Cs = (Cs + Cs') / 2;
% M = 6;

%Cs =diag(sqrt(1./diag(Cs))) *Cs *(diag(sqrt(1./diag(Cs))));
%%
% D = sqrt(diag(Cs)); 
% 
% % 2. 將 Cs 轉為相關係數矩陣 R
% R = diag(1./D) * Cs * diag(1./D); 
% 
% % 3. 修改目標相關係數 (第5行第6列 與 第6行第5列)
% % 0.2850 -> 0.6
% R(5,6) = 0.4;
% R(6,5) = 0.4; 
% 
% % 4. 將修改後的 R 轉回新的共變異數矩陣 Cs
% Cs = diag(D) * R * diag(D);
%%
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

%%
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
%% 0521新增
%% ===== 過濾 CPT-3 Bq < 0.07 的深度（Bq 與 qt1 同時去除）=====
% 找 CPT-3 在 numbb 中的索引（座標 6.401, 21.336）
dist_cpt3 = sqrt((X_all - 6.401).^2 + (Y_all - 21.336).^2);
[~, idx_cpt3_global] = min(dist_cpt3);

% 找 CPT-3 在 numbb（train+test pool）中的局部索引
idx_cpt3_local = find(numbb == idx_cpt3_global);

if ~isempty(idx_cpt3_local)
    % Bq 是 para_Cs 裡的哪個參數？Scenario 1: p=2 (B_q); Scenario 2: 不在其中
    % qt1 是哪個參數？Scenario 1: p=1; Scenario 2: p=6 (qt-σv)

    % 先從 param_list_original_all 取 Bq（需知道 Bq 對應哪個 para_Cs index）
    % Scenario 1: Bq = para_Cs(2)=8 → p=2
    % Scenario 2: Bq 不在 para_Cs 裡，需另外取原始資料

    if Scenario == 1
        p_Bq  = 2;   % param_list_all{2} = B_q
        p_qt1 = 1;   % param_list_all{1} = qt1

        Bq_col = param_list_original_all{p_Bq}(:, idx_cpt3_local);
        mask_bad = Bq_col < -0.07;   % 邏輯遮罩，nz×1

        param_list_all{p_Bq}(mask_bad, idx_cpt3_local)  = NaN;
        param_list_all{p_qt1}(mask_bad, idx_cpt3_local) = NaN;
        param_list_original_all{p_Bq}(mask_bad, idx_cpt3_local)  = NaN;
        param_list_original_all{p_qt1}(mask_bad, idx_cpt3_local) = NaN;

        fprintf('CPT-3 過濾：共 %d 個深度點 Bq < 0.07，已設為 NaN\n', sum(mask_bad));

    elseif Scenario == 2
    % Scenario 2 的 para_Cs = [1,2,3,4,5,6,7,8]
    % 第7個是 Bq，第8個是 q_t - sigma_v

    bq_param_idx = 7;   % Bq 在原始 8 參數中的第7個
    col_bq = (bq_param_idx - 1)*all_nh_db + numbb(idx_cpt3_local);
    Bq_raw = data_structure.data(dep_range, col_bq);   % nz×1

    mask_bad = Bq_raw < 0.07;

    p_Bq = 7;
    p_qt = 8;

    % Bq 本身不合理，設成 NaN
    param_list_all{p_Bq}(mask_bad, idx_cpt3_local) = NaN;
    param_list_original_all{p_Bq}(mask_bad, idx_cpt3_local) = NaN;

    % 同深度的 qt 也一起設成 NaN
    param_list_all{p_qt}(mask_bad, idx_cpt3_local) = NaN;
    param_list_original_all{p_qt}(mask_bad, idx_cpt3_local) = NaN;

    fprintf('CPT-3 過濾 (Scenario 2)：共 %d 個深度點 Bq < 0.07，Bq 與 qt-σv 已設為 NaN\n', sum(mask_bad));
    end
end
%% ===== σ'v 矩陣（對齊深度範圍）=====
% data_structure.sigvp: [全深度 × 14孔]，σ'v (kN/m2)
sigvp_all = data_structure.sigvp(dep_range, :);   % nz × 14

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
           %6.401  1.372;
           6.401  19.202;
           6.401  26.974];

E_test = [6.401  1.372];   % ← 任意 (x,y) 皆可

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
    t(:, p) = tvec ;
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


 % x_low = [-log(10), log(0.1), log(0.1),   log(0.2),  -log(10),  log(max(temp_z(:))/10), log(max(temp_h(:))/10)];
 % x_up  = [-log(0.1),log(100), log(100), log(2.0),  -log(0.1), log(max(temp_z(:))*10), log(max(temp_h(:))*10)];
 x_low = [-log(10), log(0.01), log(0.1),   log(0.1),  -log(10),  log(max(temp_z(:))/10), log(max(temp_h(:))/10)];
 x_up  = [-log(0.01),log(100), log(100), log(3.0),  -log(0.01), log(max(temp_z(:))*10), log(max(temp_h(:))*10)];
  
y.eig_thresh = 0.999;

temp_x_ele = abs([X; X_test] - [X; X_test]');
temp_y_ele = abs([Y; Y_test] - [Y; Y_test]');
temp_z_ele = abs([z; z] - [z; z]');
temp_h_ele = sqrt(temp_x_ele.^2 + temp_y_ele.^2);
y.temp_z_ele = temp_z_ele; y.temp_h_ele = temp_h_ele;

fprintf('\n開始 TMCMC...\n');
[x_mcmc, ln_S, ~, ~, ~] = iTMCMC_fun_mod1('BaytownGP_Matern_3D', y, x_low, x_up, T_mcmc, 0.5, Cs);

bhp_mcmc    = 1./exp(x_mcmc(:,1));
sofv_mcmc   = exp(x_mcmc(:,2));
sofh_mcmc   = exp(x_mcmc(:,3));
nuv_mcmc    = exp(x_mcmc(:,4));
nuh_mcmc    = nuv_mcmc;
ahp_mcmc    = 1./exp(x_mcmc(:,5));
sofv_t_mcmc = exp(x_mcmc(:,6));
sofh_t_mcmc = exp(x_mcmc(:,7));
fprintf('TMCMC 完成。\n');
%%
save('TMCMC_B3out.mat', ...
    'bhp_mcmc', 'sofv_mcmc', 'sofh_mcmc', 'nuv_mcmc', 'nuh_mcmc', ...
    'ahp_mcmc', 'sofv_t_mcmc', 'sofh_t_mcmc', ...
    'x_mcmc', 'ln_S', ...
    'x_low', 'x_up', 'T_mcmc', 'Scenario', 'para_Cs', 'param_name');
fprintf('已存出 TMCMC_B3out.mat\n');
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

%% ===== Conditional Simulation =====
fprintf('\n開始 Conditional Simulation（共 %d 個測試孔）...\n', nh_test);
jitterRh = 1e-6; jitterRz = 1e-6; jitterP = 1e-11;
z_ele = z;

for ti = 1:nh_test
    X_ele = X_test(ti);
    Y_ele = Y_test(ti);
    fprintf('  預測測試孔 %s (%d/%d)...\n', test_label_list{ti}, ti, nh_test);

    temp_x_ele = abs([X; X_ele] - [X; X_ele]');
    temp_y_ele = abs([Y; Y_ele] - [Y; Y_ele]');
    temp_z_ele = abs([z; z_ele] - [z; z_ele]');
    temp_h_ele = sqrt(temp_x_ele.^2 + temp_y_ele.^2);
    y.temp_z_ele = temp_z_ele;
    y.temp_h_ele = temp_h_ele;

    t_ele = zeros(length(z_ele) * M, T_mcmc);

    for i = 1:T_mcmc
        [y_phiz, y_phih, y_phiz_ele, y_phih_ele, ~, ln_alpha] = ...
            GP_matrices_Step3(ahp_mcmc(i), sofv_t_mcmc(i), sofh_t_mcmc(i), y, Cs);
        y.phiz = y_phiz; y.phih = y_phih;
        y.phiz_ele = y_phiz_ele; y.phih_ele = y_phih_ele;

        A_diag = exp(ln_alpha(:));
        R_h = Matern_R(nuh_mcmc(i), sofh_mcmc(i), temp_h);
        R_z = Matern_R(nuv_mcmc(i), sofv_mcmc(i), temp_z);
        Rh  = R_h + jitterRh * eye(nh_train);
        Rz  = R_z + jitterRz * eye(nz);

        Lh_R = chol(Rh,'lower'); Lz_R = chol(Rz,'lower');
        reshape_data = reshape(y.t, nz, nh_train*M);

        if Scenario == 2
            reshape_data = DW_sampler_new2(reshape_data, X, Y, z, ...
                sofv_mcmc(i), sofh_mcmc(i), nuv_mcmc(i), nuh_mcmc(i), ...
                bhp_mcmc(i), A_diag, y, Cs, sofv_t_mcmc(i), sofh_t_mcmc(i), M);
        end

        reshape_vec = reshape(reshape_data, [], 1);
        AA = (Lh_R' \ (Lh_R \ y_phih)).';
        BB = (Lz_R' \ (Lz_R \ y_phiz)).';
        CC = (L_Cs_fixed' \ (L_Cs_fixed \ phi_t_Cs)).';
        temp_vec = kronmult2({CC, AA, BB}, reshape_vec);

        bhp = bhp_mcmc(i);
        P   = spdiags(A_diag, 0, length(A_diag), length(A_diag)) + ...
              (1/bhp) * kron(CC*phi_t_Cs, kron(AA*y_phih, BB*y_phiz));
        P   = (P+P')/2 + jitterP * speye(size(P,1));

        try
            R = chol(P,'lower');
        catch
            Pf = full(P); Pf = (Pf+Pf')/2 + jitterP*eye(size(Pf,1));
            R  = chol(Pf,'lower');
        end

        mu    = (1/bhp) * (R' \ (R \ temp_vec));
        w     = mu + (R' \ (R \ randn(size(mu))));
        w_row = w.';

        dd_temp = kronmult2({phi_t_Cs, y_phih, y_phiz}, w);
        t_diff  = reshape_vec - dd_temp;

        [E_X, L_h, L_z] = vertical_dense_stats(sofv_mcmc(i), sofh_mcmc(i), ...
            nuv_mcmc(i), nuh_mcmc(i), X_ele, Y_ele, z_ele, X, Y, z, t_diff, M);

        noise = E_X + sqrt(bhp) * kronmult2({L_Cs_fixed, L_h, L_z}, ...
            randn(length(z_ele)*M, 1));
        d_ele = kronmult2({phi_t_Cs, y_phih_ele, y_phiz_ele}, reshape(w_row.', [], M));

        t_ele(:,i) = d_ele(:) + noise(:);
    end

    % 轉回原始尺度
    t_ele_original = zeros(size(t_ele));
    for i = 1:T_mcmc
        t_ele_reshaped = reshape(t_ele(:,i), nz, M);
        for p = 1:M
            param_idx   = para_Cs(p);
            data_normal = t_ele_reshaped(:,p) ;
            tmp = JS_2_original(data_normal, type(param_idx), ax(param_idx), ...
                bx(param_idx), ay(param_idx), by(param_idx));
            if log_ind(p) > 0.5
                t_ele_reshaped(:,p) = exp(tmp);
            else
                t_ele_reshaped(:,p) = tmp;
            end
        end
        t_ele_original(:,i) = t_ele_reshaped(:);
    end

    for p = 1:M
        row_idx = (1:nz) + (p-1)*nz;
        p_samp  = t_ele_original(row_idx, :);
        p_samp  = real(t_ele_original(row_idx, :));
        pred_median(:, ti, p) = prctile(p_samp, 50,   2);
        pred_p025(:,   ti, p) = prctile(p_samp, 2.5,  2);
        pred_p975(:,   ti, p) = prctile(p_samp, 97.5, 2);
        pred_single(:, ti, p) = p_samp(:, 1);
    end
end
fprintf('\n===== Conditional Simulation 完成 =====\n');
%% ===== 畫圖（GPR vs HBM 比較；訓練含 Bq，但畫圖跳過 Bq）=====
fprintf('繪製 GPR vs HBM 比較圖...\n');

%% ── 建立畫圖用參數索引：跳過 Bq / B_q ──
% 訓練資料仍然是 M 個參數，但畫圖只畫非 Bq 的參數
is_Bq = false(1, M);
for p = 1:M
    name_tmp = lower(param_name{p});
    name_tmp = erase(name_tmp, {'_', ' ', '{', '}', '\'});
    is_Bq(p) = contains(name_tmp, 'bq');
end

plot_param_idx = [1 2 3 5 6 8];      % 真正要畫的參數 index
M_plot = numel(plot_param_idx);     % 子圖數量，Scenario 2 應該會是 6

fprintf('畫圖參數數量 = %d，跳過參數：', M_plot);
disp(param_name(is_Bq));

%% ── 從 HBM_only_B3.fig 讀出線條資料（close 前存好）──
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

%% ── 顏色與版面設定 ──
col_HBM_med = [0.0  0.0  0.0];
col_HBM_CI  = [0.0  0.0  0.0];
col_GPR_med = [0.85 0.0  0.85];
col_GPR_CI  = [0.85 0.0  0.85];
col_CRF     = [0.0  0.65 0.0];
col_data    = [1.0  0.9  0.0];
col_sand    = [0.75 0.75 0.75];

lw_HBM_med = 2.0; 
lw_HBM_CI  = 1.1;
lw_GPR_med = 2.0; 
lw_GPR_CI  = 1.1;
lw_CRF     = 1.1;

%% ── 每一個訓練參數的繪圖設定：長度要對應完整 M 個參數 ──
% Scenario 2:
% 1 LL
% 2 PI
% 3 LI
% 4 OCR
% 5 su/sigma'v → 畫成 su，所以乘 sigma'v
% 6 Bq → 不畫
% 7 qt1 → 畫成 qt - sigma_v，所以乘 sigma'v

if M == 8
    x_lims_all = {
        [0,150], ...        % LL
        [0,100], ...        % PI
        [-0.75,1.25], ...   % LI
        [-0.75,1.25], ...   % SV
        [1,100], ...        % OCR
        [10,1000], ...      % su
        [], ...             % Bq，不畫
        [100,10000] ...     % qt - sigma_v
    };

    use_log_all = [
        false, ...   % LL
        false, ...   % PI
        false, ...   % LI
        false, ...   % SV
        true,  ...   % OCR
        true,  ...   % su
        false, ...   % Bq，不畫
        true   ...   % qt - sigma_v
    ];

    multiply_sigvp_all = [
        false, ...   % LL
        false, ...   % PI
        false, ...   % LI
        false, ...   % SV
        false, ...   % OCR
        true,  ...   % su/sigma'v → su
        false, ...   % Bq，不畫
        true   ...   % qt1 → qt - sigma_v
    ];

else
    % 若未來不是 7 參數版本，給一個保守預設
    x_lims_all = cell(1, M);
    use_log_all = false(1, M);
    multiply_sigvp_all = false(1, M);

    for p = 1:M
        if log_ind(p) > 0.5
            use_log_all(p) = true;
        end
    end
end

%% ── 版面設定：用 M_plot，不用 M ──
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
    hl_hbm_leg = gobjects(2,1);   % 圖例用：CI 和 Median 各一條
    hbm_leg_done = false;

    %% ── 注意：pp 是第幾張圖；p 是原本完整參數中的 index ──
    for pp = 1:M_plot
        p = plot_param_idx(pp);

        lp   = left_margin + (pp-1)*(tile_w + h_gap);
        ax_h = axes('Position',[lp, bot, tile_w, ax_height]); %#ok<LAXES>
        hold(ax_h,'on'); box(ax_h,'on');

        set(ax_h,'YDir','reverse','FontSize',9, ...
            'XGrid','on','YGrid','on','GridAlpha',0.3,'Layer','bottom');

        ylim(ax_h,[min(z) max(z)]);

        if use_log_all(p)
            set(ax_h,'XScale','log'); 
        end

        if ~isempty(x_lims_all{p})
            xlim(ax_h, x_lims_all{p}); 
        end

        xlabel(ax_h, param_name{p}, 'FontSize',9, 'FontWeight','bold');

        if pp == 1
            ylabel(ax_h,'Depth (m)','FontSize',9);
        else
            set(ax_h,'YTickLabel',{});
        end

        %% ── 複製 HBM 線 ──
        % HBM_only_B3.fig 通常是 6 張，不含 Bq
        % 所以這裡用 pp 對應 HBM 第 pp 張圖，而不是用原始 p
        if pp <= numel(HBM_line_data)
            ld = HBM_line_data{pp};
            hv = gobjects(numel(ld),1);

            for li = 1:numel(ld)
                hv(li) = plot(ax_h, ld(li).XData, ld(li).YData, ...
                    'LineStyle', ld(li).LineStyle, ...
                    'Color',     ld(li).Color, ...
                    'LineWidth', ld(li).LineWidth, ...
                    'HandleVisibility','off');
            end

            % 只存一次圖例句柄
            if ~hbm_leg_done && numel(ld) > 0
                is_dash  = arrayfun(@(h) strcmp(get(h,'LineStyle'),'--'), hv);
                is_solid = arrayfun(@(h) strcmp(get(h,'LineStyle'),'-'),  hv);

                if any(is_dash)
                    idx_dash = find(is_dash,1);
                    set(hv(idx_dash),'HandleVisibility','on');
                    hl_hbm_leg(1) = hv(idx_dash);
                end

                if any(is_solid)
                    idx_solid = find(is_solid,1);
                    set(hv(idx_solid),'HandleVisibility','on');
                    hl_hbm_leg(2) = hv(idx_solid);
                end

                hbm_leg_done = true;
            end
        end

        %% ── GPR 數據：這裡仍然從完整 M 參數結果取第 p 個 ──
        obs_val  = true_data(:, ti, p);
        pred_med = pred_median(:, ti, p);
        pred_lo  = pred_p025(:,  ti, p);
        pred_hi  = pred_p975(:,  ti, p);
        pred_crf = pred_single(:, ti, p);

        %% ── su 與 qt 轉成實際單位 ──
        if multiply_sigvp_all(p)
            obs_val  = obs_val  .* svp;
            pred_med = pred_med .* svp;
            pred_lo  = pred_lo  .* svp;
            pred_hi  = pred_hi  .* svp;
            pred_crf = pred_crf .* svp;
        end

        %% ── log 軸不能有 0 或負值 ──
        if use_log_all(p)
            pred_lo(pred_lo <= 0)     = NaN; 
            pred_hi(pred_hi <= 0)     = NaN;
            pred_med(pred_med <= 0)   = NaN; 
            pred_crf(pred_crf <= 0)   = NaN;
            obs_val(obs_val <= 0)     = NaN;
        end

        %% ── 畫 GPR 曲線 ──
        hl(3) = plot(ax_h, pred_lo, z, '--', ...
            'Color', col_GPR_CI, 'LineWidth', lw_GPR_CI);

        plot(ax_h, pred_hi, z, '--', ...
            'Color', col_GPR_CI, 'LineWidth', lw_GPR_CI, ...
            'HandleVisibility','off');

        hl(4) = plot(ax_h, pred_med, z, '-', ...
            'Color', col_GPR_med, 'LineWidth', lw_GPR_med);

        hl(5) = plot(ax_h, pred_crf, z, '-', ...
            'Color', col_CRF, 'LineWidth', lw_CRF);

        %% ── 砂層不透明蓋住曲線 ──
        xl = xlim(ax_h);
        patch(ax_h, ...
            [xl(1) xl(2) xl(2) xl(1)], ...
            [3.4 3.4 4.66 4.66], ...
            col_sand, ...
            'FaceAlpha',1.0, ...
            'EdgeColor','none', ...
            'HandleVisibility','off');

        %% ── 黃色圓點最上層 ──
        valid = isfinite(obs_val) & isfinite(z);

        if any(valid)
            hl(6) = scatter(ax_h, obs_val(valid), z(valid), 55, 'o', ...
                'MarkerFaceColor', col_data, ...
                'MarkerEdgeColor', 'k', ...
                'LineWidth', 1.1);
        else
            hl(6) = scatter(ax_h, NaN, NaN, 55, 'o', ...
                'MarkerFaceColor', col_data, ...
                'MarkerEdgeColor', 'k');
        end
    end

    %% ── 圖例 ──
    hl(1) = hl_hbm_leg(1);   % HBM CI
    hl(2) = hl_hbm_leg(2);   % HBM Median

    valid_hl = true(size(hl));
    for ii = 1:numel(hl)
        valid_hl(ii) = isgraphics(hl(ii));
    end

    if all(valid_hl)
        ax_leg = axes('Position', ...
            [right_boundary+0.01, bot+ax_height*0.25, ...
             1-right_boundary-0.02, ax_height*0.55], ...
            'Visible','off');

        lg = legend(ax_leg, hl, ...
            {'95% CI (HBM-MUSIC-3X)', ...
             'Median (HBM-MUSIC-3X)', ...
             '95% CI (GPR-MUSIC-3X)', ...
             'Median (GPR-MUSIC-3X)', ...
             'GPR CRF sample', ...
             'Actual data'}, ...
            'Location','northwest', ...
            'FontSize',8, ...
            'Box','on');

        lg.Position(1)   = right_boundary + 0.01;
        lg.ItemTokenSize = [18 8];
        lg.AutoUpdate    = 'off';
    else
        warning('部分 legend handle 無效，本張圖略過 legend。');
    end

    exportgraphics(fig, sprintf('GPR_vs_HBM_%s.png', test_label_list{ti}), 'Resolution',300);
    savefig(fig, sprintf('GPR_vs_HBM_%s.fig', test_label_list{ti}));
end

fprintf('比較圖繪製完成。\n');