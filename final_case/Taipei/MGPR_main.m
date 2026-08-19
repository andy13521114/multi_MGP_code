%% Taipei site: streamlined MGPR analysis
% -------------------------------------------------------------------------
% Purpose
%   1. Read and transform the Taipei MUSIC-X data.
%   2. Estimate MGPR hyperparameters with TMCMC.
%   3. Generate posterior conditional realizations.
%   4. Report prediction metrics and draw one five-panel result figure.
%
% Saving is OFF by default. Set SAVE_RESULTS or SAVE_FIGURE to true only
% when files are needed.
%

clear; clc; close all;
total_timer = tic;
rng(10,'twister');

%% ======================== USER SETTINGS ================================
T_mcmc     = 10;       % Use 1000 for the formal analysis
tmcmc_beta = 0.5;

SAVE_RESULTS = false;  % MAT and metric tables
SAVE_FIGURE  = false;  % PNG and FIG
PLOT_ONE_REALIZATION = true;

OUTPUT_DIR = fullfile(pwd,'Taipei_MGPR_results');
PNG_RESOLUTION = 300;

% Original 10-D order:
% 1=LL, 2=PI, 3=LI, 4=sigma'_v/Pa, 5=sigma'_p/Pa,
% 6=su/sigma'_v, 7=St, 8=Bq, 9=qt1, 10=qtu.
USE_PARAM = [1 1 1 1 1 1 0 0 1 0];

PA_KPA = 101.3;
EIG_THRESHOLD = 0.999;
PLOT_DEPTH_LIMIT = [12 27];

%% ======================== LOAD MODEL INPUTS ============================
J = load('CLAY_10_7490_para_rho.mat');
C = load('Cs_site_Md.mat');

required_fields = {'type','ax','bx','ay','by'};
for k = 1:numel(required_fields)
    assert(isfield(J,required_fields{k}), ...
        'CLAY_10_7490_para_rho.mat lacks %s.',required_fields{k});
end

keep_idx = find(USE_PARAM);
M = numel(keep_idx);

log_ind_all = [1 1 0 1 1 1 1 0 1 1];
param_name_all = {'LL (%)','PI (%)','LI', ...
    '\sigma''_v/P_a','\sigma''_p/P_a','s_u/\sigma''_v', ...
    'S_t','B_q','q_{t1}','q_{tu}'};

log_ind = log_ind_all(keep_idx);
param_name = param_name_all(keep_idx);

johnson_type = J.type(keep_idx);
johnson_ax = J.ax(keep_idx);
johnson_bx = J.bx(keep_idx);
johnson_ay = J.ay(keep_idx);
johnson_by = J.by(keep_idx);

if isfield(C,'Cs')
    Cs = double(C.Cs);
else
    names = fieldnames(C);
    Cs = double(C.(names{1}));
end

% Accept either the original 10-D matrix or an already reduced M-D matrix.
if size(Cs,1) == numel(log_ind_all)
    Cs = Cs(keep_idx,keep_idx);
elseif ~isequal(size(Cs),[M M])
    error('Cs dimension mismatch: expected 10x10 or %dx%d.',M,M);
end

% Convert Cs to a correlation matrix and enforce positive definiteness.
scale_diag = diag(Cs);
assert(all(isfinite(scale_diag) & scale_diag > 0), ...
    'Cs must have finite positive diagonal entries.');
Dcorr = diag(1./sqrt(scale_diag));
Cs = Dcorr*Cs*Dcorr;
Cs = make_spd(Cs,1e-8);

[phi_t_Cs,D_Cs] = eig(Cs);
[~,order] = sort(diag(D_Cs),'descend');
phi_t_Cs = phi_t_Cs(:,order);
L_Cs = chol(Cs,'lower');

% The five-panel result requires these six model inputs.
required_original_idx = [1 2 3 4 5 6];
assert(all(ismember(required_original_idx,keep_idx)), ...
    'USE_PARAM must include LL, PI, LI, sv, sp, and su.');

fprintf('\n%s\n',repmat('=',1,76));
fprintf('Taipei MGPR analysis\n');
fprintf('Selected parameters: %s\n',strjoin(param_name,', '));
fprintf('TMCMC population: %d\n',T_mcmc);
fprintf('%s\n',repmat('=',1,76));

%% ======================== READ AND TRANSFORM DATA ======================
data_table = xlsread('Taipei_case_table.xlsx');

% Preserve the row selection used in the supplied program.
data_table(2:2:276,:) = [];

z = data_table(:,1);
z = z(:);
nz = numel(z);

site_full = data_table(:,11:19);
site_full(:,10) = NaN;  % Add qtu so columns follow the original 10-D order

y_actual = site_full(:,keep_idx);
y_transformed = y_actual;

% Apply the same log transformations as the original program.
for p = 1:M
    if log_ind(p)
        y_transformed(y_transformed(:,p) <= 0,p) = NaN;
        y_transformed(:,p) = log(y_transformed(:,p));
    end
end

% Map each available observation to Johnson normal space.
x_site = nan(nz,M);
for p = 1:M
    valid = isfinite(y_transformed(:,p));
    x_site(valid,p) = JS_2_normal_gb(y_transformed(valid,p), ...
        johnson_type(p),johnson_ax(p),johnson_bx(p), ...
        johnson_ay(p),johnson_by(p));
end
x_site = real(x_site);

% Center every parameter using its available site observations.
param_mean = mean(x_site,1,'omitnan');
t_mat = x_site-param_mean;

%% ======================== MGPR DATA STRUCTURE ==========================
% Taipei is represented by one vertical profile at X=Y=0.
y = struct();
y.t = t_mat(:);
y.z = z;
y.X = 0;
y.Y = 0;
y.temp_h = 0;
y.temp_z = abs(z-z');
y.eig_thresh = EIG_THRESHOLD;

X_test = 0;
Y_test = 0;

% Prediction uses the same vertical profile and coordinates.
y.temp_h_ele = [0 0;0 0];
y.temp_z_ele = abs([z;z]-[z;z]');

%% ======================== HYPERPARAMETER BOUNDS ========================
% x = [b^{-1}, SOFv_e, SOFh_e, nu_e, a^{-1}, SOFv_t, SOFh_t]
z_range = max(y.temp_z(:));
if ~isfinite(z_range) || z_range <= 0
    z_range = max(max(z)-min(z),1);
end

x_low = [-log(10), log(0.2), log(1), log(0.3), ...
         -log(10), log(z_range/10), log(1)];
x_up  = [-log(0.01), log(10), log(1.001), log(3), ...
         -log(0.01), log(z_range*10), log(1.001)];

%% ======================== RUN MGPR =====================================
MGPR = run_mgpr(y,x_low,x_up,T_mcmc,tmcmc_beta,Cs, ...
    param_mean,johnson_type,johnson_ax,johnson_bx,johnson_ay,johnson_by, ...
    log_ind,z,M,phi_t_Cs,L_Cs,X_test,Y_test);

Plot = build_plot_result(MGPR.samples_original,y_actual,z,keep_idx,PA_KPA);
Metrics = calculate_metrics(Plot);

TimingEvidence = table(MGPR.time_tmcmc_sec,MGPR.time_crf_sec, ...
    MGPR.log_evidence,'VariableNames', ...
    {'TMCMC_sec','ConditionalSimulation_sec','LogEvidence'});

fprintf('\nTiming and model evidence\n');
disp(TimingEvidence);
fprintf('Prediction metrics\n');
disp(Metrics);

%% ======================== RESULT FIGURE ================================
fig_result = plot_mgpr_profiles(Plot,PLOT_ONE_REALIZATION,PLOT_DEPTH_LIMIT);

%% ======================== OPTIONAL SAVING ==============================
% No directory or file is created while both saving switches are false.
if SAVE_RESULTS || SAVE_FIGURE
    if ~exist(OUTPUT_DIR,'dir')
        mkdir(OUTPUT_DIR);
    end
end

base_name = sprintf('Taipei_MGPR_T%d',T_mcmc);

if SAVE_RESULTS
    save(fullfile(OUTPUT_DIR,[base_name '.mat']), ...
        'MGPR','Plot','Metrics','TimingEvidence','keep_idx','USE_PARAM', ...
        'Cs','z','y_actual','T_mcmc','-v7.3');

    try
        metric_file = fullfile(OUTPUT_DIR,[base_name '_metrics.xlsx']);
        writetable(TimingEvidence,metric_file,'Sheet','TimingEvidence');
        writetable(Metrics,metric_file,'Sheet','Metrics');
    catch ME
        warning('XLSX export failed: %s',ME.message);
        writetable(TimingEvidence,fullfile(OUTPUT_DIR, ...
            [base_name '_TimingEvidence.csv']));
        writetable(Metrics,fullfile(OUTPUT_DIR,[base_name '_Metrics.csv']));
    end
end

if SAVE_FIGURE
    png_file = fullfile(OUTPUT_DIR,[base_name '.png']);
    try
        exportgraphics(fig_result,png_file,'Resolution',PNG_RESOLUTION);
    catch
        print(fig_result,png_file,'-dpng',sprintf('-r%d',PNG_RESOLUTION));
    end
    savefig(fig_result,fullfile(OUTPUT_DIR,[base_name '.fig']));
end

elapsed_total = toc(total_timer);
fprintf('Total elapsed time: %.2f sec (%.2f min)\n', ...
    elapsed_total,elapsed_total/60);

%% ======================== LOCAL FUNCTIONS ==============================
function R = run_mgpr(y,x_low,x_up,T_mcmc,tmcmc_beta,Cs, ...
    param_mean,type_p,ax_p,bx_p,ay_p,by_p,log_ind, ...
    z,M,phi_t_Cs,L_Cs,X_test,Y_test)

nz = numel(z);

fprintf('\nStarting TMCMC...\n');
timer_tmcmc = tic;
[x_mcmc,ln_S,~,~,~] = iTMCMC_fun_mod1( ...
    'GP_Matern_3D',y,x_low,x_up,T_mcmc,tmcmc_beta,Cs);
time_tmcmc_sec = toc(timer_tmcmc);
fprintf('TMCMC completed in %.2f sec (%.2f min).\n', ...
    time_tmcmc_sec,time_tmcmc_sec/60);

if isempty(ln_S)
    log_evidence = NaN;
else
    log_evidence = sum(ln_S(:));
end

b = 1./exp(x_mcmc(:,1));
sofv = exp(x_mcmc(:,2));
sofh = exp(x_mcmc(:,3));
nuv = exp(x_mcmc(:,4));
nuh = nuv;
a = 1./exp(x_mcmc(:,5));
sofv_t = exp(x_mcmc(:,6));
sofh_t = exp(x_mcmc(:,7));

fprintf('Starting posterior conditional simulation...\n');
timer_crf = tic;

jitter_h = 1e-6;
jitter_z = 1e-6;
jitter_precision = 1e-11;
nh_train = numel(y.X);
Npost = size(x_mcmc,1);
samples_normal_centered = nan(nz*M,Npost);

for i = 1:Npost
    [phiz,phih,phiz_ele,phih_ele,~,ln_alpha] = ...
        GP_matrices_Step3(a(i),sofv_t(i),sofh_t(i),y,Cs);

    y_local = y;
    y_local.phiz = phiz;
    y_local.phih = phih;
    y_local.phiz_ele = phiz_ele;
    y_local.phih_ele = phih_ele;

    A_diag = exp(ln_alpha(:));

    Rh = Matern_R(nuh(i),sofh(i),y.temp_h)+jitter_h*eye(nh_train);
    Rz = Matern_R(nuv(i),sofv(i),y.temp_z)+jitter_z*eye(nz);
    Lh = chol(Rh,'lower');
    Lz = chol(Rz,'lower');

    data_grid = reshape(y.t,nz,nh_train*M);
    data_grid = DW_sampler_new2(data_grid,y.X,y.Y,z, ...
        sofv(i),sofh(i),nuv(i),nuh(i),b(i),A_diag,y_local,Cs, ...
        sofv_t(i),sofh_t(i),M);
    data_vec = data_grid(:);

    A_h = (Lh'\(Lh\phih)).';
    A_z = (Lz'\(Lz\phiz)).';
    A_c = (L_Cs'\(L_Cs\phi_t_Cs)).';
    rhs = kronmult2({A_c,A_h,A_z},data_vec);

    precision = spdiags(A_diag,0,numel(A_diag),numel(A_diag))+ ...
        (1/b(i))*kron(A_c*phi_t_Cs,kron(A_h*phih,A_z*phiz));
    precision = (precision+precision')/2+ ...
        jitter_precision*speye(size(precision,1));

    try
        Lp = chol(precision,'lower');
    catch
        precision = full((precision+precision')/2)+ ...
            jitter_precision*eye(size(precision,1));
        Lp = chol(precision,'lower');
    end

    mu_w = (1/b(i))*(Lp'\(Lp\rhs));
    w = mu_w+(Lp'\(Lp\randn(size(mu_w))));

    trend_test = kronmult2({phi_t_Cs,phih_ele,phiz_ele}, ...
        reshape(w,[],M));
    residual_train = data_vec- ...
        kronmult2({phi_t_Cs,phih,phiz},w);

    [mean_residual,Lh_test,Lz_test] = vertical_dense_stats( ...
        sofv(i),sofh(i),nuv(i),nuh(i),X_test,Y_test,z, ...
        y.X,y.Y,z,residual_train,M);

    residual_test = mean_residual+sqrt(b(i))* ...
        kronmult2({L_Cs,Lh_test,Lz_test},randn(nz*M,1));

    samples_normal_centered(:,i) = trend_test(:)+residual_test(:);
end

samples_original = inverse_to_original(samples_normal_centered,nz,M, ...
    param_mean,type_p,ax_p,bx_p,ay_p,by_p,log_ind);

time_crf_sec = toc(timer_crf);
fprintf('Conditional simulation completed in %.2f sec (%.2f min).\n', ...
    time_crf_sec,time_crf_sec/60);
fprintf('Log evidence: %.6g\n',log_evidence);

R = struct();
R.x_mcmc = x_mcmc;
R.ln_S = ln_S;
R.log_evidence = log_evidence;
R.time_tmcmc_sec = time_tmcmc_sec;
R.time_crf_sec = time_crf_sec;
R.samples_original = samples_original;
end

function X_original = inverse_to_original(X_centered,nz,M,param_mean, ...
    type_p,ax_p,bx_p,ay_p,by_p,log_ind)

Npost = size(X_centered,2);
X_original = nan(size(X_centered));

for i = 1:Npost
    Xi = reshape(X_centered(:,i),nz,M);
    for p = 1:M
        z_absolute = Xi(:,p)+param_mean(p);
        value = JS_2_original_gb(z_absolute,type_p(p),ax_p(p), ...
            bx_p(p),ay_p(p),by_p(p));
        if log_ind(p)
            value = exp(value);
        end
        Xi(:,p) = real(value);
    end
    X_original(:,i) = Xi(:);
end
end

function Plot = build_plot_result(samples_original,y_actual,z,keep_idx,Pa)
nz = numel(z);

idx_LL = find(keep_idx == 1,1);
idx_PI = find(keep_idx == 2,1);
idx_LI = find(keep_idx == 3,1);
idx_sv = find(keep_idx == 4,1);
idx_sp = find(keep_idx == 5,1);
idx_su = find(keep_idx == 6,1);

Plot = struct();
Plot.depth_m = z(:);

Plot.samples.LL = get_parameter_samples(samples_original,nz,idx_LL);
Plot.samples.PI = get_parameter_samples(samples_original,nz,idx_PI);
Plot.samples.LI = get_parameter_samples(samples_original,nz,idx_LI);

sp_ratio = get_parameter_samples(samples_original,nz,idx_sp);
sv_ratio = get_parameter_samples(samples_original,nz,idx_sv);
su_ratio = get_parameter_samples(samples_original,nz,idx_su);
Plot.samples.sigma_p_eff = sp_ratio*Pa;
Plot.samples.su = sv_ratio*Pa.*su_ratio;

Plot.observed.LL = y_actual(:,idx_LL);
Plot.observed.PI = y_actual(:,idx_PI);
Plot.observed.LI = y_actual(:,idx_LI);
Plot.observed.sigma_p_eff = y_actual(:,idx_sp)*Pa;
Plot.observed.su = y_actual(:,idx_sv)*Pa.*y_actual(:,idx_su);

fields = {'LL','PI','LI','sigma_p_eff','su'};
for k = 1:numel(fields)
    field = fields{k};
    S = Plot.samples.(field);
    Plot.mean.(field) = row_mean(S);
    Plot.median.(field) = prctile(S,50,2);
    Plot.CI_low.(field) = prctile(S,2.5,2);
    Plot.CI_up.(field) = prctile(S,97.5,2);
    Plot.one.(field) = S(:,1);
    Plot.variance.(field) = row_variance(S);
end
end

function S = get_parameter_samples(X,nz,param_idx)
rows = (1:nz)+(param_idx-1)*nz;
S = X(rows,:);
end

function T = calculate_metrics(Plot)
fields = {'LL','PI','LI','sigma_p_eff','su'};
names = {'LL (%)','PI (%)','LI', ...
    'sigma''_p (kPa)','s_u (kPa)'};
n = numel(fields);

Parameter = names(:);
N = nan(n,1);
Bias = nan(n,1);
MAE = nan(n,1);
RMSE_point = nan(n,1);
RMSE_BV = nan(n,1);
Coverage95_percent = nan(n,1);
Mean_CI_width = nan(n,1);

for k = 1:n
    f = fields{k};
    observed = Plot.observed.(f)(:);
    median_value = Plot.median.(f)(:);
    mean_value = Plot.mean.(f)(:);
    lower = Plot.CI_low.(f)(:);
    upper = Plot.CI_up.(f)(:);
    predictive_variance = Plot.variance.(f)(:);

    valid = isfinite(observed) & isfinite(median_value) & ...
        isfinite(mean_value) & isfinite(lower) & isfinite(upper) & ...
        isfinite(predictive_variance);
    N(k) = nnz(valid);

    if N(k) == 0
        continue;
    end

    error_median = median_value(valid)-observed(valid);
    error_mean = mean_value(valid)-observed(valid);
    pred_var = predictive_variance(valid);

    Bias(k) = mean(error_median);
    MAE(k) = mean(abs(error_median));
    RMSE_point(k) = sqrt(mean(error_median.^2));
    RMSE_BV(k) = sqrt(mean(error_mean.^2+pred_var));
    Coverage95_percent(k) = 100*mean( ...
        observed(valid) >= lower(valid) & observed(valid) <= upper(valid));
    Mean_CI_width(k) = mean(upper(valid)-lower(valid));
end

T = table(Parameter,N,Bias,MAE,RMSE_point,RMSE_BV, ...
    Coverage95_percent,Mean_CI_width);
end

function fig = plot_mgpr_profiles(Plot,plot_one,depth_limit)
fields = {'LL','PI','LI','sigma_p_eff','su'};
x_labels = {'LL (%)','PI (%)','LI', ...
    '\sigma''_p (kPa)','s_u (kPa)'};
x_limits = {[20 50],[0 30],[0 2.5],[1e1 1e3],[20 120]};
use_log_x = [false false false true false];
panel_labels = {'(a)','(b)','(c)','(d)','(e)'};

color_median = [1.00 0.00 1.00];
color_realization = [0.00 0.65 0.00];
color_data = [1.00 0.90 0.00];

fig = figure('Name','Taipei MGPR','Color','w', ...
    'Position',[80 80 1180 500]);

left_margin = 0.055;
right_blank = 0.205;
bottom = 0.16;
height = 0.76;
gap = 0.040;
n_panels = numel(fields);
panel_width = (1-left_margin-right_blank-gap*(n_panels-1))/n_panels;
depth = Plot.depth_m(:);

for k = 1:n_panels
    field = fields{k};
    left = left_margin+(k-1)*(panel_width+gap);
    axh = axes('Position',[left bottom panel_width height]); %#ok<LAXES>
    hold(axh,'on'); box(axh,'on');

    if use_log_x(k)
        plot_positive_profile(axh,Plot.median.(field),depth,'-', ...
            color_median,1.6);
        plot_positive_profile(axh,Plot.CI_low.(field),depth,'--', ...
            color_median,1.2);
        plot_positive_profile(axh,Plot.CI_up.(field),depth,'--', ...
            color_median,1.2);
        if plot_one
            plot_positive_profile(axh,Plot.one.(field),depth,'-', ...
                color_realization,1.0);
        end
        observed = Plot.observed.(field);
        valid = isfinite(observed) & isfinite(depth) & observed > 0;
        semilogx(axh,observed(valid),depth(valid),'o', ...
            'MarkerSize',5.5,'MarkerFaceColor',color_data, ...
            'MarkerEdgeColor','k','LineWidth',0.8,'LineStyle','none');
        set(axh,'XScale','log');
    else
        plot(axh,Plot.median.(field),depth,'-', ...
            'Color',color_median,'LineWidth',1.6);
        plot(axh,Plot.CI_low.(field),depth,'--', ...
            'Color',color_median,'LineWidth',1.2);
        plot(axh,Plot.CI_up.(field),depth,'--', ...
            'Color',color_median,'LineWidth',1.2);
        if plot_one
            plot(axh,Plot.one.(field),depth,'-', ...
                'Color',color_realization,'LineWidth',1.0);
        end
        observed = Plot.observed.(field);
        valid = isfinite(observed) & isfinite(depth);
        plot(axh,observed(valid),depth(valid),'o', ...
            'MarkerSize',5.5,'MarkerFaceColor',color_data, ...
            'MarkerEdgeColor','k','LineWidth',0.8,'LineStyle','none');
    end

    set(axh,'YDir','reverse','FontName','Times New Roman', ...
        'FontSize',10,'LineWidth',0.9,'TickDir','in','Layer','top');
    ylim(axh,depth_limit);
    xlim(axh,x_limits{k});
    xlabel(axh,x_labels{k},'Interpreter','tex','FontSize',13);
    if k == 1
        ylabel(axh,'Depth (m)','FontSize',12);
    else
        set(axh,'YTickLabel',[]);
    end
    grid(axh,'on');
    axh.GridAlpha = 0.18;
    text(axh,0.78,0.06,panel_labels{k},'Units','normalized', ...
        'FontSize',11,'FontWeight','bold');
end

% A compact English legend is placed in the reserved space on the right.
legend_ax = axes('Position',[0.815 0.62 0.15 0.20], ...
    'Visible','off'); %#ok<LAXES>
hold(legend_ax,'on');
h_median = plot(legend_ax,nan,nan,'-','Color',color_median,'LineWidth',1.6);
h_ci = plot(legend_ax,nan,nan,'--','Color',color_median,'LineWidth',1.2);
h_data = plot(legend_ax,nan,nan,'o','MarkerFaceColor',color_data, ...
    'MarkerEdgeColor','k','MarkerSize',5,'LineStyle','none');

if plot_one
    h_one = plot(legend_ax,nan,nan,'-','Color',color_realization, ...
        'LineWidth',1.0);
    legend_handles = [h_median h_ci h_one h_data];
    legend_text = {'Median (MGPR)','95% interval (MGPR)', ...
        'One realization','Observed data'};
else
    legend_handles = [h_median h_ci h_data];
    legend_text = {'Median (MGPR)','95% interval (MGPR)','Observed data'};
end

lgd = legend(legend_ax,legend_handles,legend_text, ...
    'Location','northwest','FontSize',8,'Box','on');
lgd.ItemTokenSize = [14 7];
lgd.AutoUpdate = 'off';
end

function plot_positive_profile(axh,x,z,line_style,color,line_width)
x = x(:);
z = z(:);
valid = isfinite(x) & isfinite(z) & x > 0;
semilogx(axh,x(valid),z(valid),line_style, ...
    'Color',color,'LineWidth',line_width);
end

function m = row_mean(A)
m = nan(size(A,1),1);
for i = 1:size(A,1)
    values = A(i,isfinite(A(i,:)));
    if ~isempty(values)
        m(i) = mean(values);
    end
end
end

function v = row_variance(A)
v = nan(size(A,1),1);
for i = 1:size(A,1)
    values = A(i,isfinite(A(i,:)));
    if numel(values) >= 2
        v(i) = var(values,0);
    elseif numel(values) == 1
        v(i) = 0;
    end
end
end

function A = make_spd(A,relative_floor)
A = real((A+A')/2);
[V,D] = eig(A);
d = real(diag(D));
scale = max([max(abs(d)),1]);
d = max(d,relative_floor*scale);
A = real(V*diag(d)*V');
A = real((A+A')/2);
end
