%0702導入Cs
function log_like = BaytownGP_Matern_3D(x,y,Cs)
M = size(Cs,1);
bhp = 1/exp(x(1)); ln_sof_v = x(2); ln_sof_h = x(3); ln_nu_v = x(4);ahp = 1/exp(x(5)); ln_sof_v_t = x(6); ln_sof_h_t = x(7);
nu_v = exp(ln_nu_v); nu_h = nu_v ; sof_v = exp(ln_sof_v); sof_h = exp(ln_sof_h); sof_v_t = exp(ln_sof_v_t); sof_h_t = exp(ln_sof_h_t);

%0709Cs導入
[y.phiz,y.phih,alpha] = GP_matrices(ahp,sof_v_t,sof_h_t,y,Cs);
%建立趨勢得phi omege

%0702 將資料改成nz by (nh*M)
nh = size(y.temp_h,1); nz = size(y.temp_z,1); reshape_t = reshape(y.t,nz,nh*M); nan_ind = isnan(reshape_t);
[log_like_post]  = Baytown_log_like_Step2_2c(reshape_t,nan_ind,y.X,y.Y,y.z,sof_v,sof_h,nu_v,nu_h,bhp,alpha,y,Cs,ahp,sof_v_t,sof_h_t)
% [log_like_post]  = Baytown_log_like_Step2_2cV2(reshape_t,nan_ind,y.X,y.Y,y.z,sof_v,sof_h,nu_v,nu_h,bhp,alpha,y,Cs,ahp,sof_v_t,sof_h_t)
% [log_like_post2]  = Baytown_log_like_Step2c(reshape_t,nan_ind,y.X,y.Y,y.z,sof_v,sof_h,nu_v,nu_h,bhp,alpha,y,Cs,ahp,sof_v_t,sof_h_t)

% %% 時間比較
% % 先跑一次暖機（persistent 變數初始化）
% Baytown_log_like_Step2_2cV2(reshape_t,nan_ind,y.X,y.Y,y.z,sof_v,sof_h,nu_v,nu_h,bhp,alpha,y,Cs,ahp,sof_v_t,sof_h_t);
% Baytown_log_like_Step2c(reshape_t,nan_ind,y.X,y.Y,y.z,sof_v,sof_h,nu_v,nu_h,bhp,alpha,y,Cs,ahp,sof_v_t,sof_h_t);
% 
% % 正式計時
% N_rep = 10;
% 
% t1 = zeros(N_rep,1);
% for k = 1:N_rep
%     tic;
%     Baytown_log_like_Step2_2cV2(reshape_t,nan_ind,y.X,y.Y,y.z,sof_v,sof_h,nu_v,nu_h,bhp,alpha,y,Cs,ahp,sof_v_t,sof_h_t);
%     t1(k) = toc;
% end
% 
% t2 = zeros(N_rep,1);
% for k = 1:N_rep
%     tic;
%     Baytown_log_like_Step2c(reshape_t,nan_ind,y.X,y.Y,y.z,sof_v,sof_h,nu_v,nu_h,bhp,alpha,y,Cs,ahp,sof_v_t,sof_h_t);
%     t2(k) = toc;
% end
% 
% fprintf('V2   均值=%.4f s，中位數=%.4f s，最快=%.4f s\n', mean(t1), median(t1), min(t1));
% fprintf('原版 均值=%.4f s，中位數=%.4f s，最快=%.4f s\n', mean(t2), median(t2), min(t2));
% fprintf('加速比 = %.2f x\n', median(t2)/median(t1));
%%
%[log_like_post]  =log_like_Step2c_0208(reshape_t,nan_ind,y.X,y.Y,y.z,sof_v,sof_h,nu_v,nu_h,bhp,alpha,y,Cs,ahp)
%log_prior = log_prior_fun( bhp, sof_v, sof_h, nu_v, nu_h, ahp, sof_v_t, sof_h_t,y);
%log_like = log_like_post + log_prior;
log_like = log_like_post;
% disp(['log_like_2C_new = ', num2str(log_like)]);
% disp(['log_like_2C_intial = ', num2str(log_like_ci)]);
end

function lp = log_prior_fun(bhp, sofv, sofh, nuv, nuh, ahp, sofv_t, sofh_t, y)
lp = 0;

% ---------- helper ----------
l2log = @(x,mu,sig) -0.5*((log(x)-mu)/sig)^2;   % L2 on log(x)
l2    = @(x,mu,sig) -0.5*((x-mu)/sig)^2;        % L2 on x

% ---------- pick "typical scales" from geometry ----------
% vertical scale: use median dz-ish (rough) -> here use 2m as weak center
mu_sofv  = log(2.0);  sig_sofv  = 1.0;

% horizontal scale: use median hole spacing if you want (rough)
Dh = y.temp_h;
dmed = median(Dh(Dh>0), 'omitnan');
if ~isfinite(dmed) || dmed<=0, dmed = 10; end
mu_sofh = log(dmed);  sig_sofh = 1.0;

% ---------- residual lengthscales (MAIN anti-overfit) ----------
lp = lp + l2log(sofv, mu_sofv, sig_sofv);
lp = lp + l2log(sofh, mu_sofh, sig_sofh);

% ---------- trend lengthscales (prefer smoother than residual, weakly) ----------
lp = lp + l2log(sofv_t, log(max(sofv,1e-6)*3), 1.0);   % trend ~ 3x smoother (very weak)
lp = lp + l2log(sofh_t, log(max(sofh,1e-6)*3), 1.0);

% ---------- variance levels ----------
lp = lp + l2log(bhp, log(10), 1.5);
lp = lp + l2log(ahp, log(10), 1.5);

% ---------- ratio prior (often more effective than ahp alone) ----------
lp = lp + l2log(ahp/bhp, log(1.0), 1.0);

% ---------- smoothness nu (do NOT use log(nu) unless you really want that) ----------
lp = lp + l2(nuv, 1.2, 0.6);
lp = lp + l2(nuh, 1.2, 0.6);

end