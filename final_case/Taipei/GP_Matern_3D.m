%0702導入Cs
function log_like = GP_Matern_3D(x,y,Cs)
M = size(Cs,1);
bhp = 1/exp(x(1)); ln_sof_v = x(2); ln_sof_h = x(3); ln_nu_v = x(4);  ahp = 1/exp(x(5)); ln_sof_v_t = x(6); ln_sof_h_t = x(7);
nu_v = exp(ln_nu_v); nu_h = nu_v; sof_v = exp(ln_sof_v); sof_h = exp(ln_sof_h); sof_v_t = exp(ln_sof_v_t); sof_h_t = exp(ln_sof_h_t);

%0709Cs導入
[y.phiz,y.phih,alpha] = GP_matrices(ahp,sof_v_t,sof_h_t,y,Cs);
%建立趨勢得phi omege

%0702 將資料改成nz by (nh*M)
nh = size(y.temp_h,1); nz = size(y.temp_z,1); reshape_t = reshape(y.t,nz,nh*M); nan_ind = isnan(reshape_t);
%[log_like]  = log_like_Step2c_0208_EXT(reshape_t,nan_ind,y.X,y.Y,y.z,sof_v,sof_h,nu_v,nu_h,bhp,alpha,y,Cs,ahp,sof_v_t,sof_h_t);
[log_like]  = log_like_Step2c(reshape_t,nan_ind,y.X,y.Y,y.z,sof_v,sof_h,nu_v,nu_h,bhp,alpha,y,Cs);
% disp(['log_like_2C_new = ', num2str(log_like)]);
% disp(['log_like_2C_intial = ', num2str(log_like_ci)]);