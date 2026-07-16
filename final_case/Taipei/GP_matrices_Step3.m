function [phiz,phih,phiz_ele,phih_ele,alpha,ln_alpha] = GP_matrices_Step3(ahp,sof_v_t,sof_h_t,y,Cs)
Cs = ahp*Cs;
[~, omege_t_Cs] = eig(Cs);
eigvals = diag(omege_t_Cs);         
[~, idx] = sort(eigvals, 'descend');     
omege_t_Cs = eigvals(idx); 


nh = size(y.temp_h,1); nz = size(y.temp_z,1);
nz_ele = size(y.temp_z_ele,1); nh_ele = size(y.temp_h_ele,1);
R_v_t = exp(-pi*y.temp_z_ele.^2/sof_v_t^2); [V_v_t,D_v_t] = eig(R_v_t); V_v_t = V_v_t*sqrt(nz_ele); D_v_t = D_v_t/nz_ele;
temp_v = diag(D_v_t); [temp_vv,s_ind] = sort(temp_v,'descend'); temp_v = temp_v(s_ind); V_v_t = V_v_t(:,s_ind);
cumsum_v = cumsum(temp_v)/sum(temp_v); max_id = sum(cumsum_v<y.eig_thresh)+1; ev_ind = [1:max_id];
phiz = V_v_t(1:nz,ev_ind); phiz_ele = V_v_t(nz+1:end,ev_ind); diag_v_t = temp_v(ev_ind);
R_h_t = exp(-pi*y.temp_h_ele.^2/sof_h_t^2); [V_h_t,D_h_t] = eig(R_h_t); V_h_t = V_h_t*sqrt(nh_ele); D_h_t = D_h_t/nh_ele;
temp_h = diag(D_h_t); [temp_hh,s_ind] = sort(temp_h,'descend'); temp_h = temp_h(s_ind); V_h_t = V_h_t(:,s_ind);
cumsum_h = cumsum(temp_h)/sum(temp_h); max_id = sum(cumsum_h<y.eig_thresh)+1; eh_ind = [1:max_id];
phih = V_h_t(1:nh,eh_ind); phih_ele = V_h_t(nh+1:end,eh_ind); diag_h_t = temp_hh(eh_ind);
ln_alpha = -log(kron(eigvals(idx),kron(diag_h_t,diag_v_t)));
alpha = exp(ln_alpha);
alpha = alpha(:);

