% 0702 導入 Cs
function [phiz, phih, alpha, ln_alpha] = GP_matrices(ahp, sof_v_t, sof_h_t, y, Cs)

    % --- 處理 Cross-covariance (Cs) ---
    aCs =  Cs;
    [~, omege_t_Cs] = eig(aCs);
    eigvals = diag(omege_t_Cs);         
    [~, idx] = sort(eigvals, 'descend');     
    % omege_t_Cs = eigvals(idx); % (原本的這行沒用到，註解保留你的原意)
    
    nh = size(y.temp_h, 1); 
    nz = size(y.temp_z, 1);

    % =========================================================
    % 1. 垂直向 (Vertical) 相關矩陣與特徵分解
    % =========================================================
    R_v_t = exp(-pi * y.temp_z.^2 / sof_v_t^2); 
    [V_v_t, D_v_t] = eig(R_v_t); 
    V_v_t = V_v_t * sqrt(nz); 
    D_v_t = D_v_t / nz;
    
    temp_v = diag(D_v_t); 
    [~, s_ind] = sort(temp_v, 'descend'); % 將沒用到的 temp_vv 換成 ~
    temp_v = temp_v(s_ind); 
    V_v_t = V_v_t(:, s_ind);
    
    cumsum_v = cumsum(temp_v) / sum(temp_v); 
    
    % [修正點]：避免 sum(...)+1 導致的 index out of bounds
    max_id_v = find(cumsum_v >= y.eig_thresh, 1, 'first');
    if isempty(max_id_v)
        max_id_v = length(temp_v); % 若數值精度導致找不到，則保留全部
    end
    ev_ind = 1:max_id_v;
    
    phiz = V_v_t(:, ev_ind); 
    diag_v_t = temp_v(ev_ind);

    % =========================================================
    % 2. 水平向 (Horizontal) 相關矩陣與特徵分解
    % =========================================================
    R_h_t = exp(-pi * y.temp_h.^2 / sof_h_t^2); 
    [V_h_t, D_h_t] = eig(R_h_t); 
    V_h_t = V_h_t * sqrt(nh); 
    D_h_t = D_h_t / nh; 
    
    temp_h = diag(D_h_t); 
    [~, s_ind] = sort(temp_h, 'descend'); % 將沒用到的 temp_hh 換成 ~
    temp_h = temp_h(s_ind); 
    V_h_t = V_h_t(:, s_ind);
    
    cumsum_h = cumsum(temp_h) / sum(temp_h); 
    
    % [修正點]：同樣加上安全邊界保護
    max_id_h = find(cumsum_h >= y.eig_thresh, 1, 'first');
    if isempty(max_id_h)
        max_id_h = length(temp_h);
    end
    eh_ind = 1:max_id_h;
    
    phih = V_h_t(:, eh_ind); 
    diag_h_t = temp_h(eh_ind);

    % =========================================================
    % 3. 計算先驗精度 (Prior Precision)
    % =========================================================
    % 0702之後 sig2_t 要改成 a; 0709 要導入 a Cs
    ln_alpha = -log(kron(ahp *eigvals(idx), kron(diag_h_t, diag_v_t)));
    alpha = exp(ln_alpha);
    alpha = alpha(:);
end