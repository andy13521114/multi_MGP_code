function [log_like] = Baytown_log_like_Step2_2cV2(data,nan_ind,x,y,z,sof_v,sof_h,nu_v,nu_h,bhp,alpha,Y,Cs,ahp,sof_v_t,sof_h_t) %#ok<INUSD>
% 等價原版 Baytown_log_like_Step2c，加速版
% 核心：persistent 快取不變量 + 向量化 Phi_o + 預分解

persistent pDh pDz pRh pRv pSofh pSofv pNuh pNuv pNhall pNv
persistent pNanHash pObsIdx pIzo pHido pPido pNobs pYo
persistent pPhiTCs pAlpha pMCt pMh pMz pRankW
persistent pPhiz pPhih

jitter   = 1e-8;
jitterRh = 1e-6;
jitterRv = 1e-6;

M_all  = size(Cs,1);
nh_all = round(size(data,2)/M_all);
nv     = size(data,1);

%% ── 1. 距離矩陣（只算一次）──
if isempty(pDh) || isempty(pNhall) || pNhall~=nh_all || pNv~=nv
    dx  = x(:)-x(:).'; dy = y(:)-y(:).';
    pDh = sqrt(dx.^2+dy.^2);
    pDz = abs(z(:)-z(:).');
    pNhall = nh_all; pNv = nv;
    % 清掉依賴距離的快取
    pRh=[]; pRv=[];
end

%% ── 2. Matern kernel（只在 sof/nu 改變時重算）──
if isempty(pRh) || isempty(pSofh) || abs(pSofh-sof_h)>1e-12 || abs(pNuh-nu_h)>1e-12
    Rh = Matern_R(nu_h,sof_h,pDh); Rh(isnan(Rh))=1;
    pRh = (Rh+Rh')/2 + jitterRh*eye(nh_all);
    pSofh=sof_h; pNuh=nu_h;
end

if isempty(pRv) || isempty(pSofv) || abs(pSofv-sof_v)>1e-12 || abs(pNuv-nu_v)>1e-12
    Rv = Matern_R(nu_v,sof_v,pDz); Rv(isnan(Rv))=1;
    pRv = (Rv+Rv')/2 + jitterRv*eye(nv);
    pSofv=sof_v; pNuv=nu_v;
end
Rh = pRh; Rv = pRv;

%% ── 3. phiz / phih / phi_t_Cs（從 Y 取，只存一次）──
phiz = Y.phiz; phih = Y.phih;
mz=size(phiz,2); mh=size(phih,2); mCt=M_all;
rank_w = mCt*mh*mz;

% 快取 phi_t_Cs（Cs 固定則只算一次）
if isempty(pPhiTCs) || isempty(pAlpha) || numel(pAlpha)~=numel(alpha) || any(abs(pAlpha-alpha(:))>1e-12)
    [phi_t_Cs,D] = eig(Cs);
    [~,si] = sort(diag(D),'descend');
    pPhiTCs = phi_t_Cs(:,si);
    pAlpha  = alpha(:);
    pMCt=mCt; pMh=mh; pMz=mz; pRankW=rank_w;
end
phi_t_Cs = pPhiTCs;

%% ── 4. 觀測索引快取（nan_ind 不變則重用）──
nan_hash = double(nan_ind(:))'*(1:numel(nan_ind))';
if isempty(pNanHash) || pNanHash~=nan_hash
    o_idx = find(~nan_ind(:));
    Nobs  = numel(o_idx);
    iz_o  = mod(o_idx-1,nv)+1;
    j_o   = floor((o_idx-1)/nv)+1;
    pid_o = floor((j_o-1)/nh_all)+1;
    hid_o = mod(j_o-1,nh_all)+1;
    pObsIdx=o_idx; pNobs=Nobs;
    pIzo=iz_o; pPido=pid_o; pHido=hid_o;
    pNanHash=nan_hash;
end
o_idx=pObsIdx; Nobs=pNobs;
iz_o=pIzo; pid_o=pPido; hid_o=pHido;

if Nobs==0; log_like=-Inf; return; end
y_o = data(o_idx);

%% ── 5. Sigma_res_oo（Hadamard，快）──
Cs_oo = Cs(pid_o,pid_o);
Rh_oo = Rh(hid_o,hid_o);
Rv_oo = Rv(iz_o, iz_o);

Sigma_res_oo = bhp*(Cs_oo.*Rh_oo.*Rv_oo);
Sigma_res_oo = (Sigma_res_oo+Sigma_res_oo')/2;
Sigma_res_oo(1:Nobs+1:end) = Sigma_res_oo(1:Nobs+1:end)+jitter;

%% ── 6. Phi_o（向量化，不用三層 for-loop）──
phi_t_o = phi_t_Cs(pid_o,:);   % Nobs×mCt
phih_o  = phih(hid_o,:);       % Nobs×mh
phiz_o  = phiz(iz_o,:);        % Nobs×mz

% 向量化 kron 展開：不用 for-loop
% Phi_o(:, (kCt-1)*mh*mz + (kh-1)*mz + kz) = phi_t_o(:,kCt).*phih_o(:,kh).*phiz_o(:,kz)
% 利用 bsxfun 一次算完
% Step 1: [Nobs x mCt*mh] = kron_hadamard(phi_t_o, phih_o)
A = reshape(phi_t_o, Nobs,mCt,1) .* reshape(phih_o,Nobs,1,mh);  % Nobs×mCt×mh
A = reshape(A, Nobs, mCt*mh);                                     % Nobs×(mCt*mh)
% Step 2: [Nobs x mCt*mh*mz]
Phi_o = reshape(A,Nobs,mCt*mh,1) .* reshape(phiz_o,Nobs,1,mz);  % Nobs×(mCt*mh)×mz
Phi_o = reshape(Phi_o, Nobs, rank_w);                             % Nobs×rank_w

%% ── 7. Cholesky of Sigma_res_oo ──
[L_res,p] = chol(Sigma_res_oo,'lower');
if p>0
    nug=jitter;
    for k=1:12
        nug=nug*10;
        [L_res,p]=chol(Sigma_res_oo+nug*eye(Nobs),'lower');
        if p==0; break; end
    end
    if p>0; log_like=-1e10; return; end
end

%% ── 8. Woodbury（與原版完全相同）──
invRes_yo  = L_res'\(L_res\y_o);
invRes_Phi = L_res'\(L_res\Phi_o);

M_w = diag(alpha) + Phi_o'*invRes_Phi;
M_w = (M_w+M_w')/2;

[L_Mw,p]=chol(M_w,'lower');
if p>0
    nug=1e-10;
    for k=1:12
        nug=nug*10;
        [L_Mw,p]=chol(M_w+nug*eye(rank_w),'lower');
        if p==0; break; end
    end
    if p>0; log_like=-1e10; return; end
end

logdet_res = 2*sum(log(diag(L_res)));
logdet_Mw  = 2*sum(log(diag(L_Mw)));
logdet_tot = logdet_res + logdet_Mw - sum(log(alpha));

tmp_w = Phi_o'*invRes_yo;
quad  = y_o'*invRes_yo - tmp_w'*(L_Mw'\(L_Mw\tmp_w));

log_like = -0.5*(Nobs*log(2*pi) + logdet_tot + quad);
if ~isfinite(log_like); log_like=-1e10; end
end