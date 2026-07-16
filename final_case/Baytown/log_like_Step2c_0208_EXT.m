function  [log_like] = log_like_Step2c_0208_EXT(data,nan_ind,x,y,z,sof_v,sof_h,nu_v,nu_h,bhp,alpha,Y,Cs,ahp,sof_v_t,sof_h_t) %#ok<INUSD>
%function  [log_like] = Baytown_log_like_Step2c(data,nan_ind,x,y,z,sof_v,sof_h,nu_v,nu_h,bhp,alpha,Y,Cs)
%輸入的alpha要考慮Omege_t_Cs
%0703 將Cs做特徵分解   
[phi_t_Cs, omege_t_Cs] = eig(Cs);
eigvals = diag(omege_t_Cs);         
[~, idx] = sort(eigvals, 'descend');     
omege_t_Cs = diag(eigvals(idx));       
phi_t_Cs = phi_t_Cs(:, idx);                 
L_Cs = chol(Cs,'lower');
M = size(Cs,1);
inv_Cs =  inv(Cs);
log_Cs =  sum(log( sqrt(bhp) * diag(L_Cs)) );



nh=(size(data,2))/M;
nan_r_n = sum(nan_ind,2); 
empty_depth=find(nan_r_n==nh); nan_ind(empty_depth,:)=[]; data(empty_depth,:)=[]; z(empty_depth)=[]; nan_r_n(empty_depth)=[];
nv=size(data,1); n=nh*nv;
z_minus = (nan_r_n==0); z_plus = (nan_r_n>0);
minus = false(nv,nh*M); 
minus(z_minus,:) = true; minus = minus(:);
plus = false(nv,nh*M); 
plus(z_plus,:) = true; 
local_plus = plus(z_plus,:);
plus = plus(:); 
local_plus = local_plus(:);
plus_o = false(nv,nh*M); 
plus_o(z_plus,:) = true-nan_ind(z_plus,:); 
local_plus_o = plus_o(z_plus,:);
plus_o = plus_o(:); 
local_plus_o = local_plus_o(:); 
o_loc=find(local_plus_o==true);


% D = zeros(sum(local_plus_o),sum(local_plus));
% for i=1:sum(local_plus_o)
%     D(i,o_loc(i))=1;
% end
 
disz = abs(z'-z); R_v = Matern_R(nu_v,sof_v,disz); R_v(isnan(R_v)) = 1; R_v = R_v + 1e-6*eye(size(R_v));
disx = abs(x'-x); disy = abs(y'-y); dish = sqrt(disx.^2+disy.^2); R_h = Matern_R(nu_h,sof_h,dish); R_h(isnan(R_h)) = 1; R_h = R_h + 1e-6*eye(size(R_h));
invRh = inv(R_h); invRv_minus = inv(R_v(z_minus,z_minus));
1;
log_detRv_minus = 2*sum(log(diag(chol(R_v(z_minus,z_minus),'lower')))); log_detRh = 2*sum(log(diag(chol(R_h,'lower'))));
%0708 導入L_Cs ii


phiz = Y.phiz; phiz(empty_depth,:)=[]; phih = Y.phih; O = diag(1./alpha); invO = diag(alpha); mz = size(phiz,2); mh = size(phih,2);
invRv_phiz_minus = invRv_minus*phiz(z_minus,:); invRh_phih = invRh*phih;
Rv_invRv_plus_minus = R_v(z_plus,z_minus)*invRv_minus; Rv_invRv_phiz_plus_minus = Rv_invRv_plus_minus*phiz(z_minus,:);

condRv_plus_minus = R_v(z_plus,z_plus)-Rv_invRv_plus_minus*R_v(z_minus,z_plus);
if min(eig(condRv_plus_minus))<0
    invRv_plus = inv(R_v(z_plus,z_plus)); Rv_invRv_minus_plus = R_v(z_minus,z_plus)*invRv_plus;
    condRv_plus_minus = inv(invRv_plus+Rv_invRv_minus_plus'*inv(R_v(z_minus,z_minus)-Rv_invRv_minus_plus*R_v(z_plus,z_minus))*Rv_invRv_minus_plus);
end

% phi = kron(phih,phiz); phi_minus = kron(phih,phiz(z_minus,:));
% R = kron(R_h,R_v); R_minus = kron(R_h,R_v(z_minus,z_minus));
tmp = data(z_minus,:); y_minus = tmp(:);
% C = inv(invO+(1/sig2)*phi_minus'*inv(R_minus)*phi_minus);


%0703 Cw導入Cs Omege未考慮Cs

phih_invRh_phih = phih'*invRh_phih; phiz_minus_invRv_phiz_minus = phiz(z_minus,:)'*invRv_phiz_minus;
phitCs_invCs_phitCs = phi_t_Cs'*inv_Cs*phi_t_Cs;

C = inv(invO+ (1/bhp)*kron(phitCs_invCs_phitCs,kron(phih_invRh_phih,phiz_minus_invRv_phiz_minus))); C = (C+C')/2;


if min(eig(C))>0
    LC = chol(C,'lower'); ln_det_C = 2*sum(log(diag(LC)));
else
    [u_Cs, v_Cs] = eig((phitCs_invCs_phitCs + phitCs_invCs_phitCs')/2);
    [uh,vh] = eig((phih_invRh_phih+phih_invRh_phih')/2); 
    [uz,vz] = eig((phiz_minus_invRv_phiz_minus+phiz_minus_invRv_phiz_minus')/2);
    uu = kron(u_Cs,kron(uh,uz)); vv = invO + 1/bhp*diag(kron(diag(v_Cs),kron(diag(vh),diag(vz)))); %(1/sig2)*diag
    ln_det_C = sum(-log(diag(vv))); LC = uu*diag(sqrt(1./diag(vv)))*uu';
end
%0711_1
% D_plus_o=zeros(sum(local_plus_o),sum(local_plus));
% for i=1:sum(local_plus_o),
%     D_plus_o(i,o_loc(i))=1;
% end

phihT_invRh = phih'*invRh;phizT_invRv_minus = phiz(z_minus,:)'*invRv_minus;
% 0708 xi^o更改
if sum(z_plus)>0.5
    % mu = (1/sig2)*C*phi_minus'*inv(R_minus)*y_minus;
    %0710 更改維度 y_minus E_t的維度只有Z
    tmp = kron(phihT_invRh,phizT_invRv_minus)*reshape(y_minus,sum(z_minus)*nh,M)*inv_Cs'*phi_t_Cs; mu =(1/bhp)*C*tmp(:); %0708 LINE 28
    % E_t_minus = phi*mu;
    
    tmp = kron(phih,phiz)*reshape(mu,mz*mh,M)*phi_t_Cs'; 
    tmp22N  = tmp;
    %tmp22N = reshape(tmp22N,mz*mh*M,1);
    tmp22Nv = tmp22N(:);
    E_t_minus = tmp(:);

    vvvv = {phi_t_Cs,phih,phiz};
    tmpvv=  kronmult2({phi_t_Cs,phih,phiz},mu);
   
    % E_yplus_minus_c = E_t_minus(plus)+kron(eye(nh),R_v(z_plus,z_minus)*inv(R_v(z_minus,z_minus)))*(y_minus-E_t_minus(minus));
    tmp3 = kron(eye(nh),Rv_invRv_plus_minus)*reshape(y_minus-E_t_minus(minus),sum(z_minus)*nh,M);
    E_yplus_minus = E_t_minus(plus)+tmp3(:);
    %0711_2    
%     for i=1:sum(local_plus_o),
%         D=zeros(1,sum(local_plus)); D(o_loc(i))=1;
% %         ttt1 = (phiz(z_plus,:)-Rv_invRv_phiz_plus_minus)'*reshape(D',sum(z_plus),nh)*phih; ttt1 = ttt1(:);
%         %     ttt1 = (phiz(z_plus,:)-R_v(z_plus,z_minus)*inv(R_v(z_minus,z_minus))*phiz(z_minus,:))'*reshape(D(i,:)',sum(local_plus_o),nh)*phih; ttt1 = ttt1(:);
% %         ttt1 = C*ttt1; ttt1 = (phiz(z_plus,:)-Rv_invRv_phiz_plus_minus)*reshape(ttt1,mz,mh)*phih'; ttt1 = ttt1(:);
%         ttt2 = sig2*condRv_plus_minus*reshape(D',sum(z_plus),nh)*R_h'; ttt2 = ttt2(:);
%         for j=i:sum(local_plus_o),
%             DD=zeros(1,sum(local_plus)); DD(o_loc(j))=1;
%             Var_plus_o_minus(i,j) = DD*ttt2;
%             Var_plus_o_minus(j,i) = Var_plus_o_minus(i,j);
%         end
%     end
    %0703 導入pih_t_Cs trend part
    %C^O part2 雖是C*但只取有觀測值的部分

    %---
    tmp = kron(phi_t_Cs',kron(phih',(phiz(z_plus,:)-Rv_invRv_phiz_plus_minus)')); 
    ttt1 = tmp(:,local_plus_o);
    ttt1=LC'*ttt1; 
    ttt1=ttt1'*ttt1+1e-4*eye(size(ttt1,2));
    %---
    
    %     ttt1c=kron_multiply_vector(phih',(phiz(z_plus,:)-Rv_invRv_phiz_plus_minus)',D_plus_o');
%     ttt1c=LC'*ttt1c; ttt1c=ttt1c'*ttt1c+1e-8*eye(size(ttt1c,2));

    %0703 導入Cs noise part
    %C* part1  雖是C*但只取有觀測值的部分
    tmp = kron(bhp*Cs,kron(R_h,condRv_plus_minus));
   
    ttt2 = tmp(local_plus_o,local_plus_o);
%     ttt2c=kron_multiply_vector(R_h,condRv_plus_minus,D_plus_o');
%     ttt2c=D_plus_o*sig2*ttt2c;
    Var_plus_o_minus=ttt1+ttt2; 
    Var_plus_o_minus=(Var_plus_o_minus+Var_plus_o_minus')/2;
    Var_plus_o_minus = Var_plus_o_minus + 1e-5*eye(size(Var_plus_o_minus));
    
    % Var_plus_o_minus = D*ttt;
    ttt = data(:); y_plus_o = ttt(plus_o);
    L = chol(Var_plus_o_minus,'lower'); 
    v = L\(y_plus_o - E_yplus_minus(local_plus_o));
    log_like_plus_o = -0.5*M*sum(plus_o)*log(2*pi) - sum(log(diag(L))) - 0.5*v'*v;


 
%0711_3
%    Var_plus_o_minus = D*(part1+part2*C*part2')*D';
    
%     part1 = sig2*kron(R_h,R_v(z_plus,z_plus)-R_v(z_plus,z_minus)*inv(R_v(z_minus,z_minus))*R_v(z_minus,z_plus));
%     part2 = kron(phih,phiz(z_plus,:)-R_v(z_plus,z_minus)*inv(R_v(z_minus,z_minus))*phiz(z_minus,:));
%     tmp = part1+part2*C*part2'; Var_plus_o_minus_c = tmp(local_plus_o,local_plus_o);
    % log_like_plus_o_c = -0.5*sum(plus_o)*log(2*pi) - 0.5*log(det(Var_plus_o_minus_c)) - 0.5*(y_plus_o - E_yplus_minus_c(local_plus_o))'*inv(Var_plus_o_minus_c)*(y_plus_o - E_yplus_minus_c(local_plus_o));
    % phi = kron(phih,phiz); R = kron(R_h,R_v);
    % S = phi*O*phi'+sig2*R; % E_yplus_minus_c = S(plus,minus)*inv(S(minus,minus))*y_minus;
    % Var_yplus_minus_c = S(minus,minus);

    %log_Cs =log(sum(diag(bhp*L_Cs)));
  

    vvv = {phi_t_Cs'*inv_Cs, phih'*invRh, (phiz(z_minus,:)'*invRv_minus)};
    v1 = 1/bhp*kronmult2({inv_Cs, invRh, invRv_minus},y_minus(:)); 
    v1 = v1(:);
    tmp1 = y_minus'*v1;
    v1 = 1/bhp*kronmult4(vvv,reshape(y_minus,sum(z_minus)*nh*M,1)); 
    tmp2 = v1'*C*v1;
    e = tmp1 - tmp2; 
    log_detRv=-2*sum(log(diag(chol(R_v(z_minus,z_minus),'lower'))));log_detRh = 2*sum(log(diag(chol(R_h,'lower'))));
    d = -2*sum(log(diag(LC)))  + M*sum(z_minus)*log_detRh  +  M*nh*log_detRv_minus + 2*nh*sum(z_minus)*log_Cs -sum(log(alpha));
    log_like_minus = -0.5*M*nh*sum(z_minus)*log(2*pi) - 0.5*d - 0.5*e;
    % ttt = data(:); y_plus_o = ttt(plus_o);
    % log_like_minus_c = -0.5*nh*sum(z_minus)*log(2*pi)-0.5*log(det(Var_yplus_minus_c))-0.5*y_minus'*inv(Var_yplus_minus_c)*y_minus;
 

    % ttt = data(:); y_plus_o = ttt(plus_o);
    % log_like_minus_c = -0.5*nh*sum(z_minus)*log(2*pi)-0.5*log(det(Var_yplus_minus_c))-0.5*y_minus'*inv(Var_yplus_minus_c)*y_minus;
    log_like = log_like_minus + log_like_plus_o;
    1;
   

else
    %0708 lattice 資料 直接定義alpha beta
    vvv = {phi_t_Cs'*inv_Cs, phih'*invRh, (phiz(z_minus,:)'*invRv_minus)};
    v1 = 1/bhp*kronmult2({inv_Cs, invRh, invRv_minus},y_minus(:)); 
    v1 = v1(:);
    tmp1 = y_minus'*v1;
    v1 = 1/bhp*kronmult2(vvv,reshape(y_minus,sum(z_minus)*nh*M,1)); 
    tmp2 = v1'*C*v1;
    e = tmp1 - tmp2; 
    log_detRv=-2*sum(log(diag(chol(R_v(z_minus,z_minus),'lower'))));log_detRh = 2*sum(log(diag(chol(R_h,'lower'))));
    d = -2*sum(log(diag(LC)))  + M*sum(z_minus)*log_detRh  +  M*nh*log_detRv_minus + 2*nh*sum(z_minus)*log_Cs -sum(log(alpha));
    log_like_minus = -0.5*M*nh*sum(z_minus)*log(2*pi) - 0.5*d - 0.5*e;
    % ttt = data(:); y_plus_o = ttt(plus_o);
    % log_like_minus_c = -0.5*nh*sum(z_minus)*log(2*pi)-0.5*log(det(Var_yplus_minus_c))-0.5*y_minus'*inv(Var_yplus_minus_c)*y_minus;
    log_like = log_like_minus;
   
end
1+1;
% tmp = 1-nan_ind; o_ind = logical(tmp(:)); tmp = data(:); y_o = tmp(o_ind);
% log_like_c = -0.5*sum(o_ind)*log(2*pi)-0.5*log(det(S(o_ind,o_ind)))-0.5*y_o'*inv(S(o_ind,o_ind))*y_o;

%% 1008新增返回
    if ~isfinite(log_like)
        warning('log_like 非有限! 值 = %f', log_like);
        fprintf('  log_like_minus = %f\n', log_like_minus);
        if exist('log_like_plus_o', 'var')
            fprintf('  log_like_plus_o = %f\n', log_like_plus_o);
        end
        fprintf('  檢查關鍵變量:\n');
        fprintf('    e = %f\n', e);
        fprintf('    d = %f\n', d);
        if exist('v', 'var')
            fprintf('    v''*v = %f\n', v'*v);
        end
        
        % 返回一個懲罰值而不是 -Inf
        log_like = -1e10;
    end
%%
end

% function cplus=kron_multiply_vector(A,B,D)
% nAr=size(A,1); nAc=size(A,2); nr=nAr*size(B,1); nc=size(D,2); nBc=size(B,2); 
% cplus=zeros(nr,nc);
% for ii=1:nc
%     tc=B*reshape(D(:,ii),nBc,nAc)*A';
%     cplus(:,ii)=tc(:);
% end
% 
% end