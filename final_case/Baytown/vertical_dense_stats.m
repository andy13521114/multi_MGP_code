function [E_X,L_h,L_z] = vertical_dense_stats(sofv,sofh,nuv,nuh,X_ele,Y_ele, z_ele,X,Y,z,t_d,M)
nXele = length(X_ele); nYele = length(Y_ele); nzele = length(z_ele);
X_o = X; Y_o = Y; z_o = z; nXo = length(X_o); nYo = length(Y_o); nzo = length(z_o);
temp_x = abs(X_ele*ones(1,nXo)-(X_o*ones(1,nXele))');
temp_y = abs(Y_ele*ones(1,nYo)-(Y_o*ones(1,nYele))'); 
temp_z = abs(z_ele*ones(1,nzo)-(z_o*ones(1,nzele))'); temp_h = sqrt(temp_x.^2+temp_y.^2);
R_h = Matern_R(nuh,sofh,temp_h); R_z = Matern_R(nuv,sofv,temp_z);
temp_xo = abs(X_o*ones(1,nXo)-(X_o*ones(1,nXo))');
temp_yo = abs(Y_o*ones(1,nYo)-(Y_o*ones(1,nYo))'); 
temp_zo = abs(z_o*ones(1,nzo)-(z_o*ones(1,nzo))'); temp_ho = sqrt(temp_xo.^2+temp_yo.^2);
R_ho = Matern_R(nuh,sofh,temp_ho); R_zo = Matern_R(nuv,sofv,temp_zo);
R_zo = R_zo + 1e-10*eye(nzo); R_ho = R_ho + 1e-10*eye(nXo); invR_zo = inv(R_zo); invR_ho = inv(R_ho);
temp_xele = abs(X_ele*ones(1,nXele)-(X_ele*ones(1,nXele))');
temp_yele = abs(Y_ele*ones(1,nYele)-(Y_ele*ones(1,nYele))'); 
temp_zele = abs(z_ele*ones(1,nzele)-(z_ele*ones(1,nzele))'); temp_hele = sqrt(temp_xele.^2+temp_yele.^2);
R_hele = Matern_R(nuh,sofh,temp_hele) + 1e-10*eye(nXele); R_zele = Matern_R(nuv,sofv,temp_zele) + 1e-10*eye(nzele);
L_h = chol(R_hele-R_h*invR_ho*R_h','lower'); L_h = real(L_h); 
%L_z = chol(R_zele-R_z*invR_zo*R_z','lower');
L_z = chol(R_zele,'lower');
%L_z = real(L_z);
AA = kron(R_h*invR_ho,R_z*invR_zo);
% SIGMA = kron(R_h,R_z)';
% AAA = AA * SIGMA; 

E_X=AA *reshape(t_d,nzo*nXo,M); 
E_X=E_X(:);
1;
