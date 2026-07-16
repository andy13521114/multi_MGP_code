function R = Matern_R(nu,sof,d_m)

c = sqrt(2*pi)*gamma(nu+0.5)/(sqrt(nu)*gamma(nu)); s = sof/c;
R = 2^(1-nu)/gamma(nu)*((sqrt(2*nu)*abs(d_m)/s).^nu).*besselk(nu,sqrt(2*nu)*abs(d_m)/s);
R(isnan(R)) = 1;
