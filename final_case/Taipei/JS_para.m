function [type,ax,bx,ay,by] = JS_para(y)
%% estimate the Johnson parameters of y
z = 0.7;
y_3z = prctile(y,normcdf(-3*z)*100); y_z = prctile(y,normcdf(-z)*100);
yz = prctile(y,normcdf(z)*100); y3z = prctile(y,normcdf(3*z)*100);
m = y3z - yz; n = y_z - y_3z; p = yz - y_z;

if m*n/p^2 > 1;%1.5, % SU distribution
    type = 1;
    ax = 2*z/acosh(0.5*(m/p+n/p));
    bx = ax*asinh((n/p-m/p)/2/(m*n/p^2-1)^0.5);
    ay = 2*p*(m*n/p^2-1)^0.5/(m/p+n/p-2)/(m/p+n/p+2)^0.5;
    by = (yz+y_z)/2+p*(n/p-m/p)/2/(m/p+n/p-2);
elseif m*n/p^2 < 1;%0.8, % SB
    type = 2;
    ax = z/acosh(0.5*((1+p/m)*(1+p/n))^0.5); 
    bx = ax*asinh((p/n-p/m)*((1+p/m)*(1+p/n)-4)^0.5/2/(p^2/m/n-1));
    ay = p*(((1+p/m)*(1+p/n)-2)^2-4)^0.5/(p^2/m/n-1);
    by = (yz+y_z)/2-ay/2+p*(p/n-p/m)/2/(p^2/m/n-1);
else % SL
    type = 3;
    ax = 2*z/log(m/p); 
    bx = ax*log((m/p-1)/p/(m/p)^0.5);
    by = (yz+y_z)/2-p/2*(m/p+1)/(m/p-1);
    ay = [];
end