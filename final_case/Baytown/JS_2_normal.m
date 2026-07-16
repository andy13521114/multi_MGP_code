function [x] = JS_2_normal(y,type,eta,gamma,lambda,eps)

if type == 1, % SU distribution
    x = gamma + eta*asinh((y-eps)/lambda);
elseif type == 2, % SB
    x = gamma + eta*log((y-eps)./(lambda+eps-y));
elseif type == 3 % SL
    x = gamma + eta*log(y-eps);
else
    x = y;
end