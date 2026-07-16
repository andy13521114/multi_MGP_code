function [y] = JS_2_original(x,type,eta,gamma,lambda,eps)

if type == 1, % SU distribution
    y = eps + lambda*sinh((x-gamma)/eta);
elseif type == 2, % SB
    y = (eps + (eps+lambda)*exp((x-gamma)/eta))./(1+exp((x-gamma)/eta));
elseif type == 3 % SL
    y = eps + exp((x-gamma)/eta);
else
    y = x;
end