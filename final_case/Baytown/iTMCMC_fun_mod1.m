%0702 導入Cs
function [x,ln_S,log_like,stage_n,reject_rate] = iTMCMC_fun_mod1(log_like_fun,y,x_low,x_up,N,b,Cs)
% rand('state',10000*sum(clock));randn('state',10000*sum(clock));
% rand('state',1);randn('state',1);
p = 0; ln_S = 0;
nnn = length(x_low);
x = zeros(N,nnn);log_like = zeros(1,N);
for i=1:nnn
    x(:,i) = x_low(i) + (x_up(i)-x_low(i))*rand(N,1);
end

for i = 1:N
    log_like(i) = feval(log_like_fun,x(i,:),y,Cs);

end
stage_n = 1;reject_count = 0;
while p<1  ,disp(p)
    stage_n = stage_n +1;
    low_p = p; up_p = 2; old_p = p;
    while up_p-low_p > 1e-6
        current_p = (low_p + up_p)/2;
        temp = exp((current_p-p)*(log_like-max(log_like)));
        cov_temp = std(temp)/mean(temp);
        if cov_temp > 1  %0718改了這個
            up_p = current_p;
        else
            low_p = current_p;
        end
    end
    p = current_p; % p =2;
    if p > 1, break; end
    weight = temp/sum(temp);
    ln_S = ln_S+log(mean(temp))+(p-old_p)*max(log_like);
    mu_x = weight*x; sigma = zeros(nnn,nnn);
    for i=1:N
        sigma = sigma + b^2*weight(i)*(x(i,:)-mu_x)'*(x(i,:)-mu_x);
    end
    sqrt_s = sqrtm(sigma);
%     sam_ind = deterministicR((1:N),weight'); 
    current_x = x; current_log_like = log_like;
    for i=1:N
        now_ind = min(find(cumsum(weight)>rand));
        x_c = current_x(now_ind,:) + (sqrt_s*randn(nnn,1))'; x_c = real(x_c);
        if sum(x_c < x_up)==nnn && sum(x_c > x_low)==nnn
            log_like_c = feval(log_like_fun,x_c,y,Cs);
            r = exp(p*(log_like_c - current_log_like(now_ind)));
        else
            r = 0;
        end
        if r > rand%  & sum(x_c < x_up)==nnn & sum(x_c > x_low)==nnn,
            x(i,:) = x_c; current_x(now_ind,:) = x_c; current_log_like(now_ind) = log_like_c; log_like(i) = log_like_c;
            temp = exp((p-old_p)*(current_log_like-max(current_log_like))); weight = temp/sum(temp);
        else
            x(i,:) = current_x(now_ind,:); log_like(i) = current_log_like(now_ind);
            reject_count = reject_count + 1;
        end
    end
end
temp = exp((1-old_p)*(log_like-max(log_like)));
weight = temp/sum(temp);
ln_S = ln_S+log(mean(temp))+(1-old_p)*max(log_like);
mu_x = weight*x; sigma = zeros(nnn,nnn);
for i=1:N
    sigma = sigma + b^2*weight(i)*(x(i,:)-mu_x)'*(x(i,:)-mu_x);
end
sigma = (sigma + sigma.')/2;             % 強制對稱
sigma(~isfinite(sigma)) = 0;             % 清掉 NaN/Inf（常見於數值累積誤差）
sigma = sigma + 1e-10*eye(size(sigma));  % 軟正定，避免半正定/零特徵值
sqrt_s = sqrtm(sigma);
% sam_ind = deterministicR((1:N),weight'); 
current_x = x; current_log_like = log_like;
for i=1:N
    now_ind = min(find(cumsum(weight)>rand));
    x_c = current_x(now_ind,:) + (sqrt_s*randn(nnn,1))'; x_c = real(x_c);
    if sum(x_c < x_up,'all')==nnn && sum(x_c > x_low,'all')==nnn
        log_like_c = feval(log_like_fun,x_c,y,Cs);
        r = exp(1*(log_like_c - current_log_like(now_ind)));
    else
        r = 0;
    end
    if r > rand% & sum(x_c < x_up)==nnn & sum(x_c > x_low)==nnn,
        x(i,:) = x_c; current_x(now_ind,:) = x_c; current_log_like(now_ind) = log_like_c; log_like(i) = log_like_c;
        temp = exp((1-old_p)*(current_log_like-max(current_log_like))); weight = temp/sum(temp);
    else
        x(i,:) = current_x(now_ind,:); log_like(i) = current_log_like(now_ind);
        reject_count = reject_count + 1;
    end
end
reject_rate = reject_count/(stage_n*N);