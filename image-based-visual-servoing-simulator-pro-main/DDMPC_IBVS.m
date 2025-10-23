
function [control] = DDMPC_IBVS(dt, qd, q, control_prev, ubound, wbound, lamda, z, umax, vmax, params, t)
% DDMPC_IBVS  (Method 5)  —— Data-Driven MPC + Sequential-Lambda reference
% 方法3对齐版增强：
%   A) ΔU 惩罚 + 历史软约束 + 归一化 + 窗口化（同改进版）
%   B) **序列lambda网格搜索**（base + slope），挑选使“一步预测误差最小”的lambda序列（更像方法3逐步最优）
%   C) **与 IBVS 步骤混合 (gamma_ibvs)**： control = (1-gamma)*u_ddmpc + gamma*u_ibvs
%      其中 u_ibvs = -lambda1 * J^+ * e ，J为当前交互矩阵（用解析式构造），e为未归一化误差
%
% 依然无需CVX；每步只做少量线性方程求解。

persistent U_hist Y_hist y_scale initialized
if isempty(initialized); U_hist = []; Y_hist = []; y_scale = 1.0; initialized = true; end

m = 6; p = 8;  % dims

% ----- 参数 -----
Ni         = getf(params,'Ni',10);
Np         = getf(params,'Np',4);
lambda_g   = getf(params,'lambda_g',1e-3);
lambda_min = getf(params,'lambda_min',0.5);
lambda_max = getf(params,'lambda_max',3.0);
Wy_s       = getf(params,'Wy',1.0);
Wu_s       = getf(params,'Wu',0.05);       % 更贴近方法3的激进度
sat_axis   = getf(params,'sat_per_axis',true);
min_data   = getf(params,'min_data', max(2*(Ni+Np), 24));
Wmax       = getf(params,'Wmax',200);
alpha_c    = getf(params,'alpha_c',5.0);
beta_c     = getf(params,'beta_c',5.0);
norm_y     = getf(params,'normalize',true);
use_l1     = getf(params,'use_lambda_search',true);
gamma_ibvs = getf(params,'blend_ibvs',0.3);   % 与IBVS混合比例（0~1）
nbases     = getf(params,'nbases',6);
nslopes    = getf(params,'nslopes',5);

% ----- 追加历史（执行限幅后入库） -----
if isempty(U_hist)
    U_hist = zeros(m,1);
else
    u_prev = control_prev(:);
    u_prev(1:3) = clamp(u_prev(1:3), -ubound, ubound);
    u_prev(4:6) = clamp(u_prev(4:6), -wbound, wbound);
    U_hist = [U_hist, u_prev];
end

% ----- 归一化输出 -----
if norm_y && isempty(Y_hist)
    y_scale = max(1, median(abs(q(:))));
end
q_n  = q(:)  / y_scale;
qd_n = qd(:) / y_scale;
Y_hist = [Y_hist, q_n];

% 窗口化
if size(U_hist,2) > Wmax
    U_hist = U_hist(:, end-Wmax+1:end);
    Y_hist = Y_hist(:, end-Wmax+1:end);
end

% 数据充足性
T = size(U_hist,2);
L = Ni + Np;
if T < L || T < min_data
    control = ibvs_like_step(qd, q, lamda, z, ubound, wbound, 1.0);  % 预热用 IBVS
    return;
end

% ----- Hankel -----
H_u = build_hankel_local(U_hist, L);
H_y = build_hankel_local(Y_hist, L);
Up = H_u(1:m*Ni, :);   Uf = H_u(m*Ni+1:end, :);
Yp = H_y(1:p*Ni, :);   Yf = H_y(p*Ni+1:end, :);
u_past = vec(U_hist(:, end-Ni+1:end));
y_past = vec(Y_hist(:, end-Ni+1:end));
Nc = size(Up,2); I_g = eye(Nc);

% ----- 权重展开 -----
Wy = expand_weight(Wy_s, p*Np);
Wu = expand_weight(Wu_s, m*Np);

% ----- ΔU 惩罚 -----
dimU = m*Np;
E = eye(dimU);
for r = (m+1):dimU
    E(r, r-m) = -1;
end
Uf_bar = E * Uf;

% ----- 固定的二次项H（与Yref无关） -----
H = 2*( (Yf'*(Wy'*Wy)*Yf) + (Uf_bar'*(Wu'*Wu)*Uf_bar) + (lambda_g * I_g) ...
      + alpha_c*(Up'*Up) + beta_c*(Yp'*Yp) );
eps_reg = 1e-8;
H_reg = H + eps_reg*eye(Nc);

% ----- 序列 λ 网格搜索（base & slope） -----
e_n = q_n - qd_n;

% 网格：base ∈ [λmin, λmax]，slope ∈ [0, (λmax-λmin)/(Np-1)]
base_vec  = linspace(lambda_min, lambda_max, nbases);
max_slope = (lambda_max - lambda_min) / max(1, (Np-1));
slope_vec = linspace(0, max_slope, nslopes);

best_cost = inf;
best_seq  = linspace(lambda_min, lambda_max, Np)';
for b = base_vec
    for s = slope_vec
        lam_seq = b + s*(0:Np-1)';
        lam_seq = min(lambda_max, max(lambda_min, lam_seq));  % clamp
        Yref = zeros(p*Np,1);
        for i = 1:Np
            decay = exp(-lam_seq(i) * (i*dt));
            Yref((i-1)*p+1:i*p) = qd_n + decay * e_n;
        end
        f = 2*( Yf'*(Wy'*Wy)*Yref + alpha_c*(Up'*u_past) + beta_c*(Yp'*y_past) );
        g = H_reg \ f;
        y_future = Yf * g;
        y1 = y_future(1:p);
        % 评估：一步预测误差 + 轻微控制增量代价
        du_future = Uf_bar * g;
        W1 = Wy(1:p, 1:p);
        Jcost = norm(W1*(y1 - qd_n))^2 + 0.05*norm(du_future(1:m))^2;
        if Jcost < best_cost
            best_cost = Jcost;
            best_seq = lam_seq;
        end
    end
end

% ----- 用最优 λ 序列重新求解并取第一步控制 -----
Yref = zeros(p*Np,1);
for i = 1:Np
    decay = exp(-best_seq(i) * (i*dt));
    Yref((i-1)*p+1:i*p) = qd_n + decay * e_n;
end
f = 2*( Yf'*(Wy'*Wy)*Yref + alpha_c*(Up'*u_past) + beta_c*(Yp'*y_past) );
g = H_reg \ f;
u_future = Uf * g;    u_ddmpc = u_future(1:m);

% ----- 与 IBVS 一步混合（增强与方法3一致性） -----
u_ibvs = ibvs_like_step(qd, q, lamda, z, ubound, wbound, best_seq(1));
control = (1 - gamma_ibvs)*u_ddmpc + gamma_ibvs*u_ibvs;

% 轴向限幅
if sat_axis
    control(1:3) = clamp(control(1:3), -ubound, ubound);
    control(4:6) = clamp(control(4:6), -wbound, wbound);
else
    vn = norm(control(1:3)); if vn > ubound, control(1:3) = control(1:3)*(ubound/max(vn,eps)); end
    wn = norm(control(4:6)); if wn > wbound, control(4:6) = control(4:6)*(wbound/max(wn,eps)); end
end

% ----------------- helpers -----------------
function H = build_hankel_local(D, Lb)
    [dim, Tloc] = size(D);
    H = zeros(dim*Lb, Tloc-Lb+1);
    for kcol = 1:(Tloc-Lb+1)
        seg = D(:, kcol:(kcol+Lb-1));
        H(:, kcol) = seg(:);
    end
end

function v = vec(M), v = M(:); end

function val = getf(S, name, dv)
    if isstruct(S) && isfield(S, name) && ~isempty(S.(name)), val = S.(name);
    else, val = dv; end
end

function u = clamp(u, lo, hi), u = min(max(u, lo), hi); end

function u = ibvs_like_step(qd_loc, q_loc, lam, z_loc, ub, wb, lambda1)
    % 构造解析交互矩阵并做一步 IBVS，与方法3的 u=-lambda J^+ e 一致
    J = zeros(8,6);
    qv = q_loc(:);
    for kpt = 1:4
        m_ = qv(2*kpt-1); n_ = qv(2*kpt);
        Zk = max(1e-6, z_loc(kpt));
        Jk = [ -lam/Zk,      0,  m_/Zk,  (m_*n_)/lam, -(lam^2+m_^2)/lam,  n_;
                    0, -lam/Zk,  n_/Zk, (lam^2+n_^2)/lam, -(m_*n_)/lam, -m_ ];
        J(2*kpt-1:2*kpt, :) = Jk;
    end
    e = q_loc(:) - qd_loc(:);
    Jp = pinv(J, 1e-3);
    u = - lambda1 * (Jp * e);
    % 轴限幅
    u(1:3) = clamp(u(1:3), -ub, ub);
    u(4:6) = clamp(u(4:6), -wb, wb);
end

function W = expand_weight(Ws, dim)
    if isscalar(Ws), W = sqrt(Ws) * eye(dim);
    else
        if ~ismatrix(Ws) || any(size(Ws) ~= [dim, dim])
            error('Weight must be scalar or %dx%d matrix.', dim, dim);
        end
        W = Ws;
    end
end

end
