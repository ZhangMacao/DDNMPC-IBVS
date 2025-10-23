
function [control] = DDMPC_IBVS(dt, qd, q, control_prev, ubound, wbound, lamda, z, umax, vmax, params, t)
%DDMPC_IBVS Data-Driven MPC for IBVS with sequential-lambda reference shaping
%
%   This controller uses a DeePC-style data-driven predictive control that
%   builds Hankel matrices from historical (u,y) data and solves a ridge-
%   regularized least-squares with equality constraints (no CVX required).
%   The reference over the horizon is shaped by a sequential-lambda
%   schedule, inspired by Lambda-based IBVS (sequential).
%
% Inputs:
%   dt, qd, q, control_prev, ubound, wbound, lamda, z, umax, vmax: as in other controllers
%   params: struct with fields
%       .Ni        : "initial" past horizon length (# of steps)
%       .Np        : prediction horizon length (# of steps)
%       .Wy, .Wu   : weighting matrices (p*Np x p*Np) and (m*Np x m*Np), or scalars
%       .lambda_g  : Tikhonov regularization on g (scalar)
%       .lambda_min, .lambda_max : range for sequential lambda
%       .sat_per_axis: bool, apply per-axis saturation (default true)
%       .min_data  : minimum samples before DeePC activates (default Ni+Np+5)
%   t     : time index (1-based)
%
% Output:
%   control (6x1): body velocity twist [vx vy vz wx wy wz]^T
%
% Notes:
%   - If there is not enough data to form persistent Hankel blocks, fall
%     back to a damped IBVS (pseudo-Jacobian) step for a few iterations.
%
% Author: ChatGPT (integration patch)
% -------------------------------------------------------------------------

persistent U_hist Y_hist initialized

if isempty(initialized)
    U_hist = [];     % m x T
    Y_hist = [];     % p x T
    initialized = true;
end

% Dimensions
m = 6;   % control: [vx vy vz wx wy wz]
p = 8;   % outputs: [u1 v1 u2 v2 u3 v3 u4 v4]

% Default params
if ~isstruct(params), params = struct(); end
Ni       = getfield_def(params, 'Ni', 6);            %#ok<GFLD>
Np       = getfield_def(params, 'Np', 10);
lambda_g = getfield_def(params, 'lambda_g', 1e-2);
lambda_min = getfield_def(params, 'lambda_min', 0.2);
lambda_max = getfield_def(params, 'lambda_max', 2.0);
sat_per_axis = getfield_def(params, 'sat_per_axis', true);
min_data = getfield_def(params, 'min_data', Ni+Np+5);

Wy_s = getfield_def(params, 'Wy', 1.0);
Wu_s = getfield_def(params, 'Wu', 1.0);

% Expand scalar weights to block-diagonal over horizon later
% ---------------------------------------------------------

% Append latest measurement to histories
% NOTE: at time t we have access to current output q(:,t) and previous input control_prev(:,t-1)
if isempty(U_hist)
    % First call: seed previous control with zeros (no move)
    U_hist = zeros(m, 1);
else
    % Append the last applied control (previous step)
    U_hist = [U_hist, control_prev(:)];
end
Y_hist = [Y_hist, q(:)];

% Check if we have enough data for DeePC
T = size(U_hist, 2);
L = Ni + Np;                  % total Hankel depth
if T < L || T < min_data
    % Fallback: simple damped IBVS step using pseudo-Jacobian based on interaction matrix
    control = ibvs_fallback(qd, q, lamda, z, ubound, wbound);
    return;
end

% Build block Hankel matrices from rolling history
% ------------------------------------------------
% We construct column-wise Hankel matrices H_u and H_y with L block-rows.
H_u = build_hankel_local(U_hist, L);    % (m*L) x (T-L+1)
H_y = build_hankel_local(Y_hist, L);    % (p*L) x (T-L+1)

% Split into past (Ni) and future (Np) blocks
Up = H_u(1:m*Ni, :);        Uf = H_u(m*Ni+1:end, :);        % sizes: (m*Ni) x Ncols, (m*Np) x Ncols
Yp = H_y(1:p*Ni, :);        Yf = H_y(p*Ni+1:end, :);        % sizes: (p*Ni) x Ncols, (p*Np) x Ncols

% Past measurements (last Ni samples)
u_past = vec(U_hist(:, end-Ni+1:end));
y_past = vec(Y_hist(:, end-Ni+1:end));

% Build sequential-lambda reference over horizon
% ----------------------------------------------
e = q(:) - qd(:);                      % current feature error (8x1)
lambda_seq = linspace(lambda_min, lambda_max, Np)';   % simple increasing schedule
Yref = zeros(p*Np,1);
for i = 1:Np
    % Exponential decay toward qd with per-step lambda_i
    decay = exp(-lambda_seq(i) * (i*dt));
    y_ref_i = qd(:) + decay * e;
    Yref( (i-1)*p+1 : i*p ) = y_ref_i;
end

% Expand weights
if isscalar(Wy_s)
    Wy = sqrt(Wy_s) * eye(p*Np);
else
    Wy = Wy_s;
end
if isscalar(Wu_s)
    Wu = sqrt(Wu_s) * eye(m*Np);
else
    Wu = Wu_s;
end

% Form and solve the KKT system for g (no CVX)
% --------------------------------------------
% min_g  ||Wy (Yf g - Yref)||^2 + ||Wu (Uf g)||^2 + lambda_g ||g||^2
% s.t.   Up g = u_past,   Yp g = y_past
% KKT:
% [ H  Up'  Yp'] [g]   = [ f ]
% [ Up 0    0  ] [mu1]   [ u_past ]
% [ Yp 0    0  ] [mu2]   [ y_past ]
%
% where H = 2*(Yf'Wy'Wy*Yf + Uf'Wu'Wu*Uf + lambda_g*I), f = 2*Yf'Wy'Wy*Yref

Nc = size(Up,2);            % number of Hankel columns
I_g = eye(Nc);

H = 2*( (Yf'*(Wy'*Wy)*Yf) + (Uf'*(Wu'*Wu)*Uf) + (lambda_g * I_g) );
f = 2*( Yf'*(Wy'*Wy)*Yref );

KKT = [ H, Up', Yp';
        Up, zeros(size(Up,1)), zeros(size(Up,1), size(Yp,1));
        Yp, zeros(size(Yp,1), size(Up,1)), zeros(size(Yp,1)) ];
rhs = [ f; u_past; y_past ];

% Solve robustly using \ with a small Tikhonov for numerical stability
eps_reg = 1e-8;
KKT = KKT + eps_reg * eye(size(KKT));
sol = KKT \ rhs;
g = sol(1:Nc);

% Recover future optimal sequences
u_future = Uf * g;        % (m*Np) x 1
% y_future = Yf * g;      % (p*Np) x 1   % not needed explicitly here

% First control move (receding horizon)
u1 = u_future(1:m);

% Velocity saturation
control = u1(:);
if sat_per_axis
    % Translational
    for j = 1:3
        control(j) = min(max(control(j), -ubound), ubound);
    end
    % Rotational
    for j = 4:6
        control(j) = min(max(control(j), -wbound), wbound);
    end
else
    % Norm-based saturation (optional)
    v_norm = norm(control(1:3));
    w_norm = norm(control(4:6));
    if v_norm > ubound
        control(1:3) = control(1:3) * (ubound / max(v_norm, eps));
    end
    if w_norm > wbound
        control(4:6) = control(4:6) * (wbound / max(w_norm, eps));
    end
end

% -------------------------------------------------------------------------
% Local helpers
% -------------------------------------------------------------------------
function H = build_hankel_local(D, Lb)
    % D: (dim x T)
    [dim, Tloc] = size(D);
    H = zeros(dim*Lb, Tloc-Lb+1);
    for kcol = 1:(Tloc-Lb+1)
        seg = D(:, kcol:(kcol+Lb-1));
        H(:, kcol) = seg(:);
    end
end

function v = vec(M)
    v = M(:);
end

function val = getfield_def(S, fname, default_val)
    if isstruct(S) && isfield(S, fname) && ~isempty(getfield(S, fname))
        val = getfield(S, fname);
    else
        val = default_val;
    end
end

function u = ibvs_fallback(qd_loc, q_loc, lam, z_loc, ub, wb)
    % A tiny, safe fallback: damped pseudo-inverse of interaction matrix
    % Build interaction matrices per point and stack
    J = zeros(8,6);
    qv = q_loc(:);
    for kpt = 1:4
        m = qv(2*kpt-1);
        n = qv(2*kpt);
        Zk = z_loc(kpt);
        Jk = [ -lam/Zk,      0,  m/Zk,  (m*n)/lam, -(lam^2+m^2)/lam,  n;
                    0, -lam/Zk,  n/Zk, (lam^2+n^2)/lam, -(m*n)/lam, -m ];
        J(2*kpt-1:2*kpt, :) = Jk;
    end
    e = q_loc(:) - qd_loc(:);
    Jp = pinv(J, 1e-3);
    u = - Jp * e;
    % Saturate conservatively
    for j = 1:3
        u(j) = min(max(u(j), -0.5*ub), 0.5*ub);
    end
    for j = 4:6
        u(j) = min(max(u(j), -0.5*wb), 0.5*wb);
    end
end

end