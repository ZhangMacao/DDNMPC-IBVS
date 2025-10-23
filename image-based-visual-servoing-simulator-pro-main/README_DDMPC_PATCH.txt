# DDMPC + Sequential Lambda IBVS Integration (Method 5)

This patch adds a new control method to `MAIN.m`:

- **5 = DD-MPC + Sequential Lambda IBVS**: a DeePC-style data-driven MPC controller that uses historical (u,y) data to predict and optimize camera twist without an explicit model. The reference trajectory over the prediction horizon is shaped by a *sequential lambda* schedule (increasing lambdas) to emulate the good transient behavior of Lambda-based IBVS (sequential).

## Files added
- `DDMPC_IBVS.m`: self-contained controller (no CVX, no extra toolbox). 
  - Fallback to damped IBVS for the warm-up phase until enough data is collected.
  - KKT-based closed-form solver for the DeePC optimization problem with equality constraints.

## Changes in `MAIN.m`
- Documented method 5 in the control method list.
- Added default parameter struct `KDDMPC` right after the existing gains.
- Inserted a new `elseif control_method == 5` branch that calls `DDMPC_IBVS`.

## Parameters
Edit these in `MAIN.m`:

```matlab
KDDMPC = struct('Ni',6,'Np',10,'lambda_g',1e-2, ...
                'lambda_min',0.2,'lambda_max',2.0, ...
                'Wy',1.0,'Wu',1.0,'sat_per_axis',true,'min_data',20);
```
- `Ni`: past horizon (init length)
- `Np`: prediction horizon
- `lambda_min/max`: sequential lambda range (monotone increasing along horizon)
- `Wy`, `Wu`: weighting (scalar or block-diagonal matrices accepted)
- `lambda_g`: Tikhonov on the DeePC decision vector g
- `min_data`: minimal samples before DeePC activates (fallback runs before that)

## Usage
- Open `MAIN.m` and set:
  ```matlab
  control_method = 5;
  ```
- Run as usual. A few first steps will use a safe IBVS fallback while the controller collects data; then it will switch to pure DDMPC.

## Notes
- This integrates with your existing feature vector `q` (8x1) and control twist (6x1).
- Velocity saturations (`ubound`, `wbound`) are respected axis-wise.
- If you prefer an *optimized* per-step lambda (grid search), extend `lambda_seq` creation in `DDMPC_IBVS.m` accordingly (we kept a simple increasing schedule for speed and determinism).