import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd

# =========================
# 1. Lorenz 系の定義
# =========================
sigma = 10.0
rho   = 28.0
beta  = 8.0 / 3.0

def lorenz(t, xyz):
    x, y, z = xyz
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return [dx, dy, dz]

# =========================
# 2. 時間・初期条件
# =========================
t0, t1 = 0.0, 20.0
dt = 0.01
t_eval = np.arange(t0, t1, dt)

x0 = [1.0, 1.0, 1.0]

# =========================
# 3. 真の軌道を計算
# =========================
sol = solve_ivp(lorenz, (t0, t1), x0, t_eval=t_eval)
t = sol.t
X = sol.y.T   # shape (T, 3): columns -> x, y, z

# 簡単な時系列プロット
plt.figure()
plt.plot(t, X[:, 0], label='x')
plt.plot(t, X[:, 1], label='y')
plt.plot(t, X[:, 2], label='z')
plt.legend()
plt.xlabel('t')
plt.ylabel('state')
plt.show()

# =========================
# 4. 数値微分 dX/dt
# =========================
dXdt = np.zeros_like(X)
dXdt[1:-1, :] = (X[2:, :] - X[:-2, :]) / (2 * dt)
dXdt[0, :]    = (X[1, :] - X[0, :]) / dt
dXdt[-1, :]   = (X[-1, :] - X[-2, :]) / dt

# =========================
# 5. ライブラリ Θ(X) の構築
# =========================
def build_library(X):
    x = X[:, 0]
    y = X[:, 1]
    z = X[:, 2]
    ones = np.ones_like(x)
    Theta = np.column_stack([
        ones,
        x, y, z,
        x**2, y**2, z**2,
        x*y, x*z, y*z
    ])
    return Theta

Theta = build_library(X)

# =========================
# 6. 安全な eta の自動計算（ISTA のステップ幅）
# =========================
L = np.max(np.linalg.eigvals(Theta.T @ Theta)).real
eta_auto = 1.0 / L
print(f"Suggested eta: {eta_auto}")

# =========================
# 7. SINDy（ISTA + soft-threshold）
# =========================
def soft_threshold(v, thr):
    z = np.zeros_like(v)
    z[v >  thr] = v[v >  thr] - thr
    z[v < -thr] = v[v < -thr] + thr
    return z

def sparse_regression(Theta, y, lam=0.01, eta=1e-3, n_iter=1000):
    # minimize 0.5||Theta*w - y||^2 + lam*||w||_1
    M, K = Theta.shape
    w = np.linalg.lstsq(Theta, y, rcond=None)[0]  # 初期値: 最小二乗解
    for _ in range(n_iter):
        grad = Theta.T @ (Theta @ w - y)
        w = soft_threshold(w - eta * grad, lam * eta)
    return w

# =========================
# 8. 係数行列 Xi の推定
# =========================
Xi = np.zeros((Theta.shape[1], 3))  # 10 x 3

# 上で求めた eta_auto を使う例（あるいは少し大きくした値でも可）
for i in range(3):
    Xi[:, i] = sparse_regression(
        Theta,
        dXdt[:, i],
        lam=0.01,
        eta=eta_auto,
        n_iter=5000
    )

# 結果の表示
terms = ["1", "x", "y", "z", "x^2", "y^2", "z^2", "xy", "xz", "yz"]
df = pd.DataFrame(Xi, index=terms, columns=['dx/dt', 'dy/dt', 'dz/dt'])
print(df.mask(np.abs(df) < 1e-2, 0))  # 小さい値を 0 として表示

print("Xi shape:", Xi.shape)
print("Xi =\n", Xi)

# =========================
# 9. SINDy から得たモデルで Lorenz を再構成
# =========================
def lorenz_sindy(t, xyz, Xi):
    x, y, z = xyz
    Theta_row = np.array([
        1.0,
        x, y, z,
        x**2, y**2, z**2,
        x*y, x*z, y*z
    ])
    dx, dy, dz = Theta_row @ Xi  # shape (3,)
    return [dx, dy, dz]

sol_sindy = solve_ivp(
    lambda t, xyz: lorenz_sindy(t, xyz, Xi),
    (t0, t1), x0, t_eval=t_eval
)
X_sindy = sol_sindy.y.T

# =========================
# 10. 軌道の比較プロット
# =========================
plt.figure()
plt.plot(X[:, 0],      X[:, 2], 'b-', label='true',  alpha=1.0)
plt.plot(X_sindy[:, 0], X_sindy[:, 2], 'r-', label='sindy', alpha=0.8)
plt.legend()
plt.xlabel('x')
plt.ylabel('z')
plt.show()
