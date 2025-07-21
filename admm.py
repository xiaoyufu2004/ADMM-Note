import numpy as np
import matplotlib.pyplot as plt

# ---------- 0. 原有函数，唯一改动：把残差历史存进 info ----------
def admm_lasso(A, b, lamb=1.0, rho=1.0, 
               max_iter=1000, abstol=1e-4, reltol=1e-3, 
               verbose=False):
    m, n = A.shape
    Atb = A.T @ b
    L = np.linalg.cholesky(A.T @ A + rho * np.eye(n))
    U = L.T

    x = np.zeros(n)
    z = np.zeros(n)
    u = np.zeros(n)

    r_hist, s_hist = [], []          # ← 新增

    for k in range(max_iter):
        q = Atb + rho * (z - u)
        x = np.linalg.solve(U, np.linalg.solve(L, q))

        z_old = z.copy()
        z      = np.sign(x + u) * np.maximum(np.abs(x + u) - lamb / rho, 0.0)
        u     += x - z

        r_norm = np.linalg.norm(x - z)
        s_norm = np.linalg.norm(-rho * (z - z_old))
        r_hist.append(r_norm)        # ← 新增
        s_hist.append(s_norm)

        eps_pri  = np.sqrt(n) * abstol + reltol * max(np.linalg.norm(x), np.linalg.norm(z))
        eps_dual = np.sqrt(n) * abstol + reltol * np.linalg.norm(rho * u)

        if verbose and (k % 50 == 0 or k == max_iter - 1):
            print(f"iter {k:4d} | r {r_norm:.3e} | s {s_norm:.3e} | "
                  f"eps_pri {eps_pri:.3e} | eps_dual {eps_dual:.3e}")

        if r_norm < eps_pri and s_norm < eps_dual:
            print(f"iter {k:4d} | r {r_norm:.3e} | s {s_norm:.3e} | "
                  f"eps_pri {eps_pri:.3e} | eps_dual {eps_dual:.3e}")            
            break

    info = {
        "iterations": k + 1,
        "r_norm": r_norm,
        "s_norm": s_norm,
        "r_hist": r_hist,
        "s_hist": s_hist,
        "eps_pri": eps_pri,
        "eps_dual": eps_dual
    }
    return x, info

# ---------- 1. 生成一个带稀疏真解的测试集 ----------
np.random.seed(42)
m, n, sparsity = 200, 500, 0.1
A = np.random.randn(m, n)
x_true = np.zeros(n)
support = np.random.choice(n, int(sparsity * n), replace=False)
x_true[support] = np.random.randn(len(support))
b = A @ x_true + 0.01 * np.random.randn(m)

# ---------- 2. 运行 ADMM ----------
x_est, info = admm_lasso(A, b, lamb=0.1, rho=1.0, verbose=True)

# ---------- 3. 作图 ----------
# 3‑1 收敛曲线
plt.figure()
plt.semilogy(info["r_hist"], label="primal r")
plt.semilogy(info["s_hist"], label="dual s")
plt.axhline(info["eps_pri"],  linestyle="--", label="eps_pri")
plt.axhline(info["eps_dual"], linestyle="--", label="eps_dual")
plt.xlabel("Iteration")
plt.ylabel("Residual norm")
plt.title("ADMM convergence")
plt.legend()
plt.grid(True)

# 3‑2 稀疏系数图
plt.figure()
idx = np.arange(n)
plt.stem(idx, x_true, markerfmt=" ", basefmt=" ", label="True")
plt.scatter(idx, x_est, s=8, label="Estimated")
plt.xlabel("Coefficient index")
plt.ylabel("Value")
plt.title("Coefficient sparsity pattern")
plt.legend()

# 3‑3 真值 vs 估计
plt.figure()
plt.scatter(x_true, x_est, s=12)
lims = [min(x_true.min(), x_est.min()), max(x_true.max(), x_est.max())]
plt.plot(lims, lims)        # y = x 参考线
plt.xlabel("True coefficient")
plt.ylabel("Estimated coefficient")
plt.title("Identity check")
plt.grid(True)

plt.show()
