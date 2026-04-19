import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import minimize

try:
    import cvxpy as cp
    HAS_CVXPY = True
except Exception:
    HAS_CVXPY = False


# -----------------------------
# Helper functions
# -----------------------------
def soft_threshold(v: np.ndarray, kappa: float) -> np.ndarray:
    return np.sign(v) * np.maximum(np.abs(v) - kappa, 0.0)


def objective(A: np.ndarray, b: np.ndarray, x: np.ndarray, lam: float) -> float:
    r = A @ x - b
    return 0.5 * float(r.T @ r) + lam * np.linalg.norm(x, 1)


def admm_lasso(
    A: np.ndarray,
    b: np.ndarray,
    lam: float,
    rho: float,
    max_iter: int = 1000,
    abstol: float = 1e-4,
    reltol: float = 1e-3,
):
    m, n = A.shape
    x = np.zeros(n)
    z = np.zeros(n)
    u = np.zeros(n)

    AtA = A.T @ A
    Atb = A.T @ b
    M = AtA + rho * np.eye(n)

    primal_residuals = []
    dual_residuals = []
    objective_values = []

    for k in range(max_iter):
        x = np.linalg.solve(M, Atb + rho * (z - u))
        z_old = z.copy()
        z = soft_threshold(x + u, lam / rho)
        u = u + x - z

        r = x - z
        s = rho * (z - z_old)
        primal_residuals.append(np.linalg.norm(r))
        dual_residuals.append(np.linalg.norm(s))
        objective_values.append(objective(A, b, x, lam))

        eps_pri = np.sqrt(n) * abstol + reltol * max(np.linalg.norm(x), np.linalg.norm(z))
        eps_dual = np.sqrt(n) * abstol + reltol * np.linalg.norm(rho * u)
        if np.linalg.norm(r) <= eps_pri and np.linalg.norm(s) <= eps_dual:
            break

    return {
        "x": x,
        "z": z,
        "u": u,
        "iters": k + 1,
        "primal_residuals": np.array(primal_residuals),
        "dual_residuals": np.array(dual_residuals),
        "objective_values": np.array(objective_values),
        "final_objective": objective(A, b, z, lam),
    }


def solve_reference(A: np.ndarray, b: np.ndarray, lam: float):
    n = A.shape[1]

    if HAS_CVXPY:
        x = cp.Variable(n)
        problem = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(A @ x - b) + lam * cp.norm1(x)))
        problem.solve(solver=cp.SCS, verbose=False)
        if x.value is None:
            raise RuntimeError("CVXPY failed to produce a solution.")
        return np.array(x.value).reshape(-1), "CVXPY"

    # Fallback: solve equivalent smooth constrained problem with scipy
    # Variables are w = [x; t], with constraints -t <= x <= t and t >= 0.
    def fun(w):
        x = w[:n]
        t = w[n:]
        r = A @ x - b
        return 0.5 * float(r @ r) + lam * np.sum(t)

    def grad(w):
        x = w[:n]
        r = A @ x - b
        g_x = A.T @ r
        g_t = lam * np.ones(n)
        return np.concatenate([g_x, g_t])

    constraints = []
    for i in range(n):
        constraints.append({
            "type": "ineq",
            "fun": lambda w, i=i: w[n + i] - w[i],
            "jac": lambda w, i=i: _jac_constraint_plus(n, i),
        })
        constraints.append({
            "type": "ineq",
            "fun": lambda w, i=i: w[n + i] + w[i],
            "jac": lambda w, i=i: _jac_constraint_minus(n, i),
        })
        constraints.append({
            "type": "ineq",
            "fun": lambda w, i=i: w[n + i],
            "jac": lambda w, i=i: _jac_constraint_t(n, i),
        })

    w0 = np.zeros(2 * n)
    result = minimize(fun, w0, jac=grad, constraints=constraints, method="SLSQP", options={"maxiter": 2000, "ftol": 1e-9})
    if not result.success:
        raise RuntimeError(f"Reference solver failed: {result.message}")
    return result.x[:n], "SciPy SLSQP fallback"


def _jac_constraint_plus(n, i):
    g = np.zeros(2 * n)
    g[i] = -1.0
    g[n + i] = 1.0
    return g


def _jac_constraint_minus(n, i):
    g = np.zeros(2 * n)
    g[i] = 1.0
    g[n + i] = 1.0
    return g


def _jac_constraint_t(n, i):
    g = np.zeros(2 * n)
    g[n + i] = 1.0
    return g


def sanitize_rho(rho: float) -> str:
    return str(rho).replace('.', 'p')


# -----------------------------
# Main experiment
# -----------------------------
def main():
    outdir = Path(".")
    rng = np.random.default_rng(602)

    n = 256
    m = 128

    A = rng.standard_normal((m, n))
    col_norms = np.linalg.norm(A, axis=0)
    col_norms[col_norms == 0] = 1.0
    A = A / col_norms

    x_true = np.zeros(n)
    support = rng.choice(n, size=20, replace=False)
    x_true[support] = rng.standard_normal(20)

    sigma = 0.01 * np.linalg.norm(A @ x_true) / np.sqrt(2 * m)
    eps = sigma * rng.standard_normal(m)
    b = A @ x_true + eps

    lam = 0.1 * np.linalg.norm(A.T @ b, ord=np.inf)

    print("Data generation summary")
    print(f"n = {n}, m = {m}")
    print(f"sigma = {sigma:.8f}")
    print(f"lambda = {lam:.8f}")
    print(f"support size of x_true = {np.count_nonzero(x_true)}")

    rho_values = [0.1, 1.0]
    admm_results = {}

    for rho in rho_values:
        result = admm_lasso(A, b, lam, rho, max_iter=2000)
        admm_results[rho] = result
        tag = sanitize_rho(rho)

        plt.figure(figsize=(7, 4.5))
        plt.semilogy(result["primal_residuals"], label=r"$\|r^k\|_2$")
        plt.semilogy(result["dual_residuals"], label=r"$\|s^k\|_2$")
        plt.xlabel("Iteration")
        plt.ylabel("Residual norm")
        plt.title(f"ADMM residuals for rho = {rho}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / f"residuals_rho_{tag}.png", dpi=200)
        plt.close()

    x_ref, ref_name = solve_reference(A, b, lam)

    x_admm_best = admm_results[1.0]["z"]
    plt.figure(figsize=(8, 4.5))
    markerline1, stemlines1, baseline1 = plt.stem(x_true, linefmt='-', markerfmt='o', basefmt=' ')
    markerline2, stemlines2, baseline2 = plt.stem(x_admm_best, linefmt='--', markerfmt='s', basefmt=' ')
    plt.xlabel("Index")
    plt.ylabel("Coefficient value")
    plt.title("True signal and ADMM reconstruction (rho = 1.0)")
    plt.legend([markerline1, markerline2], ["x_true", "ADMM solution"])
    plt.tight_layout()
    plt.savefig(outdir / "coefficients_comparison.png", dpi=200)
    plt.close()

    summary_path = outdir / "admm_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("ADMM Sparse Recovery Summary\n")
        f.write("============================\n")
        f.write(f"Reference solver = {ref_name}\n")
        f.write(f"n = {n}, m = {m}\n")
        f.write(f"sigma = {sigma:.10f}\n")
        f.write(f"lambda = {lam:.10f}\n\n")
        for rho in rho_values:
            res = admm_results[rho]
            x_admm = res["z"]
            f.write(f"rho = {rho}\n")
            f.write(f"  iterations = {res['iters']}\n")
            f.write(f"  objective (ADMM) = {res['final_objective']:.10f}\n")
            f.write(f"  ||x_admm - x_ref||_2 = {np.linalg.norm(x_admm - x_ref):.10f}\n")
            f.write(f"  ||x_admm - x_true||_2 = {np.linalg.norm(x_admm - x_true):.10f}\n")
            f.write(f"  nnz(x_admm, tol=1e-4) = {np.count_nonzero(np.abs(x_admm) > 1e-4)}\n")
            f.write(f"  final primal residual = {res['primal_residuals'][-1]:.10e}\n")
            f.write(f"  final dual residual   = {res['dual_residuals'][-1]:.10e}\n\n")
        f.write(f"{ref_name} reference\n")
        f.write(f"  objective (reference) = {objective(A, b, x_ref, lam):.10f}\n")
        f.write(f"  ||x_ref - x_true||_2 = {np.linalg.norm(x_ref - x_true):.10f}\n")
        f.write(f"  nnz(x_ref, tol=1e-4) = {np.count_nonzero(np.abs(x_ref) > 1e-4)}\n")

    print("\nComparison summary")
    print(f"Reference solver          = {ref_name}")
    for rho in rho_values:
        res = admm_results[rho]
        x_admm = res["z"]
        print(f"rho = {rho}")
        print(f"  iterations              = {res['iters']}")
        print(f"  objective (ADMM)        = {res['final_objective']:.10f}")
        print(f"  ||x_admm - x_ref||_2    = {np.linalg.norm(x_admm - x_ref):.10f}")
        print(f"  ||x_admm - x_true||_2   = {np.linalg.norm(x_admm - x_true):.10f}")
        print(f"  final primal residual   = {res['primal_residuals'][-1]:.10e}")
        print(f"  final dual residual     = {res['dual_residuals'][-1]:.10e}")
    print(f"Reference objective       = {objective(A, b, x_ref, lam):.10f}")
    print("\nSaved files:")
    print("  residuals_rho_0p1.png")
    print("  residuals_rho_1p0.png")
    print("  coefficients_comparison.png")
    print("  admm_summary.txt")


if __name__ == "__main__":
    main()
