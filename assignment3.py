import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
import cvxpy as cp


def soft_threshold(x, tau):
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0.0)


def prox_quadratic_l1(x, q, lam):
    return soft_threshold(x, lam) / (1.0 + q)


def objective(A, b, x, lam):
    r = A @ x - b
    return 0.5 * np.dot(r, r) + lam * np.linalg.norm(x, 1)


def ensure_figures_dir():
    os.makedirs("figures", exist_ok=True)


def save_current_figure(filename):
    plt.savefig(os.path.join("figures", filename), dpi=300, bbox_inches="tight")
    plt.close()


def verify_firm_nonexpansiveness():
    n = 20
    rng = np.random.default_rng(0)
    q = 0.5 + 2.0 * rng.random(n)
    lam = 0.2 + 0.8 * rng.random(n)

    num_trials = 50
    lhs_vals = []
    rhs_vals = []
    violations = 0

    for _ in range(num_trials):
        x = rng.standard_normal(n)
        y = rng.standard_normal(n)

        px = prox_quadratic_l1(x, q, lam)
        py = prox_quadratic_l1(y, q, lam)

        lhs = np.linalg.norm(px - py) ** 2
        rhs = np.dot(px - py, x - y)

        lhs_vals.append(lhs)
        rhs_vals.append(rhs)

        if lhs > rhs + 1e-10:
            violations += 1

    print("Problem 4(d): Total violations =", violations)

    max_val = max(max(lhs_vals), max(rhs_vals))
    plt.figure(figsize=(6, 6))
    plt.scatter(rhs_vals, lhs_vals)
    plt.plot([0, max_val], [0, max_val])
    plt.xlabel("RHS")
    plt.ylabel("LHS")
    plt.title("Firm Non-Expansiveness Verification")
    plt.grid(True)
    save_current_figure("prox_4d_verification.png")


def prox_grad(A, b, lam, t, maxit, x0=None, xtrue=None):
    n = A.shape[1]
    x = np.zeros(n) if x0 is None else x0.copy()

    obj_hist = []
    err_hist = []

    for _ in range(maxit):
        grad = A.T @ (A @ x - b)
        x = soft_threshold(x - t * grad, t * lam)

        obj_hist.append(objective(A, b, x, lam))
        if xtrue is not None:
            err_hist.append(np.linalg.norm(x - xtrue))

    return x, np.array(obj_hist), np.array(err_hist)


def fista(A, b, lam, t, maxit, x0=None, xtrue=None):
    n = A.shape[1]
    x_prev = np.zeros(n) if x0 is None else x0.copy()
    x = x_prev.copy()

    obj_hist = []
    err_hist = []

    for k in range(maxit):
        alpha = 0.0 if k == 0 else (k - 1) / (k + 2)
        y = x + alpha * (x - x_prev)

        grad = A.T @ (A @ y - b)
        x_next = soft_threshold(y - t * grad, t * lam)

        x_prev = x
        x = x_next

        obj_hist.append(objective(A, b, x, lam))
        if xtrue is not None:
            err_hist.append(np.linalg.norm(x - xtrue))

    return x, np.array(obj_hist), np.array(err_hist)


def build_blur_matrix(n=256, k=9, sigma=2.0):
    c = (k + 1) / 2
    j = np.arange(1, k + 1)
    h = np.exp(-((j - c) ** 2) / (2 * sigma ** 2))
    h = h / h.sum()

    col = np.zeros(n)
    row = np.zeros(n)
    center = int(c) - 1

    col[:k-center] = h[center:]
    row[0] = h[center]
    row[1:center+1] = h[center-1::-1]

    A = toeplitz(col, row)
    return A, h


def run_sparse_deblurring():
    rng = np.random.default_rng(0)
    n = 256

    s = rng.integers(15, 26)
    xtrue = np.zeros(n)
    idx = rng.choice(n, size=s, replace=False)
    xtrue[idx] = rng.standard_normal(s)

    A, _ = build_blur_matrix(n=n, k=9, sigma=2.0)

    Axtrue = A @ xtrue
    sigma_noise = 0.01 * np.linalg.norm(Axtrue) / np.sqrt(n)
    eps = sigma_noise * rng.standard_normal(n)
    b = Axtrue + eps

    lam = 0.05
    L = np.linalg.norm(A, 2) ** 2
    t = 1.0 / L
    maxit = 300

    x_pg, obj_pg, err_pg = prox_grad(A, b, lam, t, maxit, xtrue=xtrue)
    x_fi, obj_fi, err_fi = fista(A, b, lam, t, maxit, xtrue=xtrue)

    x_var = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(A @ x_var - b) + lam * cp.norm1(x_var)))
    prob.solve(verbose=False)

    x_cvx = x_var.value
    obj_cvx = objective(A, b, x_cvx, lam)
    err_cvx = np.linalg.norm(x_cvx - xtrue)

    print("Problem 5 results")
    print("Nonzeros in xtrue:", s)
    print("sigma_noise:", sigma_noise)
    print("L:", L)
    print("step size t:", t)
    print("Final objective PG   :", obj_pg[-1])
    print("Final objective FISTA:", obj_fi[-1])
    print("Final objective CVX  :", obj_cvx)
    print("Final error PG   :", err_pg[-1])
    print("Final error FISTA:", err_fi[-1])
    print("Final error CVX  :", err_cvx)

    plt.figure(figsize=(8, 4))
    plt.plot(xtrue)
    plt.title("True Sparse Signal")
    plt.xlabel("Index")
    plt.ylabel("Amplitude")
    plt.grid(True)
    save_current_figure("true_signal.png")

    plt.figure(figsize=(8, 4))
    plt.plot(b)
    plt.title("Blurred and Noisy Observation")
    plt.xlabel("Index")
    plt.ylabel("Amplitude")
    plt.grid(True)
    save_current_figure("observed_signal.png")

    plt.figure(figsize=(8, 4))
    plt.plot(x_pg)
    plt.title("Reconstruction by Proximal Gradient")
    plt.xlabel("Index")
    plt.ylabel("Amplitude")
    plt.grid(True)
    save_current_figure("reconstruction_pg.png")

    plt.figure(figsize=(8, 4))
    plt.plot(x_fi)
    plt.title("Reconstruction by FISTA")
    plt.xlabel("Index")
    plt.ylabel("Amplitude")
    plt.grid(True)
    save_current_figure("reconstruction_fista.png")

    plt.figure(figsize=(8, 4))
    plt.plot(obj_pg, label="Proximal Gradient")
    plt.plot(obj_fi, label="FISTA")
    plt.axhline(obj_cvx, linestyle="--", label="CVXPY optimum")
    plt.title("Objective Value vs Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Objective")
    plt.legend()
    plt.grid(True)
    save_current_figure("objective_plot.png")

    plt.figure(figsize=(8, 4))
    plt.plot(err_pg, label="Proximal Gradient")
    plt.plot(err_fi, label="FISTA")
    plt.axhline(err_cvx, linestyle="--", label="CVXPY error")
    plt.title("Reconstruction Error vs Iteration")
    plt.xlabel("Iteration")
    plt.ylabel(r"$||x^k - x_{true}||_2$")
    plt.legend()
    plt.grid(True)
    save_current_figure("error_plot.png")


if __name__ == "__main__":
    ensure_figures_dir()
    verify_firm_nonexpansiveness()
    run_sparse_deblurring()
