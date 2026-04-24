import os
import time
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. Setup
# ============================================================

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

np.random.seed(42)


# ============================================================
# 2. Data Generation
# ============================================================

def generate_sparse_signal(n=512, k=40):
    x_true = np.zeros(n)
    support = np.random.choice(n, k, replace=False)
    x_true[support] = np.random.randn(k)
    return x_true, support


def generate_measurements(x, m=200, noise_std=0.01):
    n = len(x)
    A = np.random.randn(m, n) / np.sqrt(m)
    y = A @ x + noise_std * np.random.randn(m)
    return A, y


# ============================================================
# 3. GPSR Solver (Simplified)
# ============================================================

def soft_threshold(x, tau):
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0.0)


def objective(A, x, y, tau):
    r = A @ x - y
    return 0.5 * np.dot(r, r) + tau * np.sum(np.abs(x))


def gpsr(A, y, tau=0.1, max_iter=200, step_size=1e-2):
    n = A.shape[1]
    x = np.zeros(n)

    obj_history = []
    time_history = []

    start_time = time.time()

    for k in range(max_iter):
        grad = A.T @ (A @ x - y)

        # gradient + soft threshold (proximal step)
        x = soft_threshold(x - step_size * grad, tau * step_size)

        obj = objective(A, x, y, tau)
        obj_history.append(obj)
        time_history.append(time.time() - start_time)

    return x, np.array(obj_history), np.array(time_history)


# ============================================================
# 4. Debiasing
# ============================================================

def debias(A, y, x):
    support = np.where(np.abs(x) > 1e-6)[0]

    if len(support) == 0:
        return x

    A_sub = A[:, support]
    x_sub = np.linalg.lstsq(A_sub, y, rcond=None)[0]

    x_debiased = np.zeros_like(x)
    x_debiased[support] = x_sub

    return x_debiased


# ============================================================
# 5. Continuation
# ============================================================

def gpsr_continuation(A, y, tau_values):
    x = np.zeros(A.shape[1])

    all_obj = []
    all_tau = []

    for tau in tau_values:
        x, obj_hist, _ = gpsr(A, y, tau=tau, max_iter=100)
        all_obj.extend(obj_hist)
        all_tau.extend([tau] * len(obj_hist))

    return x, np.array(all_obj), np.array(all_tau)


# ============================================================
# 6. Run Experiment
# ============================================================

n = 512
k = 40
m = 200

x_true, support = generate_sparse_signal(n, k)
A, y = generate_measurements(x_true, m)

# Run GPSR
tau = 0.1
x_est, obj_hist, time_hist = gpsr(A, y, tau=tau)

# Debias
x_debiased = debias(A, y, x_est)

# Continuation
tau_values = [0.5, 0.2, 0.1]
x_cont, obj_cont, tau_cont = gpsr_continuation(A, y, tau_values)


# ============================================================
# 7. Figures
# ============================================================

# -------- Sparse Reconstruction --------
plt.figure(figsize=(10, 4))
plt.plot(x_true, label="True Signal", linewidth=2)
plt.plot(x_est, '--', label="GPSR Estimate")
plt.plot(x_debiased, ':', label="Debiased Estimate")
plt.legend()
plt.title("Sparse Signal Reconstruction")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig_sparse_reconstruction.png"), dpi=200)
plt.close()


# -------- Objective vs Iteration --------
plt.figure()
plt.plot(obj_hist)
plt.xlabel("Iteration")
plt.ylabel("Objective Value")
plt.title("Objective vs Iteration")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig_objective_iterations.png"), dpi=200)
plt.close()


# -------- Objective vs Time --------
plt.figure()
plt.plot(time_hist, obj_hist)
plt.xlabel("Time (seconds)")
plt.ylabel("Objective Value")
plt.title("Objective vs Time")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig_objective_time.png"), dpi=200)
plt.close()


# -------- Continuation --------
plt.figure()
plt.scatter(tau_cont, obj_cont, s=5)
plt.xlabel("Tau Value")
plt.ylabel("Objective Value")
plt.title("Continuation Strategy Behavior")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig_continuation.png"), dpi=200)
plt.close()


# ============================================================
# 8. Metrics (print for table)
# ============================================================

mse_before = np.mean((x_true - x_est)**2)
mse_after = np.mean((x_true - x_debiased)**2)
support_recovery = len(set(np.where(x_true != 0)[0]) & set(np.where(x_est != 0)[0])) / k * 100

print("\n===== RESULTS =====")
print("MSE (Before Debiasing):", mse_before)
print("MSE (After Debiasing):", mse_after)
print("Support Recovery (%):", support_recovery)
print("Final Objective:", obj_hist[-1])
print("Iterations:", len(obj_hist))
print("Runtime:", time_hist[-1])
