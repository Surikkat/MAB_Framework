import numpy as np
import os

def make_reward_function(theta_dim=5, x_dim=5, m_true=2, m_extra=3, scale=0.5, seed=0):
    rng = np.random.RandomState(seed)

    W_t_true = np.zeros((theta_dim, m_true))
    W_x_true = np.zeros((x_dim, m_true))
    for i in range(m_true):
        W_t_true[i, i] = 1.0
        W_x_true[i, i] = 1.0

    W_t_extra = rng.randn(theta_dim, m_extra)
    W_x_extra = rng.randn(x_dim, m_extra)

    W_t = np.concatenate([W_t_true, W_t_extra], axis=1)
    W_x = np.concatenate([W_x_true, W_x_extra], axis=1)

    def reward_fn(theta, x):
        g = theta @ W_t
        p = x @ W_x
        dot_val = np.dot(g, p)
        return float(np.tanh(scale * dot_val) + 2 * np.sin(3 * dot_val))

    return reward_fn

def generate_dataset():
    SEED = 42
    T = 200
    pool_size = 50
    x_dim = 5
    theta_dim = 5
    save_path = "data/E1_dataset.npz"

    os.makedirs("data", exist_ok=True)
    rng = np.random.RandomState(SEED)

    X_pool = rng.randn(pool_size, x_dim).astype(np.float32)
    
    true_reward_fn = make_reward_function(theta_dim, x_dim, seed=SEED+1)

    thetas = np.zeros((T, theta_dim), dtype=np.float32)
    rewards_clean = np.zeros((T, pool_size))
    thetas[0] = rng.randn(theta_dim)

    for t in range(T):
        if t > 0:
            thetas[t] = thetas[t-1] * 0.95 + rng.randn(theta_dim) * 0.3
        
        for i in range(pool_size):
            clean_r = true_reward_fn(thetas[t], X_pool[i])
            rewards_clean[t, i] = clean_r + rng.randn() * 0.02

    np.savez(save_path,
             X_pool=X_pool, 
             rewards_clean=rewards_clean, 
             T=T, 
             thetas=thetas,
             theta_dim=theta_dim,
             pool_size=pool_size)

if __name__ == "__main__":
    generate_dataset()