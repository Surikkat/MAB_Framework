import numpy as np
import pandas as pd
from pathlib import Path

Path("data").mkdir(exist_ok=True)

print("Генерация e-commerce логов...")
import sys
sys.path.append('..')

print("Генерация финансовых логов...")

def generate_finance_logs(n_samples=50000):
    np.random.seed(42)
    
    user_income = np.random.lognormal(mean=10.5, sigma=0.8, size=n_samples)
    user_age = np.random.randint(20, 70, n_samples)
    user_credit_history = np.random.choice([0, 1, 2], n_samples, p=[0.3, 0.4, 0.3])

    credit_score = (
        np.log(user_income) * 0.4 + 
        (user_age / 70) * 0.2 + 
        user_credit_history * 0.4
    )
    
    scores = np.column_stack([np.ones(n_samples), np.exp(credit_score)])
    propensity_all = scores / scores.sum(axis=1, keepdims=True)
    
    actions = np.array([np.random.choice(2, p=p) for p in propensity_all])
    propensity = propensity_all[np.arange(n_samples), actions]
    
    true_default_prob = 1 / (1 + np.exp(credit_score * 2))
    default = np.random.random(n_samples) < true_default_prob
    reward = 1 - default
    
    df = pd.DataFrame({
        'user_income': user_income,
        'user_age': user_age,
        'user_credit_history': user_credit_history,
        'item_id': actions,
        'action': actions,
        'propensity': np.clip(propensity + np.random.normal(0, 0.01, n_samples), 0.01, 0.99),
        'pscore': np.clip(propensity + np.random.normal(0, 0.01, n_samples), 0.01, 0.99),
        'reward': reward,
        'loan_amount': user_income * np.random.uniform(3, 6, n_samples)
    })
    
    return df

print("Генерация рекламных логов...")

def generate_ads_logs(n_samples=75000):
    np.random.seed(42)
    
    n_banners = 10
    
    user_activity = np.random.beta(2, 5, n_samples)
    hour = np.random.randint(0, 24, n_samples)
    device = np.random.choice(['mobile', 'desktop', 'tablet'], n_samples, p=[0.7, 0.25, 0.05])
    
    banner_ctr = np.random.beta(1, 20, n_banners)
    
    epsilon = 0.15
    is_random = np.random.random(n_samples) < epsilon
    
    actions = np.zeros(n_samples, dtype=int)
    propensity = np.zeros(n_samples)
    
    for i in range(n_samples):
        if is_random[i]:
            actions[i] = np.random.randint(n_banners)
            propensity[i] = 1.0 / n_banners
        else:
            actions[i] = np.argmax(banner_ctr)
            propensity[i] = 0.95

    true_ctr = banner_ctr[actions]
    time_factor = np.where((hour >= 10) & (hour <= 18), 1.2, 0.8)
    true_ctr = np.clip(true_ctr * time_factor, 0.001, 0.15)
    
    reward = (np.random.random(n_samples) < true_ctr).astype(int)
    
    df = pd.DataFrame({
        'user_activity_score': user_activity,
        'hour_of_day': hour,
        'device_type': device,
        'item_id': actions,
        'action': actions,
        'propensity': np.clip(propensity + np.random.normal(0, 0.01, n_samples), 0.01, 0.99),
        'pscore': np.clip(propensity + np.random.normal(0, 0.01, n_samples), 0.01, 0.99),
        'reward': reward,
        'banner_position': np.random.randint(0, 5, n_samples)
    })
    
    return df


if __name__ == "__main__":
    data_dir = Path(__file__).parent
    df_fin = generate_finance_logs()
    fin_path = data_dir / "finance_demo.parquet"
    df_fin.to_parquet(fin_path, index=False)
    print(f"✅ Сохранено: {fin_path} ({len(df_fin):,} строк)")

    df_ads = generate_ads_logs()
    ads_path = data_dir / "ads_demo.parquet"
    df_ads.to_parquet(ads_path, index=False)
    print(f"✅ Сохранено: {ads_path} ({len(df_ads):,} строк)")

print("✅ Демо-данные готовы!")