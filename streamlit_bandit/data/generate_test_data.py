import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(123)

N_SAMPLES = 100
N_ITEMS = 10

items = pd.DataFrame({
    'item_id': np.arange(N_ITEMS),
    'item_popularity': np.random.exponential(1, N_ITEMS),
    'item_price': [15000, 3500, 800, 45000, 2500, 1500, 9000, 3200, 600, 19000],
    'category_id': np.random.choice(3, N_ITEMS),
    'item_age_days': np.random.randint(1, 365, N_ITEMS),
    'item_rating': np.clip(np.random.normal(4.0, 0.8, N_ITEMS), 1, 5).round(1)
})
items['item_popularity_norm'] = items['item_popularity'] / items['item_popularity'].sum()

user_ages = np.random.randint(18, 60, N_SAMPLES)
user_genders = np.random.choice(['M', 'F'], N_SAMPLES, p=[0.3, 0.7])
user_activity = np.random.beta(2, 5, N_SAMPLES).round(3)
user_avg_checks = np.random.lognormal(mean=7.6, sigma=0.6, size=N_SAMPLES).round(-1)
user_views = np.random.poisson(lam=3, size=N_SAMPLES)
user_clicks = np.random.binomial(np.maximum(user_views, 1), 0.25)

hour_probs = np.array([
    0.02, 0.02, 0.01, 0.01, 0.01,
    0.02, 0.03, 0.05, 0.07, 0.07,
    0.07, 0.07, 0.06, 0.05, 0.04,
    0.04, 0.04, 0.05, 0.07, 0.07,
    0.05, 0.03, 0.02, 0.02
])

hour_probs = hour_probs / hour_probs.sum()
hours = np.random.choice(24, N_SAMPLES, p=hour_probs)

devices = np.random.choice(['mobile', 'desktop', 'tablet'], N_SAMPLES, p=[0.8, 0.15, 0.05])

item_qualities = (items['item_rating'].values - 1) / 4

scores = np.zeros((N_SAMPLES, N_ITEMS))
for i in range(N_ITEMS):
    scores[:, i] = item_qualities[i] + np.random.normal(0, 0.2, N_SAMPLES)

scores_exp = np.exp(scores / 0.5)
propensity_all = scores_exp / scores_exp.sum(axis=1, keepdims=True)

epsilon = 0.1
is_exploration = np.random.random(N_SAMPLES) < epsilon
propensity_all[is_exploration] = 1.0 / N_ITEMS

actions = np.array([np.random.choice(N_ITEMS, p=p) for p in propensity_all])
propensity_chosen = propensity_all[np.arange(N_SAMPLES), actions]
propensity_chosen = np.clip(propensity_chosen + np.random.normal(0, 0.01, N_SAMPLES), 0.01, 0.99).round(4)

true_ctr = 0.01 + 0.04 * item_qualities[actions]
reward = (np.random.random(N_SAMPLES) < true_ctr).astype(int)

df_log = pd.DataFrame({
    'user_age': user_ages,
    'user_gender': user_genders,
    'user_activity_score': user_activity,
    'user_avg_check': user_avg_checks,
    'user_views_7d': user_views,
    'user_clicks_7d': user_clicks,
    'hour_of_day': hours,
    'device_type': devices,
    'item_id': actions,
    'action': actions,
    'propensity': propensity_chosen,
    'pscore': propensity_chosen,
    'reward': reward,
    'category_id': items['category_id'].values[actions],
    'item_price': items['item_price'].values[actions],
    'item_rating': items['item_rating'].values[actions],
    'n_available_items': N_ITEMS,
})

save_path = Path(__file__).parent / "test_ecommerce.parquet"
df_log.to_parquet(save_path, index=False)

print(f"✅ Сохранено: {save_path}")
print(f"   Записей: {len(df_log):,}")
print(f"   Товаров: {N_ITEMS}")
print(f"   CTR: {df_log['reward'].mean()*100:.2f}%")
print(f"   Avg propensity: {df_log['propensity'].mean():.4f}")
print(f"   Mobile: {(df_log['device_type'] == 'mobile').mean()*100:.0f}%")
print(f"   Женщин: {(df_log['user_gender'] == 'F').mean()*100:.0f}%")
print(f"   Сумма вероятностей часов: {hour_probs.sum():.4f}")
print(f"\nПервые 5 строк:")
print(df_log.head().to_string())