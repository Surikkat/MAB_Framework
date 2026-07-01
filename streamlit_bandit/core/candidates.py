import numpy as np
import pandas as pd
from core.bandit_candidates import BanditCandidatePool

class CandidatePool:
    def __init__(self, df_log, domain='ecommerce'):
        self.df = df_log
        self.domain = domain
        self.n_items = df_log['item_id'].nunique()
        self.bandit_pool = BanditCandidatePool(df_log)
        
        self._bandit_cache = {}
        
    def list_candidates(self):
        rule_based = pd.DataFrame([
            {
                'name': 'Popular',
                'description': 'Топ-10 популярных товаров',
                'category': '📏 Rule-based',
                'complexity': '⭐',
                'type': 'rule'
            },
            {
                'name': 'Category Personalization',
                'description': 'Товары из любимой категории',
                'category': '📏 Rule-based',
                'complexity': '⭐⭐',
                'type': 'rule'
            },
            {
                'name': 'Price Range Match',
                'description': 'Цена ±30% от среднего чека',
                'category': '📏 Rule-based',
                'complexity': '⭐⭐',
                'type': 'rule'
            },
            {
                'name': 'High Rated',
                'description': 'Товары с рейтингом > 4.5',
                'category': '📏 Rule-based',
                'complexity': '⭐',
                'type': 'rule'
            },
            {
                'name': 'Device-Optimized',
                'description': 'Дёшево на мобильных, дорого на десктопе',
                'category': '📏 Rule-based',
                'complexity': '⭐⭐',
                'type': 'rule'
            },
            {
                'name': 'Random Baseline',
                'description': 'Случайные рекомендации (нижняя граница)',
                'category': '📏 Rule-based',
                'complexity': '⭐',
                'type': 'rule'
            }
        ])
        bandit_based = self.bandit_pool.list_candidates()
        bandit_based['type'] = 'bandit'
        bandit_based['category'] = bandit_based['category'].replace({
            'Classic Bandit': '🎰 Classic Bandit',
            'Bayesian Bandit': '🔮 Bayesian Bandit',
            'Neural Bandit': '🧠 Neural Bandit'
        })

        all_candidates = pd.concat([rule_based, bandit_based], ignore_index=True)
        
        return all_candidates
    
    def get_candidate(self, name):
        rule_map = {
            'Popular': self._candidate_popular,
            'Category Personalization': self._candidate_category,
            'Price Range Match': self._candidate_price,
            'High Rated': self._candidate_high_rated,
            'Device-Optimized': self._candidate_device,
            'Random Baseline': self._candidate_random,
        }
        
        if name in rule_map:
            return rule_map[name]
        
        bandit_candidates = self.bandit_pool.list_candidates()
        bandit_row = bandit_candidates[bandit_candidates['name'] == name]
        
        if len(bandit_row) > 0:
            if name not in self._bandit_cache:
                wrapper = bandit_row.iloc[0]['wrapper']
                self._bandit_cache[name] = wrapper.create_offline_propensity_fn(self.df)
            
            return self._bandit_cache[name]
        
        raise ValueError(f"Unknown candidate: {name}")


    def _candidate_popular(self, df):
        popular_items = df['item_id'].value_counts().head(10).index.tolist()
        return np.where(df['item_id'].isin(popular_items), 0.1, 0.01)
    
    def _candidate_category(self, df):
        return np.where(df['category_id'] == 0, 0.1, 0.01)
    
    def _candidate_price(self, df):
        ratio = df['item_price'] / df['user_avg_check']
        return np.where((ratio >= 0.7) & (ratio <= 1.3), 0.1, 0.01)
    
    def _candidate_high_rated(self, df):
        return np.where(df['item_rating'] >= 4.5, 0.1, 0.01)
    
    def _candidate_device(self, df):
        is_mobile = df['device_type'] == 'mobile'
        cheap = df['item_price'] < 1000
        expensive = df['item_price'] > 5000
        return np.where(is_mobile,
                       np.where(cheap, 0.1, 0.01),
                       np.where(expensive, 0.1, 0.01))
    
    def _candidate_random(self, df):
        return np.full(len(df), 1.0 / self.n_items)