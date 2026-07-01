import pandas as pd
import numpy as np

class LogValidator:
    REQUIRED_COLUMNS = [
        'item_id',
        'propensity',
        'reward'
    ]

    RECOMMENDED_COLUMNS = [
        'user_age',
        'user_gender', 
        'user_activity_score',
        'user_avg_check',
        'user_views_7d',
        'user_clicks_7d',
        'hour_of_day',
        'device_type',
        'item_price',
        'item_rating',
        'category_id'
    ]
    
    def validate(self, df):
        warnings = []
        is_valid = True
        
        missing_required = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        
        if missing_required:
            is_valid = False
            warnings.append(f"❌ Отсутствуют обязательные колонки: {missing_required}")
            return is_valid, warnings

        if len(df) == 0:
            is_valid = False
            warnings.append("❌ Датафрейм пуст")
            return is_valid, warnings

        if not pd.api.types.is_numeric_dtype(df['propensity']):
            warnings.append("❌ Колонка 'propensity' должна быть числовой")
            is_valid = False
        
        if not pd.api.types.is_numeric_dtype(df['reward']):
            warnings.append("❌ Колонка 'reward' должна быть числовой")
            is_valid = False

        if df['propensity'].min() <= 0:
            warnings.append("⚠️ Обнаружены нулевые или отрицательные propensity. Они будут заменены на min_propensity")
        
        if df['propensity'].max() > 1.0:
            warnings.append("⚠️ Обнаружены propensity > 1.0. Они будут обрезаны")
        
        very_low_propensity = (df['propensity'] < 0.001).mean()
        if very_low_propensity > 0.5:
            warnings.append(f"⚠️ {very_low_propensity*100:.1f}% записей имеют propensity < 0.001. OPE будет нестабильной")

        if df['reward'].nunique() == 1:
            warnings.append("⚠️ Все reward одинаковые. Модель не сможет обучиться")
        
        if df['reward'].mean() == 0:
            warnings.append("⚠️ Нулевой средний reward. Проверьте данные")
        
        if df['reward'].mean() > 0.5:
            warnings.append(f"⚠️ Очень высокий CTR: {df['reward'].mean()*100:.1f}%. Возможно, данные сгенерированы искусственно")

        if len(df) < 1000:
            warnings.append("⚠️ Слишком мало данных (<1000 записей). Оценки будут неточными")
        
        if len(df) < 10000:
            warnings.append(f"💡 Всего {len(df)} записей. Рекомендуется минимум 10K для стабильной оценки")

        n_unique_actions = df['item_id'].nunique()
        if n_unique_actions < 3:
            warnings.append(f"⚠️ Всего {n_unique_actions} уникальных действий. Слишком мало для сравнения политик")

        missing_recommended = [col for col in self.RECOMMENDED_COLUMNS if col not in df.columns]
        if missing_recommended:
            warnings.append(f"💡 Отсутствуют рекомендуемые колонки: {missing_recommended}. Direct Method будет работать хуже")

        n_duplicates = df.duplicated().sum()
        if n_duplicates > 0:
            warnings.append(f"💡 Найдено {n_duplicates} дубликатов. Рекомендуется их удалить")
 
        if 'is_exploration' in df.columns:
            exploration_rate = df['is_exploration'].mean()
            if exploration_rate < 0.05:
                warnings.append(f"⚠️ Exploration всего {exploration_rate*100:.1f}%. OPE для новых политик будет неточной")

        n_missing = df[self.REQUIRED_COLUMNS].isnull().sum().sum()
        if n_missing > 0:
            warnings.append(f"⚠️ Найдено {n_missing} пропусков в обязательных колонках")

        propensity_median = df['propensity'].median()
        if propensity_median < 0.05:
            warnings.append("⚠️ Медианный propensity < 0.05. Exploration может быть недостаточным")

        propensity_std = df['propensity'].std()
        if propensity_std < 0.01:
            warnings.append("💡 Очень маленькая дисперсия propensity. Возможно, политика почти детерминированная")

        n_critical = sum(1 for w in warnings if w.startswith('❌'))
        n_warnings = sum(1 for w in warnings if w.startswith('⚠️'))
        
        if n_critical == 0 and n_warnings == 0:
            warnings.append("✅ Данные выглядят хорошими для OPE!")
        
        return is_valid, warnings
    
    def get_log_quality_score(self, df):
        score = 100

        if len(df) < 10000:
            score -= 20
        elif len(df) < 50000:
            score -= 10
        
        if df['propensity'].min() <= 0.001:
            score -= 15
        
        if (df['propensity'] < 0.01).mean() > 0.3:
            score -= 15
        
        if df['item_id'].nunique() < 5:
            score -= 15
        
        if df['reward'].mean() == 0 or df['reward'].mean() > 0.5:
            score -= 10
        
        missing_recommended = [c for c in self.RECOMMENDED_COLUMNS if c not in df.columns]
        score -= len(missing_recommended) * 2
        
        return max(0, score)