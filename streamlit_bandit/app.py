import sys
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from pathlib import Path

_root_dir = str(Path(__file__).resolve().parent.parent)
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

from core.evaluator import OPEEvaluator
from core.candidates import CandidatePool
from core.validator import LogValidator
from utils.visualisation import (
    plot_candidate_comparison, 
    plot_effective_sample_size,
    plot_method_agreement
)
from utils.export import generate_report

st.set_page_config(
    page_title="OPE Platform – A/B без трафика",
    page_icon="🎯",
    layout="wide"
)

if 'benchmark_results' not in st.session_state:
    st.session_state['benchmark_results'] = None
if 'online_results' not in st.session_state:
    st.session_state['online_results'] = None


with st.sidebar:
    st.title("🎯 OPE Platform")
    st.caption("A/B-тестирование без трафика")
    
    st.subheader("🔄 Режим работы")
    mode = st.radio(
        "Выберите режим:",
        ["📊 Offline (OPE)", "🚀 Online (Бенчмарки)"],
        help="Offline — оценка по историческим логам. Online — запуск на средах и бенчмарках."
    )
    
    st.divider()
    
    if mode == "📊 Offline (OPE)":
        st.subheader("📂 Загрузка логов")
        uploaded_file = st.file_uploader(
            "Загрузите CSV или Parquet:",
            type=['csv', 'parquet'],
            help="Логи должны содержать: user_features, item_id, propensity, reward"
        )
        
        use_demo = st.checkbox("Использовать демо-данные", value=False)
        
        if use_demo:
            demo_dataset = st.selectbox(
                "Выберите домен:",
                ["E-commerce (100K логов)", "Финансы (50K логов)", "Реклама (75K логов)"]
            )
        
        st.divider()
        
        st.subheader("⚙️ Настройки оценки")
        min_propensity = st.number_input("Мин. propensity:", 0.001, 0.1, 0.01)
        clipping_value = st.slider("Клиппинг весов (M):", 1, 50, 10)
    
    else:
        st.subheader("⚙️ Настройки онлайн-режима")
        st.caption("Настройки будут доступны после выбора среды")

if mode == "📊 Offline (OPE)":
    
    if uploaded_file is not None or use_demo:
        if use_demo:
            data_dir = Path(__file__).parent / 'data'
            if "E-commerce" in demo_dataset:
                df_log = pd.read_parquet(data_dir / 'test_ecommerce.parquet')
                domain = "ecommerce"
            elif "Финансы" in demo_dataset:
                df_log = pd.read_parquet(data_dir / 'finance_demo.parquet')
                domain = "finance"
            else:
                df_log = pd.read_parquet(data_dir / 'ads_demo.parquet')
                domain = "ads"
        else:
            if uploaded_file.name.endswith('.csv'):
                df_log = pd.read_csv(uploaded_file)
            else:
                df_log = pd.read_parquet(uploaded_file)
        
        validator = LogValidator()
        is_valid, warnings = validator.validate(df_log)
        
        # Метрики
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Записей", f"{len(df_log):,}")
        with col2:
            st.metric("🎯 CTR", f"{df_log['reward'].mean()*100:.2f}%")
        with col3:
            st.metric("📦 Товаров", df_log['item_id'].nunique())
        with col4:
            st.metric("🎲 Avg Propensity", f"{df_log['propensity'].mean():.4f}")
        
        if warnings:
            with st.expander(f"⚠️ Предупреждения валидации ({len(warnings)})"):
                for w in warnings:
                    st.warning(w)
        
        st.divider()
        
        st.title("🏆 Бенчмарк алгоритмов (Offline)")
        st.caption("Выберите алгоритмы для сравнения на ваших данных")
        
        pool = CandidatePool(df_log, domain if use_demo else 'custom')
        available_candidates = pool.list_candidates()
        categories = available_candidates['category'].unique()

        selection_mode = st.radio(
            "Режим выбора:",
            ["🎯 Категориями", "🔍 Поштучно"],
            horizontal=True,
            index=1,
            key="offline_mode",
            help="Категориями — быстро выбрать группы. Поштучно — отметить конкретные алгоритмы."
        )
        
        if selection_mode == "🎯 Категориями":
            st.subheader("Выберите группы алгоритмов:")
            
            selected_groups = {}
            cols = st.columns(len(categories))
            for i, cat in enumerate(categories):
                with cols[i]:
                    count = len(available_candidates[available_candidates['category'] == cat])
                    selected_groups[cat] = st.checkbox(
                        f"{cat} ({count})",
                        value=False,
                        key=f"offline_group_{i}"
                    )

            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("✅ Выбрать всё", key="offline_all_btn", use_container_width=True):
                    for i in range(len(categories)):
                        st.session_state[f"offline_group_{i}"] = True
                    st.rerun()
            with col2:
                if st.button("❌ Снять всё", key="offline_none_btn", use_container_width=True):
                    for i in range(len(categories)):
                        st.session_state[f"offline_group_{i}"] = False
                    st.rerun()
            with col3:
                if st.button("🚀 Только быстрые", key="offline_fast_btn", use_container_width=True):
                    for i, cat in enumerate(categories):
                        st.session_state[f"offline_group_{i}"] = ('Stochastic' in cat)
                    st.rerun()
            
            filtered_candidates = available_candidates[
                available_candidates['category'].isin([cat for cat, sel in selected_groups.items() if sel])
            ]
            
        else:
            st.subheader("Выберите конкретные алгоритмы:")
            
            selected_names = []
            
            for cat in categories:
                cat_candidates = available_candidates[available_candidates['category'] == cat]
                
                with st.expander(f"{cat} ({len(cat_candidates)} алгоритмов)", expanded=False):
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button("✅ Все", key=f"offline_all_{cat}"):
                            for _, row in cat_candidates.iterrows():
                                st.session_state[f"offline_algo_{row['name']}"] = True
                            st.rerun()
                    with c2:
                        if st.button("❌ Снять", key=f"offline_none_{cat}"):
                            for _, row in cat_candidates.iterrows():
                                st.session_state[f"offline_algo_{row['name']}"] = False
                            st.rerun()
                    
                    for _, row in cat_candidates.iterrows():
                        algo_name = row['name']
                        key = f"offline_algo_{algo_name}"
                        
                        if key not in st.session_state:
                            st.session_state[key] = False
                        
                        checked = st.checkbox(
                            f"{row['complexity']} {algo_name}",
                            value=st.session_state[key],
                            key=key,
                            help=row['description']
                        )
                        if checked:
                            selected_names.append(algo_name)
            
            filtered_candidates = available_candidates[available_candidates['name'].isin(selected_names)]

        if len(filtered_candidates) > 0:
            st.info(f"🎯 Выбрано алгоритмов: **{len(filtered_candidates)}** из {len(available_candidates)}")
            with st.expander("📋 Список выбранных"):
                for _, row in filtered_candidates.iterrows():
                    st.markdown(f"- {row['complexity']} **{row['name']}** — {row['description']}")
        else:
            st.warning("👆 Выберите хотя бы один алгоритм для запуска бенчмарка")

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            run_benchmark = st.button(
                "🚀 ЗАПУСТИТЬ БЕНЧМАРК", 
                type="primary", 
                use_container_width=True,
                disabled=(len(filtered_candidates) == 0),
                key="offline_run_btn"
            )

        if run_benchmark and len(filtered_candidates) > 0:
            with st.spinner(f"Оцениваем {len(filtered_candidates)} алгоритмов..."):
                
                evaluator = OPEEvaluator(
                    df_log,
                    clipping_value=clipping_value,
                    min_propensity=min_propensity,
                    methods=['dm', 'ips', 'dr']
                )
                
                results = []
                overall_progress = st.progress(0, text="Общий прогресс...")
                algo_progress = st.progress(0, text="Подготовка...")
                status_text = st.empty()
                
                total = len(filtered_candidates)
                candidates_list = list(filtered_candidates.iterrows())
                
                for i, (_, candidate) in enumerate(candidates_list):
                    algo_name = candidate['name']
                    status_text.markdown(f"**Оценка {i+1}/{total}:** {algo_name}")
                    overall_progress.progress(i / total, text=f"Общий прогресс: {i}/{total}")
                    algo_progress.progress(0.0, text=f"{algo_name}: загрузка...")
                    
                    try:
                        algo_progress.progress(0.33, text=f"{algo_name}: propensity...")
                        candidate_fn = pool.get_candidate(algo_name)
                        algo_progress.progress(0.66, text=f"{algo_name}: DM/IPS/DR...")
                        result = evaluator.evaluate(algo_name, candidate_fn)
                        algo_progress.progress(1.0, text=f"{algo_name}: готово")
                        result['category'] = candidate['category']
                        result['complexity'] = candidate['complexity']
                        results.append(result)
                    except Exception as e:
                        st.warning(f"⚠️ {algo_name}: {str(e)[:100]}")
                
                overall_progress.progress(1.0, text=f"Готово: {total}/{total} ✅")
                algo_progress.progress(1.0, text="✅")
                status_text.empty()
                
                st.session_state['benchmark_results'] = results
                st.success(f"✅ Оценено {len(results)} алгоритмов!")
                st.balloons()

        if st.session_state.get('benchmark_results'):
            results = st.session_state['benchmark_results']
            results_df = pd.DataFrame(results)
            baseline_ctr = df_log['reward'].mean()
            
            st.divider()
            st.header("📊 Результаты бенчмарка")
            
            better_than_baseline = (results_df['dr_score'] > baseline_ctr).sum()
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Оценено алгоритмов", len(results_df))
            with c2:
                st.metric("Лучше продакшна", f"{better_than_baseline}/{len(results_df)}")
            with c3:
                best_idx = results_df['dr_score'].idxmax()
                best_name = results_df.iloc[best_idx]['candidate']
                best_delta = (results_df.iloc[best_idx]['dr_score'] - baseline_ctr) / baseline_ctr * 100
                st.metric("🏆 Лидер", best_name, delta=f"{best_delta:+.1f}%")
            
            st.subheader("🏅 Лучшие в категориях")
            result_categories = results_df['category'].unique()
            cols = st.columns(len(result_categories))
            for col, cat in zip(cols, result_categories):
                cat_results = results_df[results_df['category'] == cat]
                if len(cat_results) > 0:
                    best = cat_results.loc[cat_results['dr_score'].idxmax()]
                    delta = (best['dr_score'] - baseline_ctr) / baseline_ctr * 100
                    with col:
                        st.metric(f"{cat}", f"{best['candidate']}", delta=f"{delta:+.1f}%")
            
            st.subheader("🏆 Топ-10 алгоритмов")
            top_n = min(10, len(results_df))
            top10 = results_df.nlargest(top_n, 'dr_score')
            
            fig = go.Figure()
            colors = ['#FFD700' if i == 0 else '#C0C0C0' if i == 1 else '#CD7F32' if i == 2 else '#3498db' for i in range(len(top10))]
            fig.add_trace(go.Bar(
                x=top10['candidate'], y=[r*100 for r in top10['dr_score']],
                marker_color=colors, text=[f"{r*100:.3f}%" for r in top10['dr_score']],
                textposition='outside', hovertemplate='%{x}<br>CTR: %{y:.3f}%<extra></extra>'
            ))
            fig.add_hline(y=baseline_ctr*100, line_dash="dash", line_color="gray",
                         annotation_text=f"Продакшн ({baseline_ctr*100:.2f}%)")
            fig.update_layout(title=f"Топ-{top_n} по CTR (Doubly Robust)", yaxis_title="CTR (%)", height=500, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("📋 Полная таблица")
            display_df = results_df.copy()
            display_df['CTR'] = display_df['dr_score'].apply(lambda x: f"{x*100:.3f}%")
            display_df['vs Baseline'] = display_df['dr_score'].apply(
                lambda x: f"+{(x-baseline_ctr)/baseline_ctr*100:.1f}%" if x > baseline_ctr else f"{(x-baseline_ctr)/baseline_ctr*100:.1f}%"
            )
            display_df['ESS'] = display_df['effective_sample_size'].apply(lambda x: f"{x:,.0f}" if x < 1000 else f"{x/1000:.1f}K")
            display_df['DM'] = display_df.get('dm_score', display_df['dr_score']).apply(lambda x: f"{x*100:.3f}%")
            display_df['IPS'] = display_df.get('ips_score', display_df['dr_score']).apply(lambda x: f"{x*100:.3f}%")
            display_df['DR'] = display_df['dr_score'].apply(lambda x: f"{x*100:.3f}%")
            display_df = display_df.sort_values('dr_score', ascending=False)
            
            st.dataframe(
                display_df[['category', 'candidate', 'complexity', 'CTR', 'vs Baseline', 'ESS', 'DM', 'IPS', 'DR']],
                use_container_width=True, hide_index=True
            )
            
            st.subheader("📥 Экспорт")
            c1, c2 = st.columns(2)
            with c1:
                csv = display_df.to_csv(index=False)
                st.download_button("Скачать CSV", csv, f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", "text/csv", use_container_width=True)
            with c2:
                if st.button("📄 Скачать PDF-отчёт", key="offline_pdf_btn", use_container_width=True):
                    pdf_bytes = generate_report(results, baseline_ctr, df_log)
                    st.download_button("💾 Сохранить PDF", pdf_bytes, f"ope_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf", "application/pdf")
    
    else:
        st.title("📊 Offline (OPE) — Загрузка логов")
        st.caption("Загрузите логи для оценки алгоритмов без A/B-теста")
        
        st.markdown("""
        ### 📂 Что нужно сделать:
        1. **Загрузите логи** в боковой панели (CSV или Parquet)  
           — или —
        2. **Включите демо-данные** для тестирования
        
        ---
        
        ### 📋 Требования к логам:
        - `item_id` — ID показанного действия
        - `propensity` — вероятность выбора продакшн-политикой
        - `reward` — награда (0/1)
        - `user_*` — признаки пользователя (опционально)
        - `item_*` — признаки товара (опционально)
        """)

else:
    from core.online_experiment import (
        get_available_environments, 
        get_available_algorithms_for_online,
        run_online_experiment,
        format_results_table
    )
    
    st.title("🚀 Онлайн-бенчмарки")
    st.caption("Запустите алгоритмы на стандартных средах и сравните их")

    st.subheader("1️⃣ Выберите среду")
    envs_df = get_available_environments()
    
    synthetic_envs = envs_df[envs_df['type'] == 'synthetic']
    real_envs = envs_df[envs_df['type'] == 'real']
    
    env_type = st.radio(
        "Тип среды:",
        ["🎲 Синтетические", "📦 SOTA бенчмарки", "📂 Загрузить свои данные"],
        horizontal=True,
        key="online_env_type"
    )
    
    if env_type == "🎲 Синтетические":
        selected_env = st.selectbox(
            "Синтетическая среда:",
            synthetic_envs['name'].tolist(),
            key="online_env_synthetic"
        )
        env_row = synthetic_envs[synthetic_envs['name'] == selected_env].iloc[0]
        
        with st.expander("⚙️ Настройки параметров среды", expanded=True):
            default_params = env_row.get('default_params', {})
            
            c1, c2, c3 = st.columns(3)
            with c1:
                n_arms = st.number_input(
                    "Количество рук (n_arms):",
                    min_value=2, max_value=100,
                    value=default_params.get('n_arms', 10),
                    key="env_n_arms"
                )
            with c2:
                context_dim = st.number_input(
                    "Размерность контекста (context_dim):",
                    min_value=1, max_value=100,
                    value=default_params.get('context_dim', 5),
                    key="env_context_dim"
                )
            with c3:
                noise_std = st.slider(
                    "Уровень шума (noise_std):",
                    min_value=0.01, max_value=1.0,
                    value=default_params.get('noise_std', 0.1),
                    step=0.01,
                    key="env_noise_std"
                )
            
            env_params = {
                'n_arms': n_arms,
                'context_dim': context_dim,
                'noise_std': noise_std,
            }
        
    elif env_type == "📦 SOTA бенчмарки":
        selected_env = st.selectbox(
            "Бенчмарк:",
            real_envs['name'].tolist(),
            key="online_env_real"
        )
        env_row = real_envs[real_envs['name'] == selected_env].iloc[0]
        env_params = env_row.get('default_params', {})
        
    else:
        st.info("📂 Загрузка своих данных появится в следующей версии")
        st.stop()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Тип", env_row['type'])
    with c2:
        actual_arms = env_params.get('n_arms', env_row.get('n_arms', '?'))
        st.metric("Рук (arms)", actual_arms)
    with c3:
        if 'context_dim' in env_params:
            st.metric("Размерность", env_params['context_dim'])
        else:
            st.metric("Описание", str(env_row['description'])[:50] + "...")
    
    st.subheader("2️⃣ Выберите алгоритмы")
    algos_df = get_available_algorithms_for_online()
    
    selected_algos = []
    online_categories = algos_df['category'].unique()
    
    # Считаем выбранные
    for cat in online_categories:
        cat_algos = algos_df[algos_df['category'] == cat]
        with st.expander(f"{cat} ({len(cat_algos)} алгоритмов)", expanded=False):
            for _, row in cat_algos.iterrows():
                if st.checkbox(f"{row['name']}", key=f"online_algo_{row['name']}"):
                    selected_algos.append(row)
    
    selected_algos_df = pd.DataFrame(selected_algos) if selected_algos else pd.DataFrame()
    
    st.subheader("3️⃣ Параметры эксперимента")
    c1, c2 = st.columns(2)
    with c1:
        steps = st.slider("Количество шагов:", 50, 1000, 200, 50, key="online_steps_slider")
    with c2:
        n_runs = st.slider("Количество запусков:", 1, 10, 3, key="online_runs_slider")
    
    # Запуск
    st.subheader("4️⃣ Запуск")
    st.info(f"Выбрано алгоритмов: **{len(selected_algos)}** | Шагов: {steps} | Запусков: {n_runs}")
    
    if st.button("🚀 ЗАПУСТИТЬ ОНЛАЙН-ЭКСПЕРИМЕНТ", type="primary", use_container_width=True,
                 disabled=(len(selected_algos) == 0), key="online_run_btn"):
        
        with st.spinner(f"Запускаем эксперимент на {steps} шагов x {n_runs} запусков..."):
            progress_text = st.empty()
            overall_progress = st.progress(0, text="Запуск...")
            
            def progress_callback(msg):
                progress_text.text(msg)
            
            all_results = run_online_experiment(
                env_row, selected_algos_df,
                env_params=env_params,  # ← добавить
                steps=steps, n_runs=n_runs,
                progress_callback=progress_callback
            )
            
            overall_progress.progress(1.0, text="Готово! ✅")
            progress_text.empty()
            
            st.session_state['online_results'] = all_results
            st.session_state['online_env'] = selected_env
            st.session_state['online_steps_saved'] = steps
            st.session_state['online_runs_saved'] = n_runs
            st.success("✅ Эксперимент завершён!")
            st.balloons()
    
    if st.session_state.get('online_results'):
        all_results = st.session_state['online_results']
        s = st.session_state.get('online_steps_saved', steps)
        
        st.divider()
        st.header("📊 Результаты онлайн-эксперимента")
        st.caption(f"Среда: {st.session_state.get('online_env', '')} | Шагов: {s} | Запусков: {st.session_state.get('online_runs_saved', '')}")
        
        results_df = format_results_table(all_results, s)
        st.subheader("🏆 Leaderboard")
        st.dataframe(results_df, use_container_width=True, hide_index=True)
        
        st.subheader("📈 Cumulative Regret")
        fig = go.Figure()
        t_range = np.arange(1, s + 1)
        for name, data in all_results.items():
            if 'cumulative_regret_mean' in data:
                fig.add_trace(go.Scatter(x=t_range[:len(data['cumulative_regret_mean'])], y=data['cumulative_regret_mean'],
                                        name=name, mode='lines', line=dict(width=2)))
        fig.update_layout(title="Cumulative Regret", xaxis_title="Шаг", yaxis_title="Regret", height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("📉 Average Regret")
        fig2 = go.Figure()
        for name, data in all_results.items():
            if 'average_regret_mean' in data:
                fig2.add_trace(go.Scatter(x=t_range[:len(data['average_regret_mean'])], y=data['average_regret_mean'],
                                         name=name, mode='lines', line=dict(width=2)))
        fig2.update_layout(title="Average Regret", xaxis_title="Шаг", yaxis_title="Regret", height=500)
        st.plotly_chart(fig2, use_container_width=True)

st.divider()
st.caption(f"🎯 OPE Platform v0.4.0 | {datetime.now().year}")