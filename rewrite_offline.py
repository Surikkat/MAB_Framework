import re

with open('streamlit_bandit/app.py', 'r') as f:
    content = f.read()

offline_ui_target = """        else:
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
                            # Рендерим гиперпараметры (только для bandit-алгоритмов)
                            if 'wrapper' in row and hasattr(row.get('wrapper', None), 'algorithm_class'):
                                wrapper = row['wrapper']
                                cls_name = wrapper.algorithm_class.__name__
                                hp = render_hyperparams(
                                    algo_display_name=algo_name,
                                    algo_class_name=cls_name,
                                    unique_key=f"offline_{algo_name}",
                                    preset_algo_params=wrapper.algorithm_kwargs,
                                    preset_model_params=wrapper.model_kwargs
                                )
                                if hp['algo_params']:
                                    wrapper.algorithm_kwargs.update(hp['algo_params'])
                                if hp['model_params'] and wrapper.model_kwargs is not None:
                                    wrapper.model_kwargs.update(hp['model_params'])
            
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
                    algo_name = candidate['name']"""

offline_ui_replacement = """        else:
            st.subheader("Выберите конкретные алгоритмы:")
            
            for cat in categories:
                cat_candidates = available_candidates[available_candidates['category'] == cat]
                
                with st.expander(f"{cat} ({len(cat_candidates)} алгоритмов)", expanded=False):
                    for _, row in cat_candidates.iterrows():
                        algo_name = row['name']
                        st.markdown(f"**{row['complexity']} {algo_name}** — {row['description']}")
                        
                        hp = {'algo_params': {}, 'model_params': {}}
                        if 'wrapper' in row and hasattr(row.get('wrapper', None), 'algorithm_class'):
                            wrapper = row['wrapper']
                            cls_name = wrapper.algorithm_class.__name__
                            hp = render_hyperparams(
                                algo_display_name=algo_name,
                                algo_class_name=cls_name,
                                unique_key=f"offline_{algo_name}",
                                preset_algo_params=wrapper.algorithm_kwargs,
                                preset_model_params=wrapper.model_kwargs
                            )
                        if st.button("➕", key=f"add_offline_{algo_name}"):
                            st.session_state['offline_experiment_list'].append({
                                'name': algo_name,
                                'algo_params': hp['algo_params'],
                                'model_params': hp['model_params'],
                                'category': cat,
                                'complexity': row['complexity']
                            })
                            st.rerun()

        st.subheader("Выбранные алгоритмы в эксперименте")
        for i, item in enumerate(st.session_state['offline_experiment_list']):
            col1, col2 = st.columns([0.9, 0.1])
            with col1:
                desc = f"**{item['complexity']} {item['name']}**"
                if item['algo_params'] or item['model_params']:
                    desc += f" (Algo: {item['algo_params']}, Model: {item['model_params']})"
                st.markdown(desc)
            with col2:
                if st.button("❌", key=f"remove_offline_{i}"):
                    st.session_state['offline_experiment_list'].pop(i)
                    st.rerun()

        if len(st.session_state['offline_experiment_list']) == 0:
            st.warning("👆 Добавьте хотя бы один алгоритм для запуска бенчмарка")

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            run_benchmark = st.button(
                "🚀 ЗАПУСТИТЬ БЕНЧМАРК", 
                type="primary", 
                use_container_width=True,
                disabled=(len(st.session_state['offline_experiment_list']) == 0),
                key="offline_run_btn"
            )

        if run_benchmark and len(st.session_state['offline_experiment_list']) > 0:
            with st.spinner(f"Оцениваем {len(st.session_state['offline_experiment_list'])} алгоритмов..."):
                
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
                
                total = len(st.session_state['offline_experiment_list'])
                
                for i, candidate in enumerate(st.session_state['offline_experiment_list']):
                    algo_name = candidate['name']
                    # Формируем уникальное имя для результатов
                    unique_name = f"{algo_name} #{i+1}"
"""

new_content = content.replace(offline_ui_target, offline_ui_replacement)

# Update the running logic down below (line 271)
old_eval_logic = """                    try:
                        algo_progress.progress(0.33, text=f"{algo_name}: propensity...")
                        candidate_fn = pool.get_candidate(algo_name)
                        algo_progress.progress(0.66, text=f"{algo_name}: DM/IPS/DR...")
                        result = evaluator.evaluate(algo_name, candidate_fn)
                        algo_progress.progress(1.0, text=f"{algo_name}: готово")
                        result['category'] = candidate['category']
                        result['complexity'] = candidate['complexity']
                        results.append(result)"""

new_eval_logic = """                    try:
                        algo_progress.progress(0.33, text=f"{unique_name}: propensity...")
                        candidate_fn = pool.get_candidate(
                            algo_name, 
                            custom_algo_params=candidate.get('algo_params'),
                            custom_model_params=candidate.get('model_params')
                        )
                        algo_progress.progress(0.66, text=f"{unique_name}: DM/IPS/DR...")
                        result = evaluator.evaluate(unique_name, candidate_fn)
                        algo_progress.progress(1.0, text=f"{unique_name}: готово")
                        result['category'] = candidate['category']
                        result['complexity'] = candidate['complexity']
                        results.append(result)"""

new_content = new_content.replace(old_eval_logic, new_eval_logic)

if new_content != content:
    with open('streamlit_bandit/app.py', 'w') as f:
        f.write(new_content)
    print("Offline UI updated successfully.")
else:
    print("Replacement failed. Target string not found.")

