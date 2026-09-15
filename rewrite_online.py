import re

with open('streamlit_bandit/app.py', 'r') as f:
    content = f.read()

online_ui_target = """    selected_algos = []
    online_categories = algos_df['category'].unique()
    
    # Считаем выбранные
    for cat in online_categories:
        cat_algos = algos_df[algos_df['category'] == cat]
        with st.expander(f"{cat} ({len(cat_algos)} алгоритмов)", expanded=False):
            for _, row in cat_algos.iterrows():
                if st.checkbox(f"{row['name']}", key=f"online_algo_{row['name']}"):
                    hp = render_hyperparams(
                        algo_display_name=row['name'],
                        algo_class_name=row['algo_name'],
                        unique_key=f"online_{row['name']}",
                        preset_algo_params=row.get('params', {}),
                        preset_model_params=row.get('model_params', {})
                    )
                    row_dict = row.to_dict()
                    # Мержим пользовательские параметры поверх дефолтных
                    merged_algo_params = dict(row_dict.get('params') or {})
                    merged_algo_params.update(hp['algo_params'])
                    row_dict['params'] = merged_algo_params
                    if row_dict.get('model_params') is not None:
                        merged_model_params = dict(row_dict.get('model_params') or {})
                        merged_model_params.update(hp['model_params'])
                        row_dict['model_params'] = merged_model_params
                    selected_algos.append(row_dict)
    
    selected_algos_df = pd.DataFrame(selected_algos) if selected_algos else pd.DataFrame()"""

online_ui_replacement = """    online_categories = algos_df['category'].unique()
    
    for cat in online_categories:
        cat_algos = algos_df[algos_df['category'] == cat]
        with st.expander(f"{cat} ({len(cat_algos)} алгоритмов)", expanded=False):
            for _, row in cat_algos.iterrows():
                st.markdown(f"**{row['name']}**")
                hp = render_hyperparams(
                    algo_display_name=row['name'],
                    algo_class_name=row['algo_name'],
                    unique_key=f"online_{row['name']}",
                    preset_algo_params=row.get('params', {}),
                    preset_model_params=row.get('model_params', {})
                )
                if st.button("➕", key=f"add_online_{row['name']}"):
                    row_dict = row.to_dict()
                    merged_algo_params = dict(row_dict.get('params') or {})
                    merged_algo_params.update(hp['algo_params'])
                    row_dict['params'] = merged_algo_params
                    if row_dict.get('model_params') is not None:
                        merged_model_params = dict(row_dict.get('model_params') or {})
                        merged_model_params.update(hp['model_params'])
                        row_dict['model_params'] = merged_model_params
                    # Добавим уникальное имя для отображения на графиках, чтобы они не склеивались
                    idx = len(st.session_state['online_experiment_list']) + 1
                    row_dict['display_name'] = f"{row['name']} #{idx}"
                    st.session_state['online_experiment_list'].append(row_dict)
                    st.rerun()

    st.subheader("Выбранные алгоритмы в эксперименте")
    for i, item in enumerate(st.session_state['online_experiment_list']):
        col1, col2 = st.columns([0.9, 0.1])
        with col1:
            desc = f"**{item['display_name']}**"
            if item.get('params') or item.get('model_params'):
                desc += f" (Algo: {item.get('params', {})}, Model: {item.get('model_params', {})})"
            st.markdown(desc)
        with col2:
            if st.button("❌", key=f"remove_online_{i}"):
                st.session_state['online_experiment_list'].pop(i)
                st.rerun()
                
    selected_algos_df = pd.DataFrame(st.session_state['online_experiment_list']) if st.session_state['online_experiment_list'] else pd.DataFrame()"""

new_content = content.replace(online_ui_target, online_ui_replacement)

# Also update the launch info line:
old_launch_info = """st.info(f"Выбрано алгоритмов: **{len(selected_algos)}** | Шагов: {steps} | Запусков: {n_runs}")"""
new_launch_info = """st.info(f"Выбрано алгоритмов: **{len(st.session_state['online_experiment_list'])}** | Шагов: {steps} | Запусков: {n_runs}")"""
new_content = new_content.replace(old_launch_info, new_launch_info)

old_run_btn = """if st.button("🚀 ЗАПУСТИТЬ ОНЛАЙН-ЭКСПЕРИМЕНТ", type="primary", use_container_width=True,
                 disabled=(len(selected_algos) == 0), key="online_run_btn"):"""
new_run_btn = """if st.button("🚀 ЗАПУСТИТЬ ОНЛАЙН-ЭКСПЕРИМЕНТ", type="primary", use_container_width=True,
                 disabled=(len(st.session_state['online_experiment_list']) == 0), key="online_run_btn"):"""
new_content = new_content.replace(old_run_btn, new_run_btn)

if new_content != content:
    with open('streamlit_bandit/app.py', 'w') as f:
        f.write(new_content)
    print("Online UI updated successfully.")
else:
    print("Replacement failed. Target string not found.")

