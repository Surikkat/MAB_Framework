import pandas as pd
from datetime import datetime
from io import BytesIO
import os

def generate_report(results, baseline_ctr, df_log):
    output = BytesIO()

    report = []
    report.append("=" * 60)
    report.append("OPE PLATFORM — ОТЧЁТ ОБ ОЦЕНКЕ АЛГОРИТМОВ")
    report.append("=" * 60)
    report.append(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report.append(f"Логов проанализировано: {len(df_log):,}")
    report.append(f"Baseline CTR (продакшн): {baseline_ctr*100:.3f}%")
    report.append("")

    report.append("-" * 60)
    report.append("РЕЗУЛЬТАТЫ ОЦЕНКИ")
    report.append("-" * 60)
    
    for r in results:
        report.append(f"\nКандидат: {r['candidate']}")
        report.append(f"  Direct Method:     {r.get('dm_score', 0)*100:.4f}%")
        report.append(f"  IPS (clipped):     {r.get('ips_score', 0)*100:.4f}%")
        report.append(f"  Doubly Robust:     {r.get('dr_score', 0)*100:.4f}%")
        report.append(f"  Effective SS:      {r.get('effective_sample_size', 0):,.0f}")

        dr_score = r.get('dr_score', 0)
        delta = (dr_score - baseline_ctr) / baseline_ctr * 100
        report.append(f"  vs Baseline:       {delta:+.2f}%")

        ess = r.get('effective_sample_size', 0)
        if ess < 100:
            report.append(f"  Надёжность:        🔴 Низкая (ESS < 100)")
        elif ess < 1000:
            report.append(f"  Надёжность:        🟡 Средняя (ESS < 1000)")
        else:
            report.append(f"  Надёжность:        🟢 Высокая (ESS > 1000)")

    report.append("\n" + "-" * 60)
    report.append("РЕКОМЕНДАЦИЯ")
    report.append("-" * 60)

    best = max(results, key=lambda x: x.get('dr_score', 0))
    best_delta = (best['dr_score'] - baseline_ctr) / baseline_ctr * 100
    
    if best_delta > 1 and best['effective_sample_size'] > 1000:
        report.append(f"✅ Рекомендуется к запуску: {best['candidate']}")
        report.append(f"   Ожидаемый прирост CTR: +{best_delta:.1f}%")
    elif best_delta > 0:
        report.append(f"⚠️ Потенциальный кандидат: {best['candidate']}")
        report.append(f"   Прирост небольшой: +{best_delta:.1f}%")
        report.append(f"   Рекомендуется собрать больше данных")
    else:
        report.append(f"❌ Все кандидаты хуже продакшна")
        report.append(f"   Лучший результат: {best['candidate']} ({best_delta:+.1f}%)")
    
    report.append("\n" + "=" * 60)
    report.append("КОНЕЦ ОТЧЁТА")
    report.append("=" * 60)

    report_text = "\n".join(report)
    output.write(report_text.encode('utf-8'))
    output.seek(0)
    
    return output

def generate_csv_report(results):
    df = pd.DataFrame(results)
    csv_buffer = BytesIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    return csv_buffer

def get_effective_algorithm_params(algo_dict):
    """
    Извлекает конкретные значения гиперпараметров алгоритма с учётом дефолтов из ALGO_HYPERPARAMS.
    Никаких заглушек 'По умолчанию' — возвращает фактические значения (например, 'alpha=0.5, l2_reg=1.0').
    """
    from core.hyperparams import ALGO_HYPERPARAMS
    
    algo_class = algo_dict.get('algo_name')
    hp_spec = ALGO_HYPERPARAMS.get(algo_class, {})
    
    effective = {}
    for p in hp_spec.get('algo_params', []):
        if 'default' in p:
            effective[p['key']] = p['default']
    for p in hp_spec.get('model_params', []):
        if 'default' in p:
            effective[p['key']] = p['default']
            
    if algo_dict.get('params'):
        effective.update(algo_dict['params'])
    if algo_dict.get('model_params'):
        effective.update(algo_dict['model_params'])
        
    if effective:
        return ", ".join(f"{k}={v}" for k, v in effective.items())
    return "нет параметров"


def generate_online_csv_report(results_df, algo_details_map, env_name, delay_desc, steps, n_runs):
    """
    Генерирует CSV с результатами онлайн-эксперимента, дополненный конкретными параметрами и свойствами среды.
    """
    df = results_df.copy()
    df['Среда'] = env_name
    df['Задержка'] = delay_desc
    df['Шагов'] = steps
    df['Запусков'] = n_runs
    df['Категория'] = df['Алгоритм'].apply(lambda name: algo_details_map.get(name, {}).get('category', '—'))
    df['Параметры'] = df['Алгоритм'].apply(lambda name: algo_details_map.get(name, {}).get('params_str', '—'))
    
    ordered_cols = ['Алгоритм', 'Категория', 'Параметры', 'Cum. Regret', 'Avg Regret', 'Время (с)', 'Статус', 'Среда', 'Задержка', 'Шагов', 'Запусков']
    existing_cols = [c for c in ordered_cols if c in df.columns] + [c for c in df.columns if c not in ordered_cols]
    return df[existing_cols].to_csv(index=False)


def generate_online_text_report(env_name, delay_desc, steps, n_runs, algo_details, results_df):
    """
    Генерирует текстовый отчёт об онлайн-эксперименте (по аналогии с offline generate_report).
    """
    output = BytesIO()
    report = []
    report.append("=" * 60)
    report.append("MAB FRAMEWORK — ОТЧЁТ ОБ ОНЛАЙН-ЭКСПЕРИМЕНТЕ")
    report.append("=" * 60)
    report.append(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report.append(f"Среда: {env_name}")
    report.append(f"Задержка наград: {delay_desc}")
    report.append(f"Количество шагов: {steps}")
    report.append(f"Количество запусков (seeds): {n_runs}")
    report.append("")
    
    report.append("-" * 60)
    report.append("ПАРАМЕТРЫ АЛГОРИТМОВ")
    report.append("-" * 60)
    for item in algo_details:
        report.append(f"• {item['Алгоритм']} [{item['Категория']}]: {item['Параметры']}")
    report.append("")
    
    report.append("-" * 60)
    report.append("ЛИДЕРБОРД И РЕЗУЛЬТАТЫ")
    report.append("-" * 60)
    for idx, r in results_df.iterrows():
        status = r.get('Статус', '—')
        cum = r.get('Cum. Regret', 'N/A')
        avg = r.get('Avg Regret', 'N/A')
        t = r.get('Время (с)', 'N/A')
        report.append(f"#{idx+1} {r['Алгоритм']}: Cum.Regret = {cum}, Avg.Regret = {avg}, Время = {t}s [{status}]")
    report.append("")
    
    successful = results_df[results_df['Статус'].str.contains('Успех', na=False)]
    if not successful.empty:
        best = successful.iloc[0]
        report.append("-" * 60)
        report.append("ЛУЧШИЙ АЛГОРИТМ")
        report.append("-" * 60)
        report.append(f"🏆 Победитель: {best['Алгоритм']} (наименьший кумулятивный регрет: {best['Cum. Regret']})")
    
    report.append("\n" + "=" * 60)
    report.append("КОНЕЦ ОТЧЁТА")
    report.append("=" * 60)
    
    report_text = "\n".join(report)
    output.write(report_text.encode('utf-8'))
    output.seek(0)
    return output


def generate_online_html_report(env_name, delay_desc, steps, n_runs, algo_details, results_df, figures):
    """
    Генерирует автономную HTML-страницу отчёта со всеми графиками, параметрами и лидербордом.
    Включает интерактивные графики Plotly (CDN). Открывается в любом браузере.
    """
    output = BytesIO()
    
    charts_html = []
    for fig in figures:
        charts_html.append(fig.to_html(full_html=False, include_plotlyjs='cdn'))
        
    table_rows = []
    for item in algo_details:
        table_rows.append(f"<tr><td><b>{item['Алгоритм']}</b></td><td>{item['Категория']}</td><td><code>{item['Параметры']}</code></td></tr>")
    algo_table = "".join(table_rows)
    
    results_headers = "".join(f"<th>{c}</th>" for c in results_df.columns)
    results_rows = []
    for _, row in results_df.iterrows():
        cells = "".join(f"<td>{row[c]}</td>" for c in results_df.columns)
        results_rows.append(f"<tr>{cells}</tr>")
    results_table = "".join(results_rows)
    
    html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>Отчёт об онлайн-эксперименте — {env_name}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: #0e1117;
            color: #fafafa;
            margin: 0;
            padding: 30px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1, h2, h3 {{ color: #ffffff; }}
        .meta-box {{
            background: #1a1c23;
            border-radius: 8px;
            padding: 16px 20px;
            margin-bottom: 24px;
            border: 1px solid #30363d;
        }}
        .meta-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 12px;
            margin-top: 10px;
        }}
        .meta-item {{ font-size: 14px; color: #8b949e; }}
        .meta-item strong {{ color: #e6edf3; display: block; font-size: 16px; margin-top: 4px; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 16px 0 30px 0;
            background: #1a1c23;
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid #30363d;
        }}
        th, td {{
            padding: 12px 16px;
            text-align: left;
            border-bottom: 1px solid #30363d;
        }}
        th {{ background: #21262d; color: #e6edf3; font-weight: 600; }}
        code {{
            background: #282e38;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: monospace;
            color: #58a6ff;
        }}
        .chart-box {{
            background: #1a1c23;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 30px;
            page-break-inside: avoid;
            break-inside: avoid;
        }}
        @media print {{
            body {{ background: #ffffff; color: #000000; }}
            .meta-box, table, .chart-box {{ background: #ffffff; border-color: #ddd; color: #000; }}
            th {{ background: #f0f0f0; color: #000; }}
            code {{ background: #eee; color: #000; }}
        }}
    </style>
</head>
<body>
<div class="container">
    <h1>📊 Отчёт об онлайн-эксперименте</h1>
    
    <div class="meta-box">
        <h3>Параметры эксперимента</h3>
        <div class="meta-grid">
            <div class="meta-item">Среда:<strong>{env_name}</strong></div>
            <div class="meta-item">Задержка:<strong>{delay_desc}</strong></div>
            <div class="meta-item">Шагов:<strong>{steps}</strong></div>
            <div class="meta-item">Запусков (Seeds):<strong>{n_runs}</strong></div>
            <div class="meta-item">Дата:<strong>{datetime.now().strftime('%Y-%m-%d %H:%M')}</strong></div>
        </div>
    </div>

    <h2>📝 Параметры алгоритмов</h2>
    <table>
        <thead>
            <tr><th>Алгоритм</th><th>Категория</th><th>Точные параметры</th></tr>
        </thead>
        <tbody>
            {algo_table}
        </tbody>
    </table>

    <h2>🏆 Leaderboard</h2>
    <table>
        <thead>
            <tr>{results_headers}</tr>
        </thead>
        <tbody>
            {results_table}
        </tbody>
    </table>

    <h2>📈 Графики</h2>
    {"".join(f'<div class="chart-box">{c}</div>' for c in charts_html)}
</div>
</body>
</html>"""
    output.write(html.encode('utf-8'))
    output.seek(0)
    return output


def generate_online_pdf_report(env_name, delay_desc, steps, n_runs, algo_details, results_df, all_results):
    """
    Генерирует многостраничный векторный PDF-отчёт об онлайн-эксперименте через Matplotlib PdfPages.
    Адаптивная вёрстка:
    - До 10 алгоритмов: Страница 1 (Leaderboard + Параметры), Страница 2 (4 графика 2x2).
    - Более 10 алгоритмов (до 30+): Страница 1 (Leaderboard), Страница 2 (Параметры), Страница 3 (4 графика 2x2).
    - Палитра из 60 уникальных цветов (tab20 + tab20b + tab20c).
    - Полная очистка эмодзи во избежание пустых квадратов в шрифте DejaVu Sans.
    """
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import numpy as np
    import re

    output = BytesIO()

    # 60 уникальных различимых цветов из палитр Matplotlib tab20, tab20b, tab20c
    colors = [
        *plt.cm.tab20.colors,
        *plt.cm.tab20b.colors,
        *plt.cm.tab20c.colors
    ]

    clean_delay = delay_desc.replace("🎲", "").replace("⏱️", "").replace("🟢", "").strip()
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    meta_str = f"Среда: {env_name}   |   Задержка: {clean_delay}   |   Шагов: {steps}   |   Запусков: {n_runs}   |   Дата: {date_str}"

    # Очищаем таблицу результатов от эмодзи для избежания пустых глифов
    clean_df = results_df.copy()
    if 'Статус' in clean_df.columns:
        clean_df['Статус'] = clean_df['Статус'].astype(str).str.replace('✅', '[OK]').str.replace('❌', '[Err]')

    # Очищаем параметры и категории от эмодзи
    clean_p_rows = []
    for a in algo_details:
        cat_clean = re.sub(r'[^\w\s\(\)\-\.,/]', '', str(a.get('Категория', ''))).strip()
        clean_p_rows.append([a['Алгоритм'], cat_clean, a['Параметры']])

    n_algos = len(all_results)
    multi_page_tables = len(clean_p_rows) > 10

    with PdfPages(output) as pdf:
        if not multi_page_tables:
            # ----------------------------------------------------
            # ОДНА СТРАНИЦА ДЛЯ ОБЕИХ ТАБЛИЦ (до 10 алгоритмов)
            # ----------------------------------------------------
            fig = plt.figure(figsize=(11.69, 8.27))
            ax = fig.add_axes([0.06, 0.05, 0.88, 0.84])
            ax.axis('off')

            plt.suptitle("Отчёт об онлайн-эксперименте (MAB Framework)", fontsize=15, fontweight='bold', y=0.96)
            ax.text(0.5, 0.98, meta_str, fontsize=9, ha='center', va='top', transform=ax.transAxes,
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='#f1f5f9', edgecolor='#cbd5e1'))

            cur_y = 0.90
            # Таблица 1: Leaderboard
            ax.text(0.0, cur_y, "Таблица лидеров (Leaderboard)", fontsize=11, fontweight='bold', va='top', transform=ax.transAxes)
            cur_y -= 0.035

            n_rows_lead = len(clean_df)
            t1_height = min(0.35, 0.045 + n_rows_lead * 0.038)
            t1 = ax.table(cellText=clean_df.values, colLabels=list(clean_df.columns),
                          bbox=[0.0, cur_y - t1_height, 1.0, t1_height], cellLoc='center')
            t1.auto_set_font_size(False)
            t1.set_fontsize(8.5)
            for (r, c), cell in t1.get_celld().items():
                if r == 0:
                    cell.set_facecolor('#2563eb')
                    cell.get_text().set_color('white')
                    cell.get_text().set_weight('bold')
                else:
                    cell.set_facecolor('#f8fafc' if r % 2 == 0 else '#ffffff')

            cur_y -= (t1_height + 0.06)

            # Таблица 2: Параметры
            ax.text(0.0, cur_y, "Параметры алгоритмов", fontsize=11, fontweight='bold', va='top', transform=ax.transAxes)
            cur_y -= 0.035

            n_rows_params = len(clean_p_rows)
            t2_height = min(cur_y - 0.02, 0.045 + n_rows_params * 0.038)
            t2 = ax.table(cellText=clean_p_rows, colLabels=['Алгоритм', 'Категория', 'Точные гиперпараметры'],
                          colWidths=[0.22, 0.20, 0.58],
                          bbox=[0.0, cur_y - t2_height, 1.0, t2_height], cellLoc='left')
            t2.auto_set_font_size(False)
            t2.set_fontsize(8.5)
            for (r, c), cell in t2.get_celld().items():
                if r == 0:
                    cell.set_facecolor('#1e293b')
                    cell.get_text().set_color('white')
                    cell.get_text().set_weight('bold')
                else:
                    cell.set_facecolor('#f8fafc' if r % 2 == 0 else '#ffffff')

            pdf.savefig(fig)
            plt.close(fig)

        else:
            # ----------------------------------------------------
            # ДВЕ СТРАНИЦЫ ДЛЯ ТАБЛИЦ (если алгоритмов > 10, до 30+)
            # ----------------------------------------------------
            # Страница 1: Leaderboard
            fig1 = plt.figure(figsize=(11.69, 8.27))
            ax1 = fig1.add_axes([0.06, 0.08, 0.88, 0.80])
            ax1.axis('off')
            plt.suptitle("Отчёт об онлайн-эксперименте (MAB Framework)", fontsize=15, fontweight='bold', y=0.96)
            ax1.text(0.5, 0.98, meta_str, fontsize=9, ha='center', va='top', transform=ax1.transAxes,
                     bbox=dict(boxstyle='round,pad=0.4', facecolor='#f1f5f9', edgecolor='#cbd5e1'))

            ax1.text(0.0, 0.90, "Таблица лидеров (Leaderboard)", fontsize=11, fontweight='bold', va='top', transform=ax1.transAxes)
            n_rows = len(clean_df)
            t_height = min(0.80, 0.045 + n_rows * 0.026)
            f_size = 7.5 if n_rows > 20 else 8.5
            t1 = ax1.table(cellText=clean_df.values, colLabels=list(clean_df.columns),
                           bbox=[0.0, 0.86 - t_height, 1.0, t_height], cellLoc='center')
            t1.auto_set_font_size(False)
            t1.set_fontsize(f_size)
            for (r, c), cell in t1.get_celld().items():
                if r == 0:
                    cell.set_facecolor('#2563eb')
                    cell.get_text().set_color('white')
                    cell.get_text().set_weight('bold')
                else:
                    cell.set_facecolor('#f8fafc' if r % 2 == 0 else '#ffffff')
            pdf.savefig(fig1)
            plt.close(fig1)

            # Страница 2: Параметры алгоритмов
            fig2 = plt.figure(figsize=(11.69, 8.27))
            ax2 = fig2.add_axes([0.06, 0.08, 0.88, 0.84])
            ax2.axis('off')
            plt.suptitle("Параметры алгоритмов", fontsize=15, fontweight='bold', y=0.96)

            n_rows_p = len(clean_p_rows)
            t_height_p = min(0.85, 0.045 + n_rows_p * 0.026)
            f_size_p = 7.5 if n_rows_p > 20 else 8.5
            t2 = ax2.table(cellText=clean_p_rows, colLabels=['Алгоритм', 'Категория', 'Точные гиперпараметры'],
                           colWidths=[0.22, 0.20, 0.58],
                           bbox=[0.0, 0.92 - t_height_p, 1.0, t_height_p], cellLoc='left')
            t2.auto_set_font_size(False)
            t2.set_fontsize(f_size_p)
            for (r, c), cell in t2.get_celld().items():
                if r == 0:
                    cell.set_facecolor('#1e293b')
                    cell.get_text().set_color('white')
                    cell.get_text().set_weight('bold')
                else:
                    cell.set_facecolor('#f8fafc' if r % 2 == 0 else '#ffffff')
            pdf.savefig(fig2)
            plt.close(fig2)

        # ----------------------------------------------------
        # СТРАНИЦА ГРАФИКОВ: 4 графика регрета (сетка 2x2)
        # ----------------------------------------------------
        fig_plots, axes = plt.subplots(2, 2, figsize=(11.69, 8.27))
        (ax_cum, ax_cum_log), (ax_avg, ax_avg_log) = axes
        t_range = np.arange(1, steps + 1)

        # Динамическая колоночность и размер шрифта легенды под 30+ алгоритмов
        leg_cols = 3 if n_algos > 16 else (2 if n_algos > 8 else 1)
        leg_font = 6.5 if n_algos > 16 else (7.5 if n_algos > 8 else 8.5)

        for idx, (name, data) in enumerate(all_results.items()):
            color = colors[idx % len(colors)]
            if 'cumulative_regret_mean' in data:
                c_data = data['cumulative_regret_mean']
                ax_cum.plot(t_range[:len(c_data)], c_data, label=name, color=color, lw=1.5)
                log_mask = np.array(c_data) > 0
                if np.any(log_mask):
                    ax_cum_log.plot(t_range[:len(c_data)][log_mask], np.array(c_data)[log_mask], label=name, color=color, lw=1.5)

            if 'average_regret_mean' in data:
                a_data = data['average_regret_mean']
                ax_avg.plot(t_range[:len(a_data)], a_data, label=name, color=color, lw=1.5)
                log_mask_a = np.array(a_data) > 0
                if np.any(log_mask_a):
                    ax_avg_log.plot(t_range[:len(a_data)][log_mask_a], np.array(a_data)[log_mask_a], label=name, color=color, lw=1.5)

        for ax_p, title, is_log in [
            (ax_cum, "Cumulative Regret", False),
            (ax_cum_log, "Cumulative Regret (Log Scale)", True),
            (ax_avg, "Average Regret", False),
            (ax_avg_log, "Average Regret (Log Scale)", True)
        ]:
            ax_p.set_title(title, fontsize=10.5, fontweight='bold')
            ax_p.set_xlabel("Шаг", fontsize=8.5)
            ax_p.set_ylabel("Regret", fontsize=8.5)
            if is_log:
                ax_p.set_yscale('log')
            ax_p.grid(True, linestyle='--', alpha=0.4)
            ax_p.legend(fontsize=leg_font, ncol=leg_cols, loc='best')

        plt.suptitle("Графики регрета онлайн-эксперимента", fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0.02, 1, 0.95])

        pdf.savefig(fig_plots)
        plt.close(fig_plots)

    output.seek(0)
    return output