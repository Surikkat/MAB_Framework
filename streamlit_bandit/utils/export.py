import pandas as pd
from datetime import datetime
from io import BytesIO

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