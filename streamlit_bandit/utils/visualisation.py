import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def plot_candidate_comparison(results, baseline_ctr):
    df = pd.DataFrame(results)
    
    fig = go.Figure()

    fig.add_hline(y=baseline_ctr*100, line_dash="dash", 
                  line_color="gray", annotation_text="Продакшн")

    fig.add_trace(go.Bar(
        x=df['candidate'],
        y=[r*100 for r in df['dr_score']],
        marker_color=['#2ecc71' if r > baseline_ctr else '#e74c3c' 
                     for r in df['dr_score']],
        text=[f"{r*100:.2f}%" for r in df['dr_score']],
        textposition='outside'
    ))
    
    fig.update_layout(
        title="CTR кандидатов (Doubly Robust)",
        yaxis_title="CTR (%)",
        showlegend=False,
        height=400
    )
    
    return fig

def plot_effective_sample_size(results):
    df = pd.DataFrame(results)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df['candidate'],
        y=df['effective_sample_size'],
        marker_color='#3498db',
        text=[f"{ess:,.0f}" for ess in df['effective_sample_size']],
        textposition='outside'
    ))
    
    fig.add_hline(y=1000, line_dash="dash", 
                  line_color="orange", annotation_text="Минимум")
    
    fig.update_layout(
        title="Effective Sample Size",
        yaxis_title="ESS",
        showlegend=False,
        height=400
    )
    
    return fig

def plot_method_agreement(results):
    df = pd.DataFrame(results)
    
    methods = []
    for col in ['dm_score', 'ips_score', 'dr_score']:
        if col in df.columns:
            methods.append(col)
    
    fig = go.Figure()
    
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    names = ['Direct Method', 'IPS (clipped)', 'Doubly Robust']
    
    for i, (method, color, name) in enumerate(zip(methods, colors, names)):
        fig.add_trace(go.Scatter(
            x=df['candidate'],
            y=[r*100 for r in df[method]],
            mode='lines+markers',
            name=name,
            line=dict(color=color)
        ))
    
    fig.update_layout(
        title="Согласованность методов оценки",
        yaxis_title="CTR (%)",
        height=400
    )
    
    return fig