import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import numpy as np
import datetime  # Import para a data atual

# --- Configuração da Página ---
st.set_page_config(layout="wide", page_title="JLR Torque integrity", initial_sidebar_state="expanded")


# --- Funções Auxiliares ---
@st.cache_data # Adicionado cache para a função de cálculo de Cp/Cpk
def calculate_cp_cpk(data_series, usl, lsl):
    """Calculates Cp and Cpk for a given data series and specification limits."""
    mean = data_series.mean()
    std_dev = data_series.std()

    if std_dev == 0:
        if lsl <= mean <= usl:
            return float('inf'), float('inf')
        else:
            return 0.0, 0.0

    if usl <= lsl:
        cp = 0.0
    else:
        cp = (usl - lsl) / (6 * std_dev)

    cpu = (usl - mean) / (3 * std_dev)
    cpl = (mean - lsl) / (3 * std_dev)
    cpk = min(cpu, cpl)

    return cp, cpk


@st.cache_data # Adicionado cache para a função
def suggest_optimal_limits(data_series, target_cpk):
    """
    Calculates suggested LSL and USL to achieve a target Cpk,
    assuming the process mean is centered.
    """
    mean = data_series.mean()
    std_dev = data_series.std()

    if std_dev == 0:
        return mean, mean

    half_width = 3 * target_cpk * std_dev

    lsl_suggested = mean - half_width
    usl_suggested = mean + half_width

    return lsl_suggested, usl_suggested


def generate_cp_cpk_analysis(cp_tq, cpk_tq, cp_ang, cpk_ang, df_ok_for_analysis):
    """Generates an intelligent text analysis based on Cp and Cpk values and suggests optimal limits."""
    analysis_text = []

    # --- Analysis for Torque ---
    analysis_text.append("### Análise de Capacidade para Torque (Nm) - Limites Fixos da Engenharia:")
    analysis_text.append(
        "Para o Torque, os limites de especificação (LSL/USL) são **fixos e definidos pela engenharia de produto**, não sendo passíveis de otimização via percentis ou ajuste manual nesta aplicação. A análise de Cp/Cpk abaixo reflete a capacidade do processo de Torque em relação a **esses limites pré-definidos**.")

    if cpk_tq == float('inf'):
        analysis_text.append(
            "O processo para **Torque** demonstra **capacidade perfeita** (Cp = Cpk = Infinito) em relação aos limites de engenharia. Isso indica que, para os dados 'OK' filtrados, a variação é praticamente nula, e todos os valores estão exatamente dentro dos limites de especificação. É um cenário ideal de controle.")
    elif cpk_tq < 1.0:
        analysis_text.append(
            f"O processo para **Torque** apresenta **capacidade insuficiente (Cpk = {cpk_tq:.2f} < 1.0)** em relação aos limites fixos da engenharia. Isso significa que a variação do processo é muito grande ou o processo não está bem centralizado dentro das especificações.")
        if cp_tq < 1.0:
            analysis_text.append(
                f"  - **Potencial (Cp={cp_tq:.2f}):** A variação do processo é maior que a tolerância dos limites, indicando um problema fundamental com a dispersão.")
        else:
            analysis_text.append(
                f"  - **Potencial (Cp={cp_tq:.2f}):** Embora a variação potencial seja aceitável, o processo está descentralizado em relação aos limites de engenharia, o que compromete a capacidade real.")
        analysis_text.append("**Requer atenção e ação imediata.**")
    elif 1.0 <= cpk_tq < 1.33:
        analysis_text.append(
            f"O processo para **Torque** é **marginalmente capaz (Cpk = {cpk_tq:.2f})** em relação aos limites fixos da engenharia. Embora esteja tecnicamente dentro da capacidade, há espaço para melhoria para torná-lo mais robusto.")
        if cp_tq - cpk_tq > 0.1:  # Significant difference indicates centering issue
            analysis_text.append(
                f"  - A diferença entre Cp ({cp_tq:.2f}) e Cpk ({cpk_tq:.2f}) sugere um ligeiro problema de centralização, que deve ser investigado.")
        else:
            analysis_text.append(
                f"  - O processo está razoavelmente bem centralizado para sua variabilidade atual em relação aos limites de engenharia.")
        analysis_text.append("Recomenda-se monitoramento atento e esforços de otimização.")
    elif 1.33 <= cpk_tq < 1.67:
        analysis_text.append(
            f"O processo para **Torque** demonstra **boa capacidade (Cpk = {cpk_tq:.2f})** em relação aos limites fixos da engenharia. É considerado adequado para a maioria das aplicações, indicando que o processo é estável e centrado.")
        if cp_tq - cpk_tq > 0.1:  # Significant difference indicates centering issue
            analysis_text.append(
                f"  - Uma pequena diferença entre Cp ({cp_tq:.2f}) e Cpk ({cpk_tq:.2f}) sugere que, embora capaz, há uma oportunidade para um centramento ainda melhor.")
    else:  # cpk_tq >= 1.67
        analysis_text.append(
            f"O processo para **Torque** é **altamente capaz (Cpk = {cpk_tq:.2f})** em relação aos limites fixos da engenharia. Isso indica um processo muito robusto, com baixa probabilidade de produzir itens fora das especificações, ideal para aplicações Six Sigma.")
        if cp_tq - cpk_tq > 0.1:  # Significant difference indicates centering issue
            analysis_text.append(
                f"  - Apesar da alta capacidade, uma diferença entre Cp ({cp_tq:.2f}) e Cpk ({cpk_tq:.2f}) pode indicar uma pequena oportunidade de otimização no centramento.")

    analysis_text.append("")  # Empty line for spacing

    # --- Analysis for Angle ---
    analysis_text.append("### Análise de Capacidade para Ângulo (°):")
    if cpk_ang == float('inf'):
        analysis_text.append(
            "O processo para **Ângulo** demonstra **capacidade perfeita** (Cp = Cpk = Infinito). Isso indica que, para os dados 'OK' filtrados, a variação é praticamente nula, e todos os valores estão exatamente dentro dos limites de especificação. É um cenário ideal de controle.")
    elif cpk_ang < 1.0:
        analysis_text.append(
            f"O processo para **Ângulo** apresenta **capacidade insuficiente (Cpk = {cpk_ang:.2f} < 1.0)** em relação aos limites propostos. Isso significa que a variação do processo é muito grande ou o processo não está bem centralizado dentro das especificações.")
        if cp_ang < 1.0:
            analysis_text.append(
                f"  - **Potencial (Cp={cp_ang:.2f}):** A variação do processo é maior que a tolerância dos limites, indicando um problema fundamental com a dispersão.")
        else:
            analysis_text.append(
                f"  - **Potencial (Cp={cp_ang:.2f}):** Embora a variação potencial seja aceitável, o processo está descentralizado, o que compromete a capacidade real.")
        analysis_text.append("**Requer atenção e ação imediata.**")
    elif 1.0 <= cpk_ang < 1.33:
        analysis_text.append(
            f"O processo para **Ângulo** é **marginalmente capaz (Cpk = {cpk_ang:.2f})**. Embora esteja tecnicamente dentro da capacidade, há espaço para melhoria para torná-lo mais robusto.")
        if cp_ang - cpk_ang > 0.1:  # Significant difference indicates centering issue
            analysis_text.append(
                f"  - A diferença entre Cp ({cp_ang:.2f}) e Cpk ({cpk_ang:.2f}) sugere um ligeiro problema de centralização, que deve ser investigado.")
        else:
            analysis_text.append(f"  - O processo está razoavelmente bem centralizado para sua variabilidade atual.")
        analysis_text.append("Recomenda-se monitoramento atento e esforços de otimização.")
    elif 1.33 <= cpk_ang < 1.67:
        analysis_text.append(
            f"O processo para **Ângulo** demonstra **boa capacidade (Cpk = {cpk_ang:.2f})**. É considerado adequado para a maioria das aplicações, indicando que o processo é estável e centrado.")
        if cp_ang - cpk_ang > 0.1:  # Significant difference indicates centering issue
            analysis_text.append(
                f"  - Uma pequena diferença entre Cp ({cp_ang:.2f}) e Cpk ({cpk_ang:.2f}) sugere que, embora capaz, há uma oportunidade para um centramento ainda melhor.")
    else:  # cpk_ang >= 1.67
        analysis_text.append(
            f"O processo para **Ângulo** é **altamente capaz (Cpk = {cpk_ang:.2f})**. Isso indica um processo muito robusto, com baixa probabilidade de produzir itens fora das especificações, ideal para aplicações Six Sigma.")
        if cp_ang - cpk_ang > 0.1:  # Significant difference indicates centering issue
            analysis_text.append(
                f"  - Apesar da alta capacidade, uma diferença entre Cp ({cp_ang:.2f}) e Cpk ({cpk_ang:.2f}) pode indicar uma pequena oportunidade de otimização no centramento.")

    analysis_text.append("")  # Empty line for spacing

    # --- Overall Summary ---
    analysis_text.append("### Resumo Geral da Capacidade:")
    analysis_text.append(
        "A capacidade do processo para **Torque** é avaliada em relação aos limites **fixos de engenharia**, enquanto a capacidade para **Ângulo** é avaliada em relação aos **novos limites propostos** (seja por percentis ou manualmente).")
    analysis_text.append(f"- Para Torque: **Cpk = {cpk_tq:.2f}** (ou 'Perfeito') em relação aos limites de engenharia.")
    analysis_text.append(
        f"- Para Ângulo: **Cpk = {cpk_ang:.2f}** (ou 'Perfeito') em relação aos novos limites propostos.")
    analysis_text.append(
        "É fundamental monitorar continuamente esses índices, especialmente o Cpk, para garantir a estabilidade e a centralização do processo dentro dos limites definidos. Um Cpk abaixo de 1.33 geralmente indica necessidade de ação para melhoria do processo.")

    # --- New Section: Optimal Limits Recommendation ---
    analysis_text.append("\n### Recomendações de Limites Otimizados com Base na Capacidade do Processo:\n")
    analysis_text.append("Os limites de Torque são definidos pela engenharia e não são ajustados por esta ferramenta. "
                         "Para o **Ângulo**, podemos sugerir limites que teoricamente atingiriam um determinado Cpk alvo, "
                         "com base na variabilidade inerente dos seus dados 'OK'. Essas sugestões são puramente estatísticas "
                         "e devem ser validadas pela engenharia e viabilidade prática.\n")

    # Define target Cpk values
    target_cpk_good = 1.33
    target_cpk_excellent = 1.67

    # Torque recommendations - Indicate fixed limits
    analysis_text.append("#### Para Torque (Nm):")
    analysis_text.append("Os limites de Torque são definidos pela engenharia (Nominal: "
                         f"`{df_ok_for_analysis['TQmín_nom'].min():.3f} Nm` a "
                         f"`{df_ok_for_analysis['TQmáx_nom'].max():.3f} Nm`). "
                         "Esta aplicação não propõe alterações para esses limites. "
                         "A capacidade calculada para Torque é em relação a esses limites fixos.")

    # Angle recommendations - This part is fine, as it's the only one being optimized.
    analysis_text.append("\n#### Para Ângulo (°):")
    mean_ang_ok = df_ok_for_analysis['ÂNG_rea'].mean()
    std_ang_ok = df_ok_for_analysis['ÂNG_rea'].std()

    if std_ang_ok == 0:
        analysis_text.append(
            f"  - O processo de Ângulo tem **variação nula** nos dados 'OK' filtrados. Isso significa que ele já é intrinsecamente capaz de atender a qualquer limite de engenharia razoável que contenha o valor médio de {mean_ang_ok:.3f}°. Os limites atuais que você definiu são perfeitamente adequados, ou pode-se considerar um único valor alvo.")
    else:
        lsl_ang_good, usl_ang_good = suggest_optimal_limits(df_ok_for_analysis['ÂNG_rea'], target_cpk_good)
        lsl_ang_excellent, usl_ang_excellent = suggest_optimal_limits(df_ok_for_analysis['ÂNG_rea'],
                                                                      target_cpk_excellent)

        analysis_text.append(f"  - **Meta Cpk $\ge$ {target_cpk_good:.2f} (Processo Bom):**")
        analysis_text.append(f"    - Limite Inferior Sugerido: `{lsl_ang_good:.3f}°`")
        analysis_text.append(f"    - Limite Superior Sugerido: `{usl_ang_good:.3f}°`")
        analysis_text.append(f"    - (Largura total: `{usl_ang_good - lsl_ang_good:.3f}°`)\n")

        analysis_text.append(f"  - **Meta Cpk $\ge$ {target_cpk_excellent:.2f} (Processo Altamente Capaz / 6 Sigma):**")
        analysis_text.append(f"    - Limite Inferior Sugerido: `{lsl_ang_excellent:.3f}°`")
        analysis_text.append(f"    - Limite Superior Sugerido: `{usl_ang_excellent:.3f}°`")
        analysis_text.append(f"    - (Largura total: `{usl_ang_excellent - lsl_ang_excellent:.3f}°`)\n")

    analysis_text.append(
        "\n_**Nota:** Estas são sugestões estatísticas para otimizar o Cpk do **Ângulo** dada a variabilidade observada. É crucial que esses limites sejam avaliados em termos de **engenharia, segurança e funcionalidade do produto.** Limites muito apertados, embora estatisticamente ótimos, podem não ser viáveis na prática ou gerar muitos 'NOKs'' falsos se a causa da variabilidade não for controlada._")

    return "".join(analysis_text)


@st.cache_data # Adicionado cache para a função
def generate_comparison_chart(df_filtered, current_ang_min, current_tq_min, current_ang_max, current_tq_max,
                              new_ang_min, new_tq_min, new_ang_max, new_tq_max, chart_title):
    colors = {'OK': 'green', 'NOK': 'red'}
    avaliacoes_presentes = df_filtered['Avaliação'].unique().tolist()

    fig = go.Figure()

    fig.add_shape(
        type="rect",
        x0=current_ang_min,
        y0=current_tq_min,
        x1=current_ang_max,
        y1=current_tq_max,
        line=dict(color="RoyalBlue", width=2),
        fillcolor="LightSkyBlue",
        opacity=0.3,
        layer="below",
        name="Janela Nominal"
    )
    fig.add_annotation(
        x=(current_ang_min + current_ang_max) / 2,
        y=(current_tq_min + current_tq_max) / 2,
        text="Janela Nominal Atual",
        showarrow=False,
        font=dict(color="RoyalBlue", size=10),
        yanchor="middle",
        xanchor="center"
    )

    fig.add_shape(
        type="rect",
        x0=new_ang_min,
        y0=new_tq_min,  # Torque limits are nominal
        x1=new_ang_max,
        y1=new_tq_max,  # Torque limits are nominal
        line=dict(color="DarkOrange", width=2, dash="dot"),  # Cor diferente para manual/otimizado
        fillcolor="LightSalmon",
        opacity=0.4,
        layer="above",
        name="Janela Proposta"
    )
    fig.add_annotation(
        x=(new_ang_min + new_ang_max) / 2,
        y=(new_tq_min + new_tq_max) / 2,  # Centered vertically using nominal torque limits
        text="Janela Proposta",
        showarrow=False,
        font=dict(color="DarkOrange", size=10),
        yanchor="middle",
        xanchor="center"
    )

    for status in ['OK', 'NOK']:
        if status in avaliacoes_presentes:
            df_status = df_filtered[df_filtered['Avaliação'] == status]
            fig.add_trace(go.Scatter(
                x=df_status['ÂNG_rea'],
                y=df_status['TQ_rea'],
                mode='markers',
                name=f'Pontos {status}',
                marker=dict(color=colors[status], size=8, opacity=0.7),
                customdata=df_status[['Avaliação', 'GP', 'Ferramenta']],
                hovertemplate=
                '<b>Avaliação:</b> %{customdata[0]}<br>' +
                '<b>Torque Real:</b> %{y:.2f}<br>' +
                '<b>Ângulo Real:</b> %{x:.2f}<br>' +
                '<b>GP:</b> %{customdata[1]}<br>' +
                '<b>Ferramenta:</b> %{customdata[2]}<extra></extra>'
            ))

    fig.update_layout(
        title=chart_title,
        xaxis_title="Ângulo Real Aplicado (°)",
        yaxis_title="Torque Real Aplicado (Nm)",
        hovermode="closest",
        showlegend=True,
        width=1000,
        height=600
    )
    return fig


@st.cache_data # Adicionado cache para a função
def generate_histograms(df_filtered):
    colors = {'OK': 'green', 'NOK': 'red'}
    figs = {}

    # Torque Histogram
    fig_tq_hist = px.histogram(df_filtered, x='TQ_rea', color='Avaliação',
                               marginal="box",
                               title="Histograma de Torque Real por Avaliação",
                               labels={'TQ_rea': 'Torque Real (Nm)'},
                               color_discrete_map=colors)
    fig_tq_hist.update_layout(bargap=0.1)

    global_tq_min_data = df_filtered['TQ_rea'].min()
    global_tq_max_data = df_filtered['TQ_rea'].max()
    x_range_tq_global = np.linspace(global_tq_min_data, global_tq_max_data, 500)

    for status in df_filtered['Avaliação'].unique():
        df_status_for_curve = df_filtered[df_filtered['Avaliação'] == status]
        if not df_status_for_curve.empty:
            mean_tq = df_status_for_curve['TQ_rea'].mean()
            std_tq = df_status_for_curve['TQ_rea'].std()

            if std_tq > 0:
                pdf_values_tq = stats.norm.pdf(x_range_tq_global, mean_tq, std_tq)

                counts, bins = np.histogram(df_status_for_curve['TQ_rea'], bins='auto')
                if len(bins) > 1:
                    bin_width_tq = bins[1] - bins[0]
                    scaled_pdf_values_tq = pdf_values_tq * len(df_status_for_curve) * bin_width_tq
                else:
                    scaled_pdf_values_tq = pdf_values_tq * len(df_status_for_curve)

                fig_tq_hist.add_trace(go.Scatter(
                    x=x_range_tq_global,
                    y=scaled_pdf_values_tq,
                    mode='lines',
                    name=f'Curva Normal {status}',
                    line=dict(color=colors[status], dash='dash', width=2),
                    showlegend=True
                ))
    figs['torque_histogram'] = fig_tq_hist

    # Angle Histogram
    fig_ang_hist = px.histogram(df_filtered, x='ÂNG_rea', color='Avaliação',
                                marginal="box",
                                title="Histograma de Ângulo Real por Avaliação",
                                labels={'ÂNG_rea': 'Ângulo Real (°)'},
                                color_discrete_map=colors)
    fig_ang_hist.update_layout(bargap=0.1)

    global_ang_min_data = df_filtered['ÂNG_rea'].min()
    global_ang_max_data = df_filtered['ÂNG_rea'].max()
    x_range_ang_global = np.linspace(global_ang_min_data, global_ang_max_data, 500)

    for status in df_filtered['Avaliação'].unique():
        df_status_for_curve = df_filtered[df_filtered['Avaliação'] == status]
        if not df_status_for_curve.empty:
            mean_ang = df_status_for_curve['ÂNG_rea'].mean()
            std_ang = df_status_for_curve['ÂNG_rea'].std()
            if std_ang > 0:
                pdf_values_ang = stats.norm.pdf(x_range_ang_global, mean_ang, std_ang)

                counts, bins = np.histogram(df_status_for_curve['ÂNG_rea'], bins='auto')
                if len(bins) > 1:
                    bin_width_ang = bins[1] - bins[0]
                    scaled_pdf_values_ang = pdf_values_ang * len(df_status_for_curve) * bin_width_ang
                else:
                    scaled_pdf_values_ang = pdf_values_ang * len(df_status_for_curve)

                fig_ang_hist.add_trace(go.Scatter(
                    x=x_range_ang_global,
                    y=scaled_pdf_values_ang,
                    mode='lines',
                    name=f'Curva Normal {status}',
                    line=dict(color=colors[status], dash='dash', width=2),
                    showlegend=True
                ))
    figs['angle_histogram'] = fig_ang_hist
    return figs


# --- Título da Aplicação ---
st.title("Análise e Otimização da Janela de Aperto - JLR Torque integrity")
with st.expander("💡 Entendimento da Aplicação"):
    st.markdown("""
    Esta aplicação permite carregar dados de aperto, visualizar a relação entre torque e ângulo,
    e aplicar metodologias estatísticas e manuais para propor uma **nova e mais precisa 'janela de aperto'**,
    visando aumentar a precisão do controle de qualidade e reduzir a possibilidade de erro operacional.
    **Importante:** Os limites de Torque são definidos pela engenharia de produto do Reino Unido e não são alterados por esta aplicação. A otimização e restrição da janela de aperto foca-se exclusivamente nos limites de **Ângulo**.
    Desenvolvido por: Eng Diógenes Oliveira
    """)

# --- 0. Inicialização do Session State para Compartilhamento de Dados ---
# Isso é crucial para que o DataFrame filtrado esteja disponível entre as abas.
if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_filtered' not in st.session_state:
    st.session_state.df_filtered = None
if 'current_tq_min' not in st.session_state:
    st.session_state.current_tq_min = 0.0
if 'current_tq_max' not in st.session_state:
    st.session_state.current_tq_max = 0.0
if 'current_ang_min' not in st.session_state:
    st.session_state.current_ang_min = 0.0
if 'current_ang_max' not in st.session_state:
    st.session_state.current_ang_max = 0.0

# Initialize session state variables for manual tab results (for report tab)
# Estes são inicializados com 0.0 e depois preenchidos pelos cálculos ou widgets
if 'manual_tq_min' not in st.session_state:
    st.session_state.manual_tq_min = 0.0
if 'manual_tq_max' not in st.session_state:
    st.session_state.manual_tq_max = 0.0
if 'manual_ang_min' not in st.session_state:
    st.session_state.manual_ang_min = 0.0 # Valor inicial
if 'manual_ang_max' not in st.session_state:
    st.session_state.manual_ang_max = 0.0 # Valor inicial
if 'cp_tq_manual' not in st.session_state:
    st.session_state.cp_tq_manual = 0.0
if 'cpk_tq_manual' not in st.session_state:
    st.session_state.cpk_tq_manual = 0.0
if 'cp_ang_manual' not in st.session_state:
    st.session_state.cp_ang_manual = 0.0
if 'cpk_ang_manual' not in st.session_state:
    st.session_state.cpk_ang_manual = 0.0
if 'percent_total_manual' not in st.session_state:
    st.session_state.percent_total_manual = 0.0

# Initialize session state for dynamic form fields in the new report tab
if 'num_steps' not in st.session_state:
    st.session_state.num_steps = 1
if 'num_spacers' not in st.session_state:
    st.session_state.num_spacers = 1
if 'num_resp_analise' not in st.session_state:
    st.session_state.num_resp_analise = 1
if 'num_resp_aprov' not in st.session_state:
    st.session_state.num_resp_aprov = 1
if 'resultado_aprovado' not in st.session_state: # Usado para o st.radio
    st.session_state.resultado_aprovado = None

# --- Upload de Arquivo ---
st.header("1. Carregar Dados de Aperto")
uploaded_file = st.file_uploader("Arraste e solte seu arquivo CSV aqui", type=["csv"])

if uploaded_file is not None:
    try:
        # Carregar e pré-processar o DataFrame
        df_loaded = pd.read_csv(uploaded_file, encoding='latin-1')

        required_columns = [
            'TQ_rea', 'TQmín_nom', 'TQmáx_nom',
            'ÂNG_rea', 'ÂNGmín', 'ÂNGmáx_nom',
            'Avaliação', 'GP', 'Ferramenta'
        ]
        if not all(col in df_loaded.columns for col in required_columns):
            st.error(f"O CSV deve conter as seguintes colunas: {', '.join(required_columns)}")
            st.stop()

        for col in ['TQ_rea', 'TQmín_nom', 'TQmáx_nom', 'ÂNG_rea', 'ÂNGmín', 'ÂNGmáx_nom']:
            if df_loaded[col].dtype == 'object':
                df_loaded[col] = df_loaded[col].str.replace(',', '.').astype(float)

        st.session_state.df = df_loaded.copy()  # Armazena o DataFrame original no session state
        st.success("Arquivo CSV carregado com sucesso!")

    except Exception as e:
        st.error(f"Ocorreu um erro ao processar o arquivo: {e}")
        st.info(
            "Por favor, verifique se o arquivo é um CSV válido, se as colunas estão corretas e tente uma codificação diferente (ex: 'windows-1252' ou 'utf-8').")
        st.exception(e)
        st.session_state.df = None  # Reseta o df em caso de erro

if st.session_state.df is None:
    st.info("Aguardando o upload de um arquivo CSV.")
    st.stop()  # Interrompe a execução se não houver arquivo carregado

# --- Sidebar de Filtros (aplicada ao df carregado) ---
st.sidebar.header("Filtros de Análise")

# Filtros que atuam sobre st.session_state.df
opcoes_avaliacao = ['Todos'] + sorted(st.session_state.df['Avaliação'].unique().tolist())
avaliacao_selecionada = st.sidebar.selectbox("Filtrar por Avaliação:", opcoes_avaliacao)

grupamentos_unicos = ['Todos'] + sorted(st.session_state.df['GP'].unique().tolist())
gp_selecionado = st.sidebar.selectbox("Filtrar por Grupamento (GP):", grupamentos_unicos)

ferramentas_unicas = ['Todas'] + sorted(st.session_state.df['Ferramenta'].unique().tolist())
ferramenta_selecionada = st.sidebar.selectbox("Filtrar por Ferramenta:", ferramentas_unicas)

# Cria o df_filtered com base nos filtros da sidebar
df_temp_filtered = st.session_state.df.copy()
if avaliacao_selecionada != 'Todos':
    df_temp_filtered = df_temp_filtered[df_temp_filtered['Avaliação'] == avaliacao_selecionada]
if gp_selecionado != 'Todos':
    df_temp_filtered = df_temp_filtered[df_temp_filtered['GP'] == gp_selecionado]
if ferramenta_selecionada != 'Todas':
    df_temp_filtered = df_temp_filtered[df_temp_filtered['Ferramenta'] == ferramenta_selecionada]

st.session_state.df_filtered = df_temp_filtered.copy()

if st.session_state.df_filtered.empty:
    st.warning("Nenhum dado encontrado com os filtros selecionados. Por favor, ajuste seus filtros.")
    st.stop()

# Calcular os limites globais da janela de aperto nominal a partir dos dados FILTRADOS e armazenar no session_state
st.session_state.current_tq_min = st.session_state.df_filtered['TQmín_nom'].min()
st.session_state.current_tq_max = st.session_state.df_filtered['TQmáx_nom'].max()
st.session_state.current_ang_min = st.session_state.df_filtered['ÂNGmín'].min()
st.session_state.current_ang_max = st.session_state.df_filtered['ÂNGmáx_nom'].max()

# --- ABAS DE NAVEGAÇÃO ---
tab1, tab2, tab3 = st.tabs(["Otimização por Percentis", "Definição Manual da Janela", "Relatório de Análise"])

with tab1:
    st.header("Análise Detalhada: Otimização por Percentis")

    # --- EXIBIÇÃO DO DATAFRAME FILTRADO ---
    st.subheader("Primeiras linhas do DataFrame com filtros aplicados:")
    st.dataframe(st.session_state.df_filtered.head())

    # --- TEXTO: INTERVALO NOMINAL ATUAL PARA OS DADOS FILTRADOS ---
    with st.expander("ℹ️ Intervalo Nominal Atual (Detalhes)"):
        st.info(f"""
        Para os dados atualmente filtrados (Grupamento: **{gp_selecionado}**, Ferramenta: **{ferramenta_selecionada}**, Avaliação: **{avaliacao_selecionada}**),
        os limites nominais atuais (originais do seu CSV) são:
        - **Torque (TQ):** de `{st.session_state.current_tq_min:.3f} Nm` a `{st.session_state.current_tq_max:.3f} Nm`
        - **Ângulo (ÂNG):** de `{st.session_state.current_ang_min:.3f}°` a `{st.session_state.current_ang_max:.3f}°`
        """)

    # --- 2. Visualização da Janela de Aperto Atual ---
    st.header("2. Visualização da Janela de Aperto Atual e Pontos Reais")
    with st.expander("📊 Sobre o Gráfico de Dispersão"):
        st.markdown(
            "O gráfico de dispersão abaixo ilustra a relação entre o torque real e o ângulo real, e a área hachurada representa a janela de aperto nominal atual. Observe como os pontos 'OK' e 'NOK' se distribuem em relação a esta janela, que muitas vezes é mais ampla do que o necessário.")

    fig_scatter = go.Figure()

    # Adicionar a área da janela de aperto nominal
    fig_scatter.add_shape(
        type="rect",
        x0=st.session_state.current_ang_min,
        y0=st.session_state.current_tq_min,
        x1=st.session_state.current_ang_max,
        y1=st.session_state.current_tq_max,
        line=dict(color="RoyalBlue", width=2),
        fillcolor="LightSkyBlue",
        opacity=0.3,
        layer="below",
        name="Janela Nominal"
    )
    fig_scatter.add_annotation(
        x=(st.session_state.current_ang_min + st.session_state.current_ang_max) / 2,
        y=(st.session_state.current_tq_min + st.session_state.current_tq_max) / 2,
        text="Janela Nominal Atual",
        showarrow=False,
        font=dict(color="RoyalBlue", size=10),
        yanchor="middle",
        xanchor="center"
    )

    # Adicionar os pontos de aperto reais, coloridos por Avaliação
    colors = {'OK': 'green', 'NOK': 'red'}
    avaliacoes_presentes = st.session_state.df_filtered['Avaliação'].unique().tolist()

    for status in ['OK', 'NOK']:
        if status in avaliacoes_presentes:
            df_status = st.session_state.df_filtered[st.session_state.df_filtered['Avaliação'] == status]
            fig_scatter.add_trace(go.Scatter(
                x=df_status['ÂNG_rea'],
                y=df_status['TQ_rea'],
                mode='markers',
                name=f'Pontos {status}',
                marker=dict(color=colors[status], size=8, opacity=0.7),
                hovertemplate=
                '<b>Avaliação:</b> %{customdata[0]}<br>' +
                '<b>Torque Real:</b> %{y:.2f}<br>' +
                '<b>Ângulo Real:</b> %{x:.2f}<br>' +
                '<b>GP:</b> %{customdata[1]}<br>' +
                '<b>Ferramenta:</b> %{customdata[2]}<extra></extra>',
                customdata=df_status[['Avaliação', 'GP', 'Ferramenta']]
            ))

    st.plotly_chart(fig_scatter, use_container_width=True, key="scatter_tab1_original") # Adicionada key

    # --- 3. Análise da Distribuição dos Dados Reais ---
    st.header("3. Análise da Distribuição dos Dados Reais")
    with st.expander("📈 Sobre os Histogramas com Curva Normal"):
        st.markdown("""
        Os histogramas abaixo mostram a distribuição dos valores de Torque Real e Ângulo Real para os dados filtrados, separados por avaliação 'OK'/'NOK'. A **curva normal** sobreposta ilustra a distribuição teórica normal (Gaussiana) com a média e desvio padrão dos seus dados. Isso ajuda a entender a dispersão e o centramento dos seus dados atuais, e se eles seguem uma distribuição aproximadamente normal.
        """)

    col1, col2 = st.columns(2)

    # REFACTOR: Usando a função generate_histograms para evitar duplicação de código
    histograms_tab1 = generate_histograms(st.session_state.df_filtered)

    with col1:
        st.subheader("Distribuição do Torque Real (TQ_rea)")
        st.plotly_chart(histograms_tab1['torque_histogram'], use_container_width=True, key="tq_hist_tab1") # Adicionada key

    with col2:
        st.subheader("Distribuição do Ângulo Real (ÂNG_rea)")
        st.plotly_chart(histograms_tab1['angle_histogram'], use_container_width=True, key="ang_hist_tab1") # Adicionada key

    # --- 4. Proposta de Nova Janela de Aperto Otimizada e Mais Restritiva ---
    st.header("4. Proposta de Nova Janela de Aperto Otimizada para o Ângulo")
    with st.expander("⚙️ Detalhes da Otimização da Janela"):
        st.markdown("""
        O objetivo é **reduzir e otimizar a largura do ÂNGULO da 'janela de aperto'** com base no comportamento dos apertos considerados **"OK"** para os dados **atualmente filtrados**. Isso visa diminuir a tolerância e aumentar a precisão do seu processo, **reduzindo a possibilidade de falhas ou de "falsos OKs"** no que diz respeito ao ângulo. Os limites de Torque são definidos pela engenharia e não são alterados por esta aplicação.

        **Metodologia Sugerida: Percentis dos Dados 'OK'**
        Calcularemos os percentis para `ÂNG_rea` dos dados **"OK"** do subconjunto filtrado. Isso nos dará uma janela que engloba a maior parte dos apertos bem-sucedidos para essa combinação específica de Grupamento e Ferramenta, permitindo uma restrição mais rigorosa e realista do que seu processo consegue entregar consistentemente.
        """)

    df_ok_for_optimization = st.session_state.df_filtered[st.session_state.df_filtered['Avaliação'] == 'OK']

    if df_ok_for_optimization.empty:
        st.warning("""
        Não há dados 'OK' no subconjunto de dados atualmente filtrado para propor uma nova janela de otimização.
        Por favor, ajuste seus filtros (Grupamento, Ferramenta, Avaliação) para incluir dados 'OK' na sua seleção.
        """)
    else:
        # Opção de percentil
        st.subheader("Ajuste dos Percentis para Definir a Janela Otimizada do Ângulo")
        with st.expander("🖐️ Como Ajustar os Percentis?"):
            st.markdown("""
            Utilize os sliders abaixo para definir os percentis inferior e superior para o **Ângulo**. Esses percentis determinarão o quão "apertada" ou "permissiva" será a nova janela de aceitação para o ângulo. **Os limites de Torque permanecem fixos e definidos pela engenharia.**

            -   **Percentil Inferior:** Define o limite mínimo. Se você escolher 0.1%, a nova janela será o valor acima do qual 99.9% dos apertos "OK" se encontram (ou seja, apenas 0.1% dos apertos "OK" mais baixos são desconsiderados). **Para tornar a janela MENOS restritiva no limite inferior (i.e., mais permissiva em valores baixos), DIMINUA este valor.**
            -   **Percentil Superior:** Define o limite máximo. Se você escolher 99.9%, a nova janela será o valor abaixo do qual 99.9% dos apertos "OK" se encontram (ou seja, apenas 0.1% dos apertos "OK" mais altos são desconsiderados). **Para tornar a janela MENOS restritiva no limite superior (i.e., mais permissiva em valores altos), AUMENTE este valor.**

            O objetivo é encontrar um equilíbrio entre a precisão (janela mais restrita no ângulo) e a realidade operacional (janela que não gere alarmes excessivos para variações aceitáveis do operador). Os valores padrão (0.1% e 99.9%) já oferecem uma janela significativamente menos restritiva que 5% e 95%.
            """)

        # Ajustado padrão dos sliders para 0.1 e 99.9 para ser menos restritivo inicialmente
        percentil_inferior = st.slider("Percentil Inferior do Ângulo (ex: 0.1% para tolerância mínima)", 0.0, 5.0, 0.1,
                                       0.1, key="percentil_inferior_slider")
        percentil_superior = st.slider("Percentil Superior do Ângulo (ex: 99.9% para tolerância máxima)", 95.0, 100.0,
                                       99.9, 0.1, key="percentil_superior_slider")

        # Torque limits remain the current nominal limits
        tq_novo_min = st.session_state.current_tq_min
        tq_novo_max = st.session_state.current_tq_max
        # Angle limits are optimized by percentiles
        ang_novo_min = df_ok_for_optimization['ÂNG_rea'].quantile(percentil_inferior / 100)
        ang_novo_max = df_ok_for_optimization['ÂNG_rea'].quantile(percentil_superior / 100)

        # --- TEXTO: NOVO INTERVALO OTIMIZADO ---
        with st.expander("✅ Novo Intervalo Otimizado (Detalhes)"):
            st.success(f"""
            O **novo intervalo otimizado e mais restritivo** (baseado nos dados 'OK' do filtro atual e nos percentis {percentil_inferior}% e {percentil_superior}%) é:
            - **Limite de Torque (TQ_rea):** Permanece o nominal, de `{tq_novo_min:.3f} Nm` a `{tq_novo_max:.3f} Nm` (definido pela engenharia).
            - **Novo Limite Mínimo de Ângulo (ÂNG_rea):** `{ang_novo_min:.3f}°`
            - **Novo Limite Máximo de Ângulo (ÂNG_rea):** `{ang_novo_max:.3f}°`
            """)

        # --- DATAFRAME DE COMPARAÇÃO DE LIMITES ---
        st.subheader("Comparativo de Limites: Atual vs. Proposto (Torque fixo, Ângulo otimizado)")
        data_limites = {
            'Parâmetro': ['Torque Mínimo (Nm)', 'Torque Máximo (Nm)', 'Ângulo Mínimo (°)', 'Ângulo Máximo (°)'],
            'Limite Nominal Atual': [st.session_state.current_tq_min, st.session_state.current_tq_max,
                                     st.session_state.current_ang_min, st.session_state.current_ang_max],
            'Novo Limite Otimizado': [tq_novo_min, tq_novo_max, ang_novo_min, ang_novo_max]
            # tq_novo_min/max are current nominal
        }
        df_limites = pd.DataFrame(data_limites)
        st.dataframe(df_limites.set_index('Parâmetro'))

        # --- CÁLCULO E EXIBIÇÃO DE ST.METRICS DE REDUÇÃO DE ÁREA E DADOS REAIS ---
        st.subheader("Métricas de Desempenho, Redução da Janela e Capacidade do Processo")

        # Metrics for Real Data Overview
        col_data1, col_data2, col_data3 = st.columns(3)
        with col_data1:
            st.metric(label="Total de Apertos Avaliados", value=len(st.session_state.df_filtered))
        with col_data2:
            st.metric(label="Torque Real Mínimo (Nm)", value=f"{st.session_state.df_filtered['TQ_rea'].min():.3f}")
            st.metric(label="Torque Real Máximo (Nm)", value=f"{st.session_state.df_filtered['TQ_rea'].max():.3f}")
        with col_data3:
            st.metric(label="Ângulo Real Mínimo (°)", value=f"{st.session_state.df_filtered['ÂNG_rea'].min():.3f}")
            st.metric(label="Ângulo Real Máximo (°)", value=f"{st.session_state.df_filtered['ÂNG_rea'].max():.3f}")

        st.markdown("---")  # Separador visual

        # Metrics for Area Reduction
        area_nominal_tq = st.session_state.current_tq_max - st.session_state.current_tq_min
        area_otimizada_tq = st.session_state.current_tq_max - st.session_state.current_tq_min  # Torque width remains nominal

        area_nominal_ang = st.session_state.current_ang_max - st.session_state.current_ang_min
        area_otimizada_ang = ang_novo_max - ang_novo_min

        percent_tq = 0.0  # Torque reduction is 0%
        # No need for calculation, it's always 0% as per requirements

        percent_ang = 0
        if area_nominal_ang > 0:
            percent_ang = ((area_nominal_ang - area_otimizada_ang) / area_nominal_ang) * 100

        area_nominal_total = area_nominal_tq * area_nominal_ang
        area_otimizada_total = area_otimizada_tq * area_otimizada_ang  # This implicitly uses area_nominal_tq for torque part

        percent_total = 0
        if area_nominal_total > 0:
            percent_total = ((area_nominal_total - area_otimizada_total) / area_nominal_total) * 100

        col_met1, col_met2, col_met3 = st.columns(3)
        with col_met1:
            st.metric(label="Redução na Largura de Torque", value=f"{percent_tq:.2f}%",
                      delta=f"Permanece em {area_nominal_tq:.3f} Nm (definido pela engenharia)", delta_color="off")
        with col_met2:
            st.metric(label="Redução na Largura de Ângulo", value=f"{percent_ang:.2f}%",
                      delta=f"De {area_nominal_ang:.3f} para {area_otimizada_ang:.3f}", delta_color="inverse")
        with col_met3:
            st.metric(label="Redução na Área Total da Janela", value=f"{percent_total:.2f}%",
                      delta=f"De {area_nominal_total:.3f} para {area_otimizada_total:.3f}", delta_color="inverse")

        st.markdown("---")  # Separador visual

        # Capacidade do Processo (Cp/Cpk)
        col_cp1, col_cp2, col_cp3, col_cp4 = st.columns(4)

        # Cp/Cpk for Torque always uses the nominal limits
        cp_tq, cpk_tq = calculate_cp_cpk(df_ok_for_optimization['TQ_rea'], st.session_state.current_tq_max,
                                         st.session_state.current_tq_min)
        # Cp/Cpk for Angle uses the new optimized limits
        cp_ang, cpk_ang = calculate_cp_cpk(df_ok_for_optimization['ÂNG_rea'], ang_novo_max, ang_novo_min)

        with col_cp1:
            if cp_tq == float('inf'):
                st.metric(label="Cp Torque", value="Perfeito")
            else:
                st.metric(label="Cp Torque", value=f"{cp_tq:.2f}")
        with col_cp2:
            if cpk_tq == float('inf'):
                st.metric(label="Cpk Torque", value="Perfeito")
            else:
                st.metric(label="Cpk Torque", value=f"{cpk_tq:.2f}")
        with col_cp3:
            if cp_ang == float('inf'):
                st.metric(label="Cp Ângulo", value="Perfeito")
            else:
                st.metric(label="Cp Ângulo", value=f"{cp_ang:.2f}")
        with col_cp4:
            if cpk_ang == float('inf'):
                st.metric(label="Cpk Ângulo", value="Perfeito")
            else:
                st.metric(label="Cpk Ângulo", value=f"{cpk_ang:.2f}")

        # Intelligent Cp/Cpk Analysis Text Box
        with st.expander("📝 Interpretação e Recomendações de Limites (Cp/Cpk)"):
            # Pass df_ok_for_optimization to the analysis function for optimal limits calculation
            st.markdown(generate_cp_cpk_analysis(cp_tq, cpk_tq, cp_ang, cpk_ang, df_ok_for_optimization))

        st.markdown("---")  # Separador visual

        st.markdown("#### Comparação Visual: Janela Nominal vs. Janela Otimizada por Percentis")
        # Este gráfico é crucial para comparação visual e por isso não está em um expander.

        fig_optimized = generate_comparison_chart(
            st.session_state.df_filtered,
            st.session_state.current_ang_min, st.session_state.current_tq_min,
            st.session_state.current_ang_max, st.session_state.current_tq_max,
            ang_novo_min, tq_novo_min, ang_novo_max, tq_novo_max,
            "Comparação Visual: Janela Nominal vs. Janela Otimizada por Percentis"
        )

        st.plotly_chart(fig_optimized, use_container_width=True, key="comparison_optimized_tab1") # Adicionada key

        with st.expander("❓ Como os percentis ajudam a otimizar e restringir a janela (para Ângulo)?"):
            st.markdown("""
            Os percentis atuam como uma ferramenta direta para você definir o "ponto ótimo" de confiança e restrição desejado **para o Ângulo**:
            -   **Controlam a Abrangência do Ângulo:** Ao escolher percentis como 0.1% e 99.9%, você está definindo que a nova janela para o ângulo deve conter 99.8% dos apertos "OK" mais consistentes do seu histórico. Os 0.2% extremos (0.1% abaixo, 0.1% acima) são considerados variações menos ideais.
            -   **Foco no Desempenho Real do Ângulo:** A "otimização" aqui não é puramente algorítmica para um único ponto fixo, mas sim uma decisão estratégica para alinhar os limites com o **desempenho real e desejado** do processo de ângulo. Se o seu processo "OK" de ângulo nunca foi abaixo de um certo valor, por exemplo, não faz sentido ter um limite mínimo muito mais baixo.
            -   **Robustez a Outliers:** Percentis são menos sensíveis a outliers do que a média e desvio padrão. Isso significa que a janela proposta para o ângulo reflete com fidelidade a variação natural do seu processo "OK", sem ser distorcida por eventos extremos isolados.
            """)

        # --- 5. Próximos Passos e Recomendações ---
        st.header("5. Próximos Passos e Recomendações")
        with st.expander("➡️ Próximos Passos e Validação"):
            st.markdown("""
            Para validar e implementar esta nova janela de aperto **mais restritiva no Ângulo**, sugiro as seguintes ações:

            1.  **Validação Piloto:** Implemente os novos limites de **ângulo** em um grupo menor de ferramentas ou GPs para monitoramento. Isso permite um controle mais seguro durante a transição, avaliando o impacto real na taxa de "NOK". Os limites de Torque permanecem os originais de engenharia.
            2.  **Monitoramento Contínuo:** Utilize as novas janelas de **ângulo** e monitore de perto a proporção de "NOK" (rejeitos) e "OK". Com uma janela mais apertada para o ângulo, os "NOK" deverão ser indicativos de desvios reais e não de variações aceitáveis dentro de uma tolerância excessivamente larga.
            3.  **Análise de Capacidade do Processo (Cp/Cpk):** Com os novos limites de **ângulo** mais rigorosos (e os limites fixos de torque), realize uma análise de capacidade do processo para cada `GP` e `Ferramenta` individualmente. Isso quantificará quão bem seu processo está atendendo a essas especificações mais apertadas e qual a margem de segurança.
                *   `Cp = (USL - LSL) / (6 * σ)`: Índice de capacidade potencial do processo.
                *   `Cpk = min((X̄ - LSL) / (3 * σ), (USL - X̄) / (3 * σ))`: Índice de capacidade real do processo, considerando o centramento.
            4.  **Otimização de Parâmetros da Ferramenta:** Se, mesmo com os dados "OK", ainda houver uma variação significativa no **ângulo**, considere otimizar os parâmetros de aperto das ferramentas (velocidade, rampa, etc.) para buscar uma dispersão ainda menor.
            5.  **Manutenção Preditiva:** Desvios constantes ou aumento de "NOK" após a implementação da janela mais restritiva no **ângulo** em uma ferramenta específica podem indicar a necessidade urgente de calibração ou manutenção preventiva.

            Lembre-se, o objetivo é ter uma janela de aperto que seja **realista** para a sua capacidade de produção (baseada no que o "OK" realmente produz) e que ao mesmo tempo sirva como um **limite eficaz** para identificar falhas genuínas, sem ser excessivamente permissiva. A restrição é aplicada somente ao Ângulo.
            """)

        # --- SEÇÃO: CONSIDERAÇÕES FINAIS E CONFIANÇA ---
        st.header("6. Considerações Finais e Confiança da Metodologia")
        with st.expander("✨ Conclusões e Confiança da Metodologia"):
            st.markdown("""
            A metodologia proposta nesta aplicação baseia-se na análise dos dados históricos dos apertos considerados **"OK"** para **restringir e otimizar** a janela de aceitação para o **Ângulo**. Os limites de Torque são considerados fixos por definição de engenharia. Esta abordagem é altamente confiável por diversas razões:

            *   **Base Empírica e Objetiva:** Os novos limites para o ângulo são derivados diretamente do comportamento **real e bem-sucedido** do seu processo, filtrados pela combinação específica de Grupamento e Ferramenta que você está analisando. Não são valores teóricos ou arbitrários, mas sim uma representação estatística do que funciona na prática para aquela condição.
            *   **Robustez Estatística:** O uso de **percentis** torna a definição dos limites de ângulo robusta a outliers e distribuições não-normais. Isso significa que a janela proposta para o ângulo reflete com fidelidade a variação natural do seu processo "OK", sem ser distorcida por eventos extremos.
            *   **Maior Sensibilidade do Controle:** Ao **estreitar a janela para o ângulo** para o que o processo realmente é capaz de produzir com qualidade, a aplicação se torna mais sensível a pequenas variações. Isso permite a **detecção precoce** de tendências ou desvios que antes poderiam passar despercebidos dentro de uma janela mais ampla, otimizando o controle de qualidade.
            *   **Suporte à Melhoria Contínua:** Esta ferramenta fornece um `feedback` claro sobre a performance do processo. Uma janela mais precisa para o ângulo não só identifica falhas de forma mais acurada, mas também impulsiona a otimização dos parâmetros da ferramenta e das condições do processo para operar dentro de tolerâncias mais rigorosas.

            **Recomendação de Ajuste dos Limites:**
            A recomendação é ajustar os limites de **ângulo** na linha de produção para os valores calculados como **"novo intervalo otimizado"**, que são **mais restritivos** e baseados na capacidade real do processo para a combinação selecionada de Grupamento e Ferramenta. Os limites de Torque devem permanecer inalterados. Este ajuste deve ser feito com um plano de validação detalhado, começando possivelmente com um piloto em uma ferramenta ou linha específica, e monitorando de perto os resultados.

            **Índice de Confiabilidade da Aplicação/Metodologia:**
            Embora não exista um "índice de confiança" numérico único para a aplicação em si, a **confiança nesta metodologia é intrínseca à sua base estatística e empírica**. A precisão e a confiabilidade dos novos limites são diretas consequências da análise dos seus dados reais. A efetividade da aplicação e da metodologia se manifestará em:
            *   **Redução de 'NOK' genuínos (para ângulo):** Com uma janela mais apertada no ângulo, um 'NOK' significa que o aperto *realmente* está fora do padrão de qualidade aceitável do seu processo, não sendo apenas uma variação dentro de uma tolerância excessivamente larga.
            *   **Melhora nos Índices de Capacidade (Cp/Cpk):** Após a implementação dos novos limites de **ângulo**, os índices Cp e Cpk do seu processo tenderão a melhorar, indicando uma maior capacidade de atender às **especificações mais rigorosas**.
            *   **Aumento da Qualidade Percebida:** Menos variações permitidas no ângulo resultam em maior consistência e qualidade do produto final, reduzindo a chance de problemas de montagem ou falhas em campo.

            Esta ferramenta é um passo significativo para transformar dados históricos em decisões proativas para a melhoria da qualidade e eficiência na sua fábrica, focando na **restrição da janela do Ângulo** para um controle mais robusto.
            """)

with tab2:
    st.header("Definição Manual da Janela Otimizada")
    with st.expander("📝 Sobre a Definição Manual"):
        st.markdown("""
        Esta seção permite que você defina os limites da nova janela de aperto manualmente, oferecendo controle total sobre o nível de restrição para o **Ângulo**. Os limites de **Torque** são fixos e definidos pela engenharia, não podendo ser alterados aqui. Isso é útil se você tiver um conhecimento prévio dos limites desejados para o ângulo ou quiser explorar cenários específicos que não são diretamente otimizados pelos percentis.
        """)

    if st.session_state.df_filtered is None or st.session_state.df_filtered.empty:
        st.warning(
            "Por favor, carregue um arquivo CSV e aplique os filtros na aba 'Otimização por Percentis' para usar esta funcionalidade.")
    else:
        # CHAVE para Cp/Cpk: df_ok_for_optimization a partir do df_filtered
        df_ok_for_optimization_manual = st.session_state.df_filtered[st.session_state.df_filtered['Avaliação'] == 'OK']

        if df_ok_for_optimization_manual.empty:
            st.warning("""
            Não há dados 'OK' no subconjunto de dados atualmente filtrado para calcular Cp/Cpk.
            Por favor, ajuste seus filtros (Grupamento, Ferramenta, Avaliação) para incluir dados 'OK' na sua seleção.
            """)
        else:
            st.subheader("Defina os Limites da Nova Janela Manualmente:")
            st.info(f"""
            Os limites de **Torque (Nm)** são fixos e definidos pela engenharia:
            - **Mínimo:** `{st.session_state.current_tq_min:.3f} Nm`
            - **Máximo:** `{st.session_state.current_tq_max:.3f} Nm`
            """)

            # Sugerir valores iniciais para os inputs manuais
            # Usar o valor atual do session_state, se já definido, ou calcular um valor inicial
            # Isso evita que o valor inicial do number_input seja redefinido a cada rerun,
            # mantendo o que o usuário digitou, a menos que seja a primeira vez ou o DF mude.
            if st.session_state.manual_ang_min == 0.0 and st.session_state.manual_ang_max == 0.0:
                 min_ang_sug = st.session_state.df_filtered['ÂNG_rea'].min()
                 max_ang_sug = st.session_state.df_filtered['ÂNG_rea'].max()
            else:
                 min_ang_sug = st.session_state.manual_ang_min
                 max_ang_sug = st.session_state.manual_ang_max


            # Inputs para os novos limites manuais
            col_manual1, col_manual2 = st.columns(2)
            with col_manual1:
                st.markdown("**Defina os Limites do Ângulo (°):**")
                # O valor retornado por st.number_input já é armazenado na session_state pela key
                manual_ang_min = st.number_input("Novo Ângulo Mínimo (°):", value=float(f"{min_ang_sug:.3f}"),
                                                 format="%.3f", key="manual_ang_min")
            with col_manual2:
                # O valor retornado por st.number_input já é armazenado na session_state pela key
                manual_ang_max = st.number_input("Novo Ângulo Máximo (°):", value=float(f"{max_ang_sug:.3f}"),
                                                 format="%.3f", key="manual_ang_max")

            # Fixed torque values for manual window calculation
            manual_tq_min = st.session_state.current_tq_min
            manual_tq_max = st.session_state.current_tq_max

            # Store manual TQ values in session state for the report tab.
            # Essas linhas estão OK, pois manual_tq_min/max NÃO SÃO as chaves de um widget st.number_input
            st.session_state.manual_tq_min = manual_tq_min
            st.session_state.manual_tq_max = manual_tq_max
            # REMOVIDO: st.session_state.manual_ang_min = manual_ang_min (Já gerenciado pela key do widget)
            # REMOVIDO: st.session_state.manual_ang_max = manual_ang_max (Já gerenciado pela key do widget)


            # Validação simples dos inputs manuais
            if manual_ang_min >= manual_ang_max:
                st.error("O limite mínimo do ângulo deve ser menor que o limite máximo. Por favor, ajuste os valores.")
            else:
                st.markdown("#### Comparativo de Limites: Atual vs. Manualmente Definido (Torque fixo, Ângulo manual)")
                data_limites_manual = {
                    'Parâmetro': ['Torque Mínimo (Nm)', 'Torque Máximo (Nm)', 'Ângulo Mínimo (°)', 'Ângulo Máximo (°)'],
                    'Limite Nominal Atual': [st.session_state.current_tq_min, st.session_state.current_tq_max,
                                             st.session_state.current_ang_min, st.session_state.current_ang_max],
                    'Novo Limite Manual': [manual_tq_min, manual_tq_max, manual_ang_min, manual_ang_max]
                    # tq_min/max are current nominal
                }
                df_limites_manual = pd.DataFrame(data_limites_manual)
                st.dataframe(df_limites_manual.set_index('Parâmetro'))

                st.markdown("---")

                # --- CÁLCULO E EXIBIÇÃO DE ST.METRICS PARA DEFINIÇÃO MANUAL ---
                st.subheader("Métricas para a Janela Manualmente Definida")

                col_data_man1, col_data_man2, col_data_man3 = st.columns(3)
                with col_data_man1:
                    st.metric(label="Total de Apertos Avaliados", value=len(st.session_state.df_filtered))
                with col_data_man2:
                    st.metric(label="Torque Real Mínimo (Nm)",
                              value=f"{st.session_state.df_filtered['TQ_rea'].min():.3f}")
                    st.metric(label="Torque Real Máximo (Nm)",
                              value=f"{st.session_state.df_filtered['TQ_rea'].max():.3f}")
                with col_data_man3:
                    st.metric(label="Ângulo Real Mínimo (°)",
                              value=f"{st.session_state.df_filtered['ÂNG_rea'].min():.3f}")
                    st.metric(label="Ângulo Real Máximo (°)",
                              value=f"{st.session_state.df_filtered['ÂNG_rea'].max():.3f}")

                st.markdown("---")  # Separador visual

                area_nominal_tq_manual = st.session_state.current_tq_max - st.session_state.current_tq_min
                area_otimizada_tq_manual = st.session_state.current_tq_max - st.session_state.current_tq_min  # Torque width remains nominal

                area_nominal_ang_manual = st.session_state.current_ang_max - st.session_state.current_ang_min
                area_otimizada_ang_manual = manual_ang_max - manual_ang_min

                percent_tq_manual = 0.0  # Torque reduction is 0%
                # No need for calculation, it's always 0% as per requirements

                percent_ang_manual = 0
                if area_nominal_ang_manual > 0:
                    percent_ang_manual = ((
                                                  area_nominal_ang_manual - area_otimizada_ang_manual) / area_nominal_ang_manual) * 100

                area_nominal_total_manual = area_nominal_tq_manual * area_nominal_ang_manual
                area_otimizada_total_manual = area_otimizada_tq_manual * area_otimizada_ang_manual

                percent_total_manual = 0
                if area_nominal_total_manual > 0:
                    percent_total_manual = ((
                                                    area_nominal_total_manual - area_otimizada_total_manual) / area_nominal_total_manual) * 100

                # Store total percentage in session state
                st.session_state.percent_total_manual = percent_total_manual

                col_met_man1, col_met_man2, col_met_man3 = st.columns(3)
                with col_met_man1:
                    st.metric(label="Redução na Largura de Torque", value=f"{percent_tq_manual:.2f}%",
                              delta=f"Permanece em {area_nominal_tq_manual:.3f} Nm (definido pela engenharia)",
                              delta_color="off")
                with col_met_man2:
                    st.metric(label="Redução na Largura de Ângulo", value=f"{percent_ang_manual:.2f}%",
                              delta=f"De {area_nominal_ang_manual:.3f} para {area_otimizada_ang_manual:.3f}",
                              delta_color="inverse")
                with col_met_man3:
                    st.metric(label="Redução na Área Total da Janela", value=f"{percent_total_manual:.2f}%",
                              delta=f"De {area_nominal_total_manual:.3f} para {area_otimizada_total_manual:.3f}",
                              delta_color="inverse")

                st.markdown("---")  # Separador visual

                # Capacidade do Processo (Cp/Cpk) para limites manuais
                col_cp_man1, col_cp_man2, col_cp_man3, col_cp_man4 = st.columns(4)

                cp_tq_manual, cpk_tq_manual = calculate_cp_cpk(df_ok_for_optimization_manual['TQ_rea'],
                                                               st.session_state.current_tq_max,
                                                               st.session_state.current_tq_min)
                cp_ang_manual, cpk_ang_manual = calculate_cp_cpk(df_ok_for_optimization_manual['ÂNG_rea'],
                                                                 manual_ang_max, manual_ang_min)

                # Store Cp/Cpk values in session state
                st.session_state.cp_tq_manual = cp_tq_manual
                st.session_state.cpk_tq_manual = cpk_tq_manual
                st.session_state.cp_ang_manual = cp_ang_manual
                st.session_state.cpk_ang_manual = cpk_ang_manual

                with col_cp_man1:
                    if cp_tq_manual == float('inf'):
                        st.metric(label="Cp Torque", value="Perfeito")
                    else:
                        st.metric(label="Cp Torque", value=f"{cp_tq_manual:.2f}")
                with col_cp2:
                    if cpk_tq_manual == float('inf'):
                        st.metric(label="Cpk Torque", value="Perfeito")
                    else:
                        st.metric(label="Cpk Torque", value=f"{cpk_tq_manual:.2f}")
                with col_cp3:
                    if cp_ang_manual == float('inf'):
                        st.metric(label="Cp Ângulo", value="Perfeito")
                    else:
                        st.metric(label="Cp Ângulo", value=f"{cp_ang_manual:.2f}")
                with col_cp4:
                    if cpk_ang_manual == float('inf'):
                        st.metric(label="Cpk Ângulo", value="Perfeito")
                    else:
                        st.metric(label="Cpk Ângulo", value=f"{cpk_ang_manual:.2f}")

                # Intelligent Cp/Cpk Analysis Text Box for Manual Limits
                with st.expander("📝 Interpretação e Recomendações de Limites (Cp/Cpk)"):
                    # Pass df_ok_for_optimization_manual to the analysis function
                    st.markdown(generate_cp_cpk_analysis(cp_tq_manual, cpk_tq_manual, cp_ang_manual, cpk_ang_manual,
                                                         df_ok_for_optimization_manual))

                st.markdown("---")  # Separador visual

                st.markdown("#### Comparação Visual: Janela Nominal vs. Janela Definida Manualmente")

                fig_manual = generate_comparison_chart(
                    st.session_state.df_filtered,
                    st.session_state.current_ang_min, st.session_state.current_tq_min,
                    st.session_state.current_ang_max, st.session_state.current_tq_max,
                    manual_ang_min, manual_tq_min, manual_ang_max, manual_tq_max,
                    "Comparação Visual: Janela Nominal vs. Janela Definida Manualmente"
                )

                st.plotly_chart(fig_manual, use_container_width=True, key="comparison_manual_tab2") # Adicionada key

with tab3:
    st.title("RELATÓRIO DE ANÁLISE DE TORQUE E ÂNGULO")

    # 1° Seção
    st.subheader("1. Informações do Processo")
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.text_input("Modelo:", key="report_model")
        st.text_input("Gspas do Processo:", key="report_gspas")
        st.text_input("Workstation:", key="report_workstation")
    with col_info2:
        st.text_input("Nome do Processo:", key="report_process_name")
        st.date_input("Data da Análise:", value=datetime.date.today(), key="report_analysis_date")

    # 2° Seção
    st.subheader("2. Anexo de Imagem GSPAS")
    st.write("Faça upload da imagem GSPAS")
    st.file_uploader("Upload da Imagem GSPAS", type=["png", "jpg", "jpeg"], key="report_gspas_image")

    # 3° Seção
    st.subheader("3. Detalhamento")
    col_det1, col_det2 = st.columns(2)
    with col_det1:
        # Check if df_filtered has data to prevent errors
        sample_size = len(st.session_state.df_filtered) if st.session_state.df_filtered is not None else 0
        st.text_input("Tamanho da amostra:", value=str(sample_size), disabled=True, key="report_sample_size")
        st.number_input("Número do JOB:", key="report_job_number")

    with col_det2:
        # Assuming gp_selecionado and ferramenta_selecionada are available from sidebar
        st.text_input("Número do programa (GP):", value=gp_selecionado, disabled=True, key="report_program_number")
        st.text_input("Número da ferramenta:", value=ferramenta_selecionada, disabled=True, key="report_tool_number")
        st.text_input("Número da Virtual Station:", key="report_virtual_station")

    # 4° Seção - Estratégia atual (Dynamic Steps)
    st.subheader("4. Estratégia Atual")
    for i in range(st.session_state.num_steps):
        st.markdown(f"##### Step {i + 1}")
        col_step1, col_step2 = st.columns(2)
        with col_step1:
            st.number_input(f"Torque aplicado (Nm):", key=f"step_tq_aplicado_{i}", format="%.3f")
            st.number_input(f"Ângulo aplicado (°):", key=f"step_ang_aplicado_{i}", format="%.3f")
        with col_step2:
            st.number_input(f"Faixa de torque aplicada +- (Nm):", key=f"step_tq_faixa_{i}", format="%.3f")
            st.number_input(f"Faixa de ângulo aplicada +- (°):", key=f"step_ang_faixa_{i}", format="%.3f")
        st.number_input(f"RPM aplicado:", key=f"step_rpm_aplicado_{i}")
        st.markdown("---")

    if st.button("Adicionar Step", key="add_step_button"):
        st.session_state.num_steps += 1
        st.rerun()  # Rerun to display new step fields

    # 5° Seção - Resultado da Análise de Ângulo
    st.subheader("5. Resultado da Análise de Ângulo")
    st.markdown("##### Janela Ângulo Atual:")
    st.write(f"Intervalo: `{st.session_state.current_ang_min:.3f}°` à `{st.session_state.current_ang_max:.3f}°`")

    st.markdown("##### Nova Janela de Ângulo Após Análise:")
    # Usar os valores da session_state diretamente, pois são populados via key do widget ou atribuição.
    st.write(f"Intervalo: `{st.session_state.manual_ang_min:.3f}°` à `{st.session_state.manual_ang_max:.3f}°`")
    st.write(f"Porcentagem de Restrição Obtida: `{st.session_state.percent_total_manual:.2f}%`")

    # 6° Seção - Gráficos
    st.subheader("6. Gráficos e Métricas da Análise Manual")

    # Display metrics from manual tab
    st.markdown("##### Métricas dos Dados Reais:")
    col_report_data1, col_report_data2 = st.columns(2)
    with col_report_data1:
        st.metric(label="Torque Real Mínimo (Nm)", value=f"{st.session_state.df_filtered['TQ_rea'].min():.3f}")
        st.metric(label="Torque Real Máximo (Nm)", value=f"{st.session_state.df_filtered['TQ_rea'].max():.3f}")
    with col_report_data2:
        st.metric(label="Ângulo Real Mínimo (°)", value=f"{st.session_state.df_filtered['ÂNG_rea'].min():.3f}")
        st.metric(label="Ângulo Real Máximo (°)", value=f"{st.session_state.df_filtered['ÂNG_rea'].max():.3f}")

    st.markdown("##### Métricas de Capacidade (Cp/Cpk) com Limites Manuais:")
    col_report_cp1, col_report_cp2, col_report_cp3, col_report_cp4 = st.columns(4)
    with col_report_cp1:
        if st.session_state.cp_tq_manual == float('inf'):
            st.metric(label="Cp Torque", value="Perfeito")
        else:
            st.metric(label="Cp Torque", value=f"{st.session_state.cp_tq_manual:.2f}")
    with col_report_cp2:
        if st.session_state.cpk_tq_manual == float('inf'):
            st.metric(label="Cpk Torque", value="Perfeito")
        else:
            st.metric(label="Cpk Torque", value=f"{st.session_state.cpk_tq_manual:.2f}")
    with col_report_cp3:
        if st.session_state.cp_ang_manual == float('inf'):
            st.metric(label="Cp Ângulo", value="Perfeito")
        else:
            st.metric(label="Cp Ângulo", value=f"{st.session_state.cp_ang_manual:.2f}")
    with col_report_cp4:
        if st.session_state.cpk_ang_manual == float('inf'):
            st.metric(label="Cpk Ângulo", value="Perfeito")
        else:
            st.metric(label="Cpk Ângulo", value=f"{st.session_state.cpk_ang_manual:.2f}")

    # Display charts
    if st.session_state.df_filtered is not None and not st.session_state.df_filtered.empty:
        st.markdown("##### Gráfico: Janela Nominal vs. Janela Definida Manualmente")
        fig_manual_report = generate_comparison_chart(
            st.session_state.df_filtered,
            st.session_state.current_ang_min, st.session_state.current_tq_min,
            st.session_state.current_ang_max, st.session_state.current_tq_max,
            st.session_state.manual_ang_min, st.session_state.manual_tq_min,
            st.session_state.manual_ang_max, st.session_state.manual_tq_max,
            "Comparação Visual: Janela Nominal vs. Janela Definida Manualmente (Relatório)"
        )
        st.plotly_chart(fig_manual_report, use_container_width=True, key="comparison_manual_report_tab3") # Adicionada key

        st.markdown("##### Histograma de Torque Real com Curva Normal")
        histograms = generate_histograms(st.session_state.df_filtered)
        st.plotly_chart(histograms['torque_histogram'], use_container_width=True, key="tq_hist_report_tab3") # Adicionada key

        st.markdown("##### Histograma de Ângulo Real com Curva Normal")
        st.plotly_chart(histograms['angle_histogram'], use_container_width=True, key="ang_hist_report_tab3") # Adicionada key
    else:
        st.info("Carregue e filtre os dados para visualizar os gráficos e métricas.")

    # 7° Seção - Método da validação
    st.subheader("7. Método da Validação")
    st.checkbox("Validação do ângulo por falta de pré-rosqueio", key="val_falta_prerosqueio")
    st.checkbox("Validação do ângulo por pré-rosqueio excessivo", key="val_prerosqueio_excessivo")
    st.checkbox("Validação ângulo por duplo aperto", key="val_duplo_aperto")

    for i in range(st.session_state.num_spacers):
        col_spacer_val, col_spacer_btn = st.columns([0.8, 0.2])
        with col_spacer_val:
            st.number_input(f"Validação com espaçador de (mm):", key=f"spacer_val_{i}", format="%.3f")
    if st.button("Adicionar Espaçador", key="add_spacer_button"):
        st.session_state.num_spacers += 1
        st.rerun()

    # 8° Seção - Imagem de validação
    st.subheader("8. Imagem de Validação")
    st.file_uploader("Upload da Imagem de Validação", type=["png", "jpg", "jpeg"], key="report_validation_image")

    # 9° Seção - Resultado (st.radio para seleção exclusiva)
    st.subheader("9. Resultado")
    st.session_state.resultado_aprovado = st.radio(
        "Status da Análise:",
        ("Aprovado", "Reprovado"),
        index=None if st.session_state.resultado_aprovado is None else (0 if st.session_state.resultado_aprovado == "Aprovado" else 1),
        key="resultado_radio_selection"
    )

    # 10° Seção - Observações de engenharia
    st.subheader("10. Observações de Engenharia")
    st.text_area("Observações:", max_chars=3000, height=150, key="report_observations")

    # 11° Seção - Responsabilidade pela análise
    st.subheader("11. Responsabilidade pela Análise")
    for i in range(st.session_state.num_resp_analise):
        st.markdown(f"##### Responsável {i + 1}")
        st.text_input("Nome:", key=f"resp_analise_nome_{i}")
        st.text_input("Cargo/Função:", key=f"resp_analise_cargo_{i}")
        st.write("Assinatura: _________________________")
        st.markdown("---")
    if st.button("Adicionar Responsável pela Análise", key="add_resp_analise_button"):
        st.session_state.num_resp_analise += 1
        st.rerun()

    # 12° Seção - Responsabilidade pela aprovação
    st.subheader("12. Responsabilidade pela Aprovação")
    for i in range(st.session_state.num_resp_aprov):
        st.markdown(f"##### Responsável {i + 1}")
        st.text_input("Nome:", key=f"resp_aprov_nome_{i}")
        st.text_input("Cargo/Função:", key=f"resp_aprov_cargo_{i}")
        st.write("Assinatura: _________________________")
        st.markdown("---")
    if st.button("Adicionar Responsável pela Aprovação", key="add_resp_aprov_button"):
        st.session_state.num_resp_aprov += 1
        st.rerun()
