import streamlit as st
import numpy as np
import cv2
import plotly.graph_objects as go
from PIL import Image
from matplotlib import colors as mcolors
from scipy.interpolate import interp1d
import pandas as pd


# --- FUNÇÕES AUXILIARES ---

def pixel_to_data_coords(x_pixel, y_pixel, x_range_full, y_range_full, px_left, px_bottom, px_right, px_top):
    """
    Converte coordenadas de pixel de uma imagem para coordenadas de dados (ângulo/torque),
    assumindo que o retângulo (px_left, px_bottom, px_right, px_top) na imagem corresponde
    ao intervalo de dados (x_range_full, y_range_full).

    Args:
        x_pixel (int): Coordenada X do pixel.
        y_pixel (int): Coordenada Y do pixel.
        x_range_full (tuple): (x_min, x_max) do range de dados.
        y_range_full (tuple): (y_min, y_max) do range de dados.
        px_left (int): Coordenada X do pixel do lado esquerdo da área de plotagem.
        px_bottom (int): Coordenada Y do pixel do lado inferior da área de plotagem.
        px_right (int): Coordenada X do pixel do lado direito da área de plotagem.
        px_top (int): Coordenada Y do pixel do lado superior da área de plotagem.

    Returns:
        tuple: (x_data, y_data) - Coordenadas de dados convertidas.
    """
    data_x_min, data_x_max = x_range_full
    data_y_min, data_y_max = y_range_full

    # Calcula a largura e altura em pixels da área de plotagem detectada
    pixel_width_of_plot = px_right - px_left
    pixel_height_of_plot = px_bottom - px_top  # Y pixels aumentam para baixo, então a altura é bottom - top

    if pixel_width_of_plot <= 0 or pixel_height_of_plot <= 0:
        st.warning(
            f"Largura ou altura da área de plotagem inválida: Largura={pixel_width_of_plot}, Altura={pixel_height_of_plot}. Retornando NaN para dados.")
        return np.nan, np.nan  # Evita divisão por zero e indica erro

    # Calcula as escalas (unidades de dados por pixel)
    scale_x = (data_x_max - data_x_min) / pixel_width_of_plot
    scale_y = (data_y_max - data_y_min) / pixel_height_of_plot

    # Converte pixel para dado
    # x_data: Começa no data_x_min e avança com a escala_x em relação ao px_left
    x_data = data_x_min + (x_pixel - px_left) * scale_x
    # y_data: Começa no data_y_min e avança com a escala_y, mas o eixo Y dos pixels é invertido
    # (px_bottom - y_pixel) calcula a distância do pixel atual até o fundo da área de plotagem
    y_data = data_y_min + (px_bottom - y_pixel) * scale_y

    return x_data, y_data


def extract_color_points_hsv(image_np, lower_hsv, upper_hsv):
    """
    Extrai pontos de uma imagem com base em um range de cor HSV.
    Retorna uma array de pontos (y, x) onde a cor foi detectada.
    """
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    points = np.column_stack(np.where(mask > 0))  # Retorna (y, x)
    return points


# --- A FUNÇÃO find_graph_plot_area_corners FOI REMOVIDA AQUI ---


# --- CONFIGURAÇÃO DA PÁGINA STREAMLIT ---
st.set_page_config(layout="wide")
st.title("JLR Torque Integrity Analyser")

# --- BARRA LATERAL (SIDEBAR) ---
with st.sidebar:
    st.header("Data viewer")
    # --- Resumo da aplicação ---
    st.write(
        "Faça o upload e a análise de imagens de gráficos de torque e ângulo. "
        "Serão detectadas as curvas coloridas, convertidas para dados numéricos calibrados "
        ", fornecidas ferramentas para visualização detalhada e análises consolidadas da variabilidade. "
        "Ideal para  insights precisos de dados gráficos dos apertos da linha de montagem."
    )
    # --- FIM NOVO ---
    uploaded_files = st.file_uploader("Upload de imagens de gráfico", type=["png", "jpg", "jpeg"],
                                      accept_multiple_files=True)

    st.subheader("Configuração da Escala Padrão do Gráfico Original")
    st.info(
        "Estas são as configurações padrão para novas imagens. Você pode ajustá-las individualmente na seção 'Visualização Detalhada' para cada imagem, se necessário.")
    x_min_default = st.number_input("Ângulo mínimo padrão (X)", value=-1400.0, format="%.1f", key="x_min_default")
    x_max_default = st.number_input("Ângulo máximo padrão (X)", value=200.0, format="%.1f", key="x_max_default")
    y_min_default = st.number_input("Torque mínimo padrão (Y)", value=0.0, format="%.1f", key="y_min_default")
    y_max_default = st.number_input("Torque máximo padrão (Y)", value=20.0, format="%.1f", key="y_max_default")

# --- DEFINIÇÃO DOS RANGES DE CORES HSV ---
pink_lower = np.array([140, 50, 100])
pink_upper = np.array([170, 255, 255])
blue_lower = np.array([100, 100, 100])
blue_upper = np.array([130, 255, 255])

# Paleta de cores para os gráficos
color_palette = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())

# --- INICIALIZAÇÃO DO SESSION STATE ---
if 'processed_file_results' not in st.session_state:
    st.session_state.processed_file_results = []
if 'selected_file_index' not in st.session_state:
    st.session_state.selected_file_index = None
if 'last_uploaded_file_names' not in st.session_state:
    st.session_state.last_uploaded_file_names = set()

    st.write(
        "Desenvolvido por: Eng Diógenes Oliveira"
    )

# --- FUNÇÃO PARA RE-PROCESSAR UMA ÚNICA IMAGEM (PARA CALIBRAÇÃO) ---
def reprocess_image(index_to_reprocess, x_range_new, y_range_new, plot_area_pixel_corners):
    """Re-processa uma imagem específica com novos ranges de eixo e atualiza o session_state."""

    current_result = st.session_state.processed_file_results[index_to_reprocess]

    image_pil_raw = current_result['original_image_pil_raw']
    image_np = np.array(image_pil_raw)

    # Initialize coords lists (defensive programming)
    pink_coords = []
    blue_coords = []
    new_file_specific_curves_data = []

    # Re-extrair pontos coloridos (a detecção de cor não muda, só a conversão para dados)
    pink_points = extract_color_points_hsv(image_np, pink_lower, pink_upper)
    blue_points = extract_color_points_hsv(image_np, blue_lower, blue_upper)

    # Utiliza os cantos da área de plotagem que foram detectados ou recalibrados manualmente
    px_left, px_bottom, px_right, px_top = plot_area_pixel_corners

    # Re-converter para coordenadas de dados usando os NOVOS RANGES e os cantos da área de plotagem
    pink_coords = [pixel_to_data_coords(x, y, x_range_new, y_range_new, px_left, px_bottom, px_right, px_top)
                   for y, x in pink_points]
    blue_coords = [pixel_to_data_coords(x, y, x_range_new, y_range_new, px_left, px_bottom, px_right, px_top)
                   for y, x in blue_points]

    # Re-criar a figura Plotly para esta imagem
    new_fig = go.Figure()

    if pink_coords:
        sorted_coords = sorted([p for p in pink_coords if not np.isnan(p[0])], key=lambda p: p[0])  # Filtra NaNs
        if sorted_coords:  # Garante que há dados após o filtro
            x_vals, y_vals = zip(*sorted_coords)
            new_fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines+markers',
                                         name=f'Primeiro estágio ({current_result["file_name"]})',
                                         marker=dict(color='deeppink', size=3), line=dict(color='deeppink', width=1.5)))
            new_file_specific_curves_data.append(("Primeiro estágio", current_result["file_name"], x_vals, y_vals))

    if blue_coords:
        sorted_coords = sorted([p for p in blue_coords if not np.isnan(p[0])], key=lambda p: p[0])  # Filtra NaNs
        if sorted_coords:  # Garante que há dados após o filtro
            x_vals, y_vals = zip(*sorted_coords)
            new_fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines+markers',
                                         name=f'Segundo estágio ({current_result["file_name"]})',
                                         marker=dict(color='royalblue', size=3),
                                         line=dict(color='royalblue', width=1.5)))
            new_file_specific_curves_data.append(("Segundo estágio", current_result["file_name"], x_vals, y_vals))

    # Adicionar linhas de referência (no sistema de dados do gráfico)
    new_fig.add_shape(
        type="line", x0=0, x1=0, y0=y_range_new[0], y1=y_range_new[1],
        line=dict(color="red", dash="dash"), xref="x", yref="y"
    )


    new_fig.update_layout(
        title=f"Gráfico Reconstruído - {current_result['file_name']} (Calibrado com [{x_range_new[0]}:{x_range_new[1]}] / [{y_range_new[0]}:{y_range_new[1]}])",
        xaxis_title="Ângulo (°)",
        yaxis_title="Torque (Nm)",
        hovermode="x unified",
        showlegend=False,
        height=500,
        xaxis_range=[x_range_new[0], x_range_new[1]],
        yaxis_range=[y_range_new[0], y_range_new[1]]
    )

    # Prepara a imagem original para exibição com o ponto da origem (canto inferior esquerdo da área de plotagem)
    display_image_np = image_np.copy()
    cv2.circle(display_image_np, (px_left, px_bottom), 7, (0, 0, 255), -1)  # Ponto vermelho no canto inferior esquerdo
    cv2.rectangle(display_image_np, (px_left, px_top), (px_right, px_bottom), (0, 255, 0),
                  2)  # Desenha retângulo da área de plotagem
    display_image_pil = Image.fromarray(display_image_np)

    # Atualizar o item específico no session_state
    st.session_state.processed_file_results[index_to_reprocess].update({
        "reconstructed_plotly_fig": new_fig,
        "file_specific_curves_data": new_file_specific_curves_data,
        "x_range_used": x_range_new,
        "y_range_used": y_range_new,
        "plot_area_pixel_corners": plot_area_pixel_corners,  # Armazena os 4 pontos
        "original_image_pil": display_image_pil
    })
    st.rerun()


# --- PROCESSAMENTO DAS IMAGENS ENVIADAS ---
if uploaded_files:
    current_uploaded_file_names = {f.name for f in uploaded_files}

    if current_uploaded_file_names != st.session_state.last_uploaded_file_names:
        st.session_state.processed_file_results = []
        st.session_state.selected_file_index = None
        st.session_state.last_uploaded_file_names = current_uploaded_file_names

        st.info("Detectei novas imagens. Processando todos os arquivos...")

        for uploaded_file in uploaded_files:
            # Inicializa as variáveis para garantir que sempre existam no escopo
            pink_coords = []
            blue_coords = []
            file_specific_curves_data = [] # Também inicializa esta lista
            image_pil_display = None # Para garantir que exista antes do 'try' em caso de erro na abertura

            try:
                image_pil_raw = Image.open(uploaded_file).convert("RGB")
                image_np_raw = np.array(image_pil_raw)
                img_height, img_width = image_np_raw.shape[:2]

                with st.spinner(f"Configurando área de plotagem para {uploaded_file.name}..."):
                    # Define os cantos da área de plotagem para ser a imagem inteira inicialmente
                    # px_left, px_top = 0, 0 (canto superior esquerdo da imagem)
                    # px_right = img_width - 1 (borda direita da imagem)
                    # px_bottom = img_height - 1 (borda inferior da imagem)
                    px_left, px_bottom, px_right, px_top = 0, img_height - 1, img_width - 1, 0

                    plot_area_pixel_corners = (px_left, px_bottom, px_right, px_top)

                    st.info(f"A área de plotagem inicial para {uploaded_file.name} foi definida como a imagem inteira. "
                            "Por favor, ajuste-a manualmente na seção 'Visualização Detalhada' "
                            "se o gráfico reconstruído não estiver correto.")

                    # Desenha o ponto vermelho e o retângulo da área de plotagem para exibição
                    image_np_display = image_np_raw.copy()
                    cv2.circle(image_np_display, (px_left, px_bottom), 7, (0, 0, 255),
                               -1)  # Ponto vermelho no canto inferior esquerdo
                    cv2.rectangle(image_np_display, (px_left, px_top), (px_right, px_bottom), (0, 255, 0),
                                  2)  # Retângulo verde
                    image_pil_display = Image.fromarray(image_np_display)

                    # Usar os ranges padrão do sidebar para o processamento inicial
                    x_range_initial = (x_min_default, x_max_default)
                    y_range_initial = (y_min_default, y_max_default)

                    # --- EXTRAÇÃO E CONVERSÃO DOS PONTOS DA CURVA ---
                    pink_points = extract_color_points_hsv(image_np_raw, pink_lower, pink_upper)
                    blue_points = extract_color_points_hsv(image_np_raw, blue_lower, blue_upper)

                    # Usa os cantos da área de plotagem definidos para a conversão pixel -> data
                    pink_coords = [
                        pixel_to_data_coords(x, y, x_range_initial, y_range_initial, px_left, px_bottom, px_right,
                                             px_top)
                        for y, x in pink_points]
                    blue_coords = [
                        pixel_to_data_coords(x, y, x_range_initial, y_range_initial, px_left, px_bottom, px_right,
                                             px_top)
                        for y, x in blue_points]

                fig = go.Figure()
                # file_specific_curves_data já inicializada acima

                if pink_coords:
                    sorted_coords = sorted([p for p in pink_coords if not np.isnan(p[0])], key=lambda p: p[0])
                    if sorted_coords:
                        x_vals, y_vals = zip(*sorted_coords)
                        fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines+markers',
                                                 name=f'Primeiro estágio ({uploaded_file.name})',
                                                 marker=dict(color='deeppink', size=3),
                                                 line=dict(color='deeppink', width=1.5)))
                        file_specific_curves_data.append(("Primeiro estágio", uploaded_file.name, x_vals, y_vals))

                if blue_coords:
                    sorted_coords = sorted([p for p in blue_coords if not np.isnan(p[0])], key=lambda p: p[0])
                    if sorted_coords:
                        x_vals, y_vals = zip(*sorted_coords)
                        fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines+markers',
                                                 name=f'Segundo estágio ({uploaded_file.name})',
                                                 marker=dict(color='royalblue', size=3),
                                                 line=dict(color='royalblue', width=1.5)))
                        file_specific_curves_data.append(("Segundo estágio", uploaded_file.name, x_vals, y_vals))

                # Adicionar linhas de referência (no sistema de dados do gráfico)
                fig.add_shape(
                    type="line", x0=0, x1=0, y0=y_range_initial[0], y1=y_range_initial[1],
                    line=dict(color="red", dash="dash"), xref="x", yref="y"
                )


                fig.update_layout(
                    title=f"Gráfico Reconstruído - {uploaded_file.name} (Escala Padrão)",
                    xaxis_title="Ângulo (°)",
                    yaxis_title="Torque (Nm)",
                    hovermode="x unified",
                    showlegend=False,
                    height=500,
                    xaxis_range=[x_range_initial[0], x_range_initial[1]],
                    yaxis_range=[y_range_initial[0], y_range_initial[1]]
                )

                st.session_state.processed_file_results.append({
                    "file_name": uploaded_file.name,
                    "original_image_pil_raw": image_pil_raw,
                    "original_image_pil": image_pil_display,
                    "reconstructed_plotly_fig": fig,
                    "file_specific_curves_data": file_specific_curves_data,
                    "x_range_used": x_range_initial,
                    "y_range_used": y_range_initial,
                    "plot_area_pixel_corners": plot_area_pixel_corners  # Armazena os 4 pontos dos cantos
                })

            except Exception as e:
                # Melhoria: Inclui o tipo de erro para melhor depuração
                st.error(f"Erro ao processar a imagem {uploaded_file.name}: {type(e).__name__}: {e}")
        st.success("Todas as imagens processadas!")

# --- POPULAR all_curves PARA ANÁLISES COMBINADAS ---
all_curves = []
global_color_index = 0
for result in st.session_state.processed_file_results:
    for label, fname, x_vals, y_vals in result['file_specific_curves_data']:
        current_color = color_palette[global_color_index % len(color_palette)]
        all_curves.append((label, fname, x_vals, y_vals, current_color))
        global_color_index += 1

# --- DEFINIÇÃO DAS ABAS ---
tab_visualizacao, tab_analises = st.tabs(["🖼️ Visualização", "📊 Análises"])

with tab_visualizacao:
    # --- SEÇÃO DE RESUMO DOS GRÁFICOS PROCESSADOS (GRADE DE THUMBNAILS) ---
    if st.session_state.processed_file_results:
        st.write("---")
        st.header("Resumo dos Gráficos Processados")
        cols_per_row = 3

        grid_container = st.container()
        with grid_container:
            num_results = len(st.session_state.processed_file_results)

            for i in range(0, num_results, cols_per_row):
                cols = st.columns(cols_per_row)
                for j in range(cols_per_row):
                    current_index = i + j
                    if current_index < num_results:
                        result = st.session_state.processed_file_results[current_index]
                        with cols[j]:
                            st.markdown(f"**{result['file_name']}**")
                            st.image(result['original_image_pil'], width=300)
                            st.plotly_chart(result['reconstructed_plotly_fig'], width=300, height=300,
                                            config={'displayModeBar': False})

                            if st.button("Ver Detalhes", key=f"btn_details_{current_index}"):
                                st.session_state.selected_file_index = current_index
                                st.rerun()

    # --- SEÇÃO DE VISUALIZAÇÃO DETALHADA (CONDICIONAL) ---
    if st.session_state.selected_file_index is not None and st.session_state.selected_file_index < len(
            st.session_state.processed_file_results):
        st.write("---")
        st.header("Visualização Detalhada do Gráfico Selecionado")

        selected_result = st.session_state.processed_file_results[st.session_state.selected_file_index]

        st.subheader(f"Detalhes do Arquivo: {selected_result['file_name']}")

        col_img, col_fig = st.columns([1, 2])
        with col_img:
            st.image(selected_result['original_image_pil'], use_column_width=True,
                     caption=f"Imagem Original: {selected_result['file_name']} (Área de Plotagem Configurada)")
        with col_fig:
            st.plotly_chart(selected_result['reconstructed_plotly_fig'], use_container_width=True, height=600)

        st.write("---")
        st.subheader("Ajustar Escala e Área de Plotagem para esta Imagem")
        st.info(
            "Altere os valores abaixo para ajustar a área de plotagem na imagem. "
            "Os cantos do retângulo verde e o ponto vermelho serão atualizados. "
            "Clique em 'Aplicar Nova Calibração' para ver as mudanças.")

        current_x_min_img = st.number_input("Ângulo mínimo para esta imagem", value=selected_result['x_range_used'][0],
                                            format="%.1f", key="x_min_img_detail")
        current_x_max_img = st.number_input("Ângulo máximo para esta imagem", value=selected_result['x_range_used'][1],
                                            format="%.1f", key="x_max_img_detail")
        current_y_min_img = st.number_input("Torque mínimo para esta imagem", value=selected_result['y_range_used'][0],
                                            format="%.1f", key="y_min_img_detail")
        current_y_max_img = st.number_input("Torque máximo para esta imagem", value=selected_result['y_range_used'][1],
                                            format="%.1f", key="y_max_img_detail")

        # Campos para recalibrar os cantos da área de plotagem em pixels
        px_left_detail, px_bottom_detail, px_right_detail, px_top_detail = selected_result['plot_area_pixel_corners']

        st.markdown("**Ajustar Cantos da Área de Plotagem (Pixels):**")
        col_px1, col_px2 = st.columns(2)
        with col_px1:
            new_px_left = st.number_input("Pixel X - Lado Esquerdo", value=int(px_left_detail), format="%d",
                                          key="px_left_detail")
            new_px_top = st.number_input("Pixel Y - Lado Superior", value=int(px_top_detail), format="%d",
                                         key="px_top_detail")
        with col_px2:
            new_px_right = st.number_input("Pixel X - Lado Direito", value=int(px_right_detail), format="%d",
                                           key="px_right_detail")
            new_px_bottom = st.number_input("Pixel Y - Lado Inferior", value=int(px_bottom_detail), format="%d",
                                            key="px_bottom_detail")

        # Botão para aplicar a nova calibração
        if st.button("Aplicar Nova Calibração para esta Imagem"):
            reprocess_image(st.session_state.selected_file_index,
                            (current_x_min_img, current_x_max_img),
                            (current_y_min_img, current_y_max_img),
                            (new_px_left, new_px_bottom, new_px_right, new_px_top))

        # Botão para esconder a visualização detalhada
        if st.button("Esconder Detalhes", key="hide_details"):
            st.session_state.selected_file_index = None
            st.rerun()

with tab_analises:
    if all_curves:
        st.header("Análises Consolidadas de Curvas")

        # 1. Gráfico Combinado de Todas as Curvas
        st.subheader(" Gráfico Combinado de Todas as Curvas Selecionadas")

        _temp_options = set()
        for res_data in st.session_state.processed_file_results:
            file_name = res_data['file_name']
            x_range_used = res_data['x_range_used']
            y_range_used = res_data['y_range_used']
            for curve_info in res_data['file_specific_curves_data']:
                label = curve_info[0]
                option_str = f"{label} ({file_name}) (Ângulo: {x_range_used[0]} a {x_range_used[1]}° | Torque: {y_range_used[0]} a {y_range_used[1]} Nm)"
                _temp_options.add(option_str)
        selected_curves_options = sorted(list(_temp_options))

        selected_curves = st.multiselect(
            "Selecione as curvas para exibir no gráfico combinado:",
            options=selected_curves_options,
            default=selected_curves_options,
            key="combined_multiselect"
        )

        fig_combined = go.Figure()
        for label, fname, x_vals, y_vals, color in all_curves:
            original_result = next(
                (res for res in st.session_state.processed_file_results if res['file_name'] == fname), None)
            if original_result:
                full_option_str = f"{label} ({fname}) (Ângulo: {original_result['x_range_used'][0]} a {original_result['x_range_used'][1]}° | Torque: {original_result['y_range_used'][0]} a {original_result['y_range_used'][1]} Nm)"
                if full_option_str in selected_curves:
                    fig_combined.add_trace(go.Scatter(
                        x=x_vals, y=y_vals, mode='lines+markers',
                        name=f"{label} ({fname})",
                        marker=dict(color=color, size=3), line=dict(color=color, width=1.5)
                    ))

        # Linhas de referência x=0 e y=0 no gráfico combinado
        fig_combined.add_shape(
            type="line", x0=0, x1=0, y0=y_min_default, y1=y_max_default,
            line=dict(color="red", dash="dash"), xref="x", yref="y"
        )

        fig_combined.update_layout(
            title="Gráfico Combinado de Todas as Curvas Selecionadas",
            xaxis_title="Ângulo (°)", yaxis_title="Torque (Nm)",
            showlegend=False,
            hovermode="x unified",
            height=500,
            xaxis_range=[x_min_default, x_max_default],
            yaxis_range=[y_min_default, y_max_default]
        )
        st.plotly_chart(fig_combined, use_container_width=True)

        with st.expander("��️ Diagnóstico: Comportamento Geral das Curvas"):
            st.markdown("""
            **Consistência da Trajetória Torque-Ângulo:** A sobreposição das curvas individuais neste gráfico combinado é um indicador crítico da **repetibilidade do processo de aperto**. 

            *   **Alta sobreposição** e linhas bem agrupadas sugerem um processo estável, com baixa variabilidade nas propriedades da junta, no atrito do conjunto e na consistência da ferramenta. Isso é ideal para a confiabilidade do aperto.
            *   **Dispersão significativa** nas curvas indica instabilidades. Isso pode ser causado por variações na lubrificação, na tolerância dimensional dos componentes, na rigidez da junta, ou na calibração da ferramenta de aperto. Tais inconsistências podem levar a pré-cargas imprevisíveis, afetando diretamente a vida útil da junta e a segurança da aplicação.

            **Relevância para a Restrição de Falhas:** Para mitigar falhas, é fundamental que a transição entre os estágios de aperto e a trajetória geral da curva sejam altamente consistentes. Desvios podem resultar em apertos excessivos (risco de deformação plástica permanente, fadiga precoce do material ou danos ao componente) ou apertos insuficientes (risco de afrouxamento da junta sob vibração ou carga, ou falha por falta de pré-carga). O foco deve ser a minimização da área entre as curvas para garantir uniformidade.
            """)

        # 2. Gráfico Apenas Primeiro Estágio
        st.subheader(" Gráfico - Apenas Primeiro Estágio")
        fig_primeiro = go.Figure()
        for label, fname, x_vals, y_vals, color in all_curves:
            if label == "Primeiro estágio":
                fig_primeiro.add_trace(go.Scatter(
                    x=x_vals, y=y_vals, mode='lines+markers',
                    name=f"{label} ({fname})",
                    marker=dict(color=color, size=3), line=dict(color=color, width=1.5)
                ))
        # Linhas de referência x=0 e y=0
        fig_primeiro.add_shape(
            type="line", x0=0, x1=0, y0=y_min_default, y1=y_max_default,
            line=dict(color="red", dash="dash"), xref="x", yref="y"
        )

        fig_primeiro.update_layout(
            title="Gráfico - Apenas Primeiro Estágio",
            xaxis_title="Ângulo (°)", yaxis_title="Torque (Nm)",
            showlegend=False,
            hovermode="x unified",
            height=500,
            xaxis_range=[x_min_default, x_max_default],
            yaxis_range=[y_min_default, y_max_default]
        )
        st.plotly_chart(fig_primeiro, use_container_width=True)

        # 3. Gráfico Apenas Segundo Estágio
        st.subheader("📊 Gráfico - Apenas Segundo Estágio")
        fig_segundo = go.Figure()
        for label, fname, x_vals, y_vals, color in all_curves:
            if label == "Segundo estágio":
                fig_segundo.add_trace(go.Scatter(
                    x=x_vals, y=y_vals, mode='lines+markers',
                    name=f"{label} ({fname})",
                    marker=dict(color=color, size=3), line=dict(color=color, width=1.5)
                ))
        # Linhas de referência x=0 e y=0
        fig_segundo.add_shape(
            type="line", x0=0, x1=0, y0=y_min_default, y1=y_max_default,
            line=dict(color="red", dash="dash"), xref="x", yref="y"
        )

        fig_segundo.update_layout(
            title="Gráfico - Apenas Segundo Estágio",
            xaxis_title="Ângulo (°)", yaxis_title="Torque (Nm)",
            showlegend=False,
            hovermode="x unified",
            height=500,
            xaxis_range=[x_min_default, x_max_default],
            yaxis_range=[y_min_default, y_max_default]
        )
        st.plotly_chart(fig_segundo, use_container_width=True)

        st.write("---")
        st.subheader("Análise de Tendência e Variação")

        # --- PREPARAÇÃO DOS DADOS PARA ANÁLISE (CURVAS MÉDIAS E ENVELOPE) ---
        common_x_analysis = np.linspace(x_min_default, x_max_default, 500)

        interpolated_data = {
            "Primeiro estágio": [],
            "Segundo estágio": []
        }

        for label, _, x_vals, y_vals, _ in all_curves:
            if label in interpolated_data:
                if len(x_vals) > 1:
                    try:
                        sorted_indices = np.argsort(x_vals)
                        x_vals_sorted = np.array(x_vals)[sorted_indices]
                        y_vals_sorted = np.array(y_vals)[sorted_indices]

                        # Garantir que x_vals_sorted é único para interp1d
                        unique_x_sorted, unique_indices = np.unique(x_vals_sorted, return_index=True)
                        unique_y_sorted = y_vals_sorted[unique_indices]

                        if len(unique_x_sorted) > 1:
                            f_interp = interp1d(unique_x_sorted, unique_y_sorted, kind='linear', bounds_error=False,
                                                fill_value=np.nan)
                            interpolated_y = f_interp(common_x_analysis)
                            interpolated_data[label].append(interpolated_y)
                        else:
                            st.warning(
                                f"Curva '{label}' de um arquivo tem apenas um ponto X único após pré-processamento. Ignorada na interpolação.")
                    except ValueError as ve:
                        st.warning(
                            f"Não foi possível interpolar a curva {label}. Verifique os dados ou o range. Erro: {ve}")
                else:
                    st.warning(
                        f"Curva '{label}' possui menos de 2 pontos para interpolação e foi ignorada na análise de tendência.")

        mean_curves = {}
        std_dev_curves = {}

        for stage, data_list in interpolated_data.items():
            if data_list:
                stacked_data = np.array(data_list)
                mean_curves[stage] = np.nanmean(stacked_data, axis=0)
                std_dev_curves[stage] = np.nanstd(stacked_data, axis=0)
            else:
                mean_curves[stage] = np.full_like(common_x_analysis, np.nan)
                std_dev_curves[stage] = np.full_like(common_x_analysis, np.nan)

        std_factor = st.slider("Fator do Desvio Padrão para o Envelope (e.g., para 1, 2 ou 3-sigma)", min_value=0.5,
                               max_value=3.0, value=2.0, step=0.1)

        for stage_name in ["Primeiro estágio", "Segundo estágio"]:
            if not np.all(np.isnan(mean_curves[stage_name])):
                fig_analysis = go.Figure()

                fig_analysis.add_trace(go.Scatter(
                    x=common_x_analysis, y=mean_curves[stage_name], mode='lines',
                    name=f'Média {stage_name}', line=dict(color='black', width=3)
                ))

                upper_bound = mean_curves[stage_name] + std_factor * std_dev_curves[stage_name]
                lower_bound = mean_curves[stage_name] - std_factor * std_dev_curves[stage_name]

                fig_analysis.add_trace(go.Scatter(
                    x=common_x_analysis, y=upper_bound, mode='lines',
                    line=dict(width=0), showlegend=False, name=f'+/- {std_factor}σ'
                ))
                fig_analysis.add_trace(go.Scatter(
                    x=common_x_analysis, y=lower_bound, mode='lines',
                    fill='tonexty', fillcolor='rgba(0,100,80,0.2)', line=dict(width=0),
                    name=f'{std_factor}σ Envelope', showlegend=False
                ))

                # Linhas de referência x=0 e y=0
                fig_analysis.add_shape(
                    type="line", x0=0, x1=0, y0=y_min_default, y1=y_max_default,
                    line=dict(color="red", dash="dash"), xref="x", yref="y"
                )


                fig_analysis.update_layout(
                    title=f'Curva Média e Envelope de Variação ({stage_name})',
                    xaxis_title="Ângulo (°)", yaxis_title="Torque (Nm)",
                    hovermode="x unified", height=500,
                    xaxis_range=[x_min_default, x_max_default],
                    yaxis_range=[y_min_default, y_max_default]
                )
                st.plotly_chart(fig_analysis, use_container_width=True)

                with st.expander(f"🛠️ Diagnóstico Avançado: Curva Média e Envelope de Variação ({stage_name})"):
                    st.markdown("""
                    ### Interpretação da Curva Média e do Envelope de Variação
                    A **curva média** (linha preta) representa a trajetória típica e esperada do aperto, ou seja, o "caminho" ideal que o torque percorre em função do ângulo para o seu processo. Desvios significativos dessa forma esperada, como picos ou vales inesperados, ou uma inclinação muito diferente da teórica, podem indicar:

                    *   **Problemas com a Junta:** Variações na geometria da rosca, rugosidade da superfície, ou presença de detritos/lubrificantes não controlados que alteram o atrito de forma não linear.
                    *   **Comportamento Anômalo da Ferramenta:** Falhas intermitentes no controle do torque ou ângulo pela ferramenta de aperto, ou problemas na sua rigidez mecânica.
                    *   **Mudanças no Material:** Variações nas propriedades elásticas ou plásticas dos componentes apertados.

                    O **envelope de variação** (área sombreada) é um indicador crítico da **capacidade e estabilidade do processo**. A largura desse envelope, definida pelo fator do desvio padrão (σ), reflete diretamente a dispersão das curvas individuais em torno da média.

                    *   **Envelope Estreito:**
                        *   **Interpretação:** Indica um processo altamente controlado, repetível e com baixa variabilidade. Isso sugere que os fatores que influenciam o aperto (ferramenta, material, lubrificação, operário, ambiente) estão sob controle estatístico.
                        *   **Implicação:** Alta confiança de que cada aperto individual se comportará de maneira muito similar à média, resultando em pré-cargas consistentes e menor risco de falhas por sub ou sobre-aperto. Foco deve ser na otimização e busca de melhorias incrementais.

                    *   **Envelope Largo:**
                        *   **Interpretação:** Sinaliza um processo com alta variabilidade. Existem causas especiais de variação que precisam ser identificadas e eliminadas. Pode ser devido a:
                            *   **Variações no Coeficiente de Atrito:** Inconsistências na lubrificação ou acabamento superficial dos componentes.
                            *   **Rigidez da Junta Variável:** Flutuações nas propriedades dos materiais ou na montagem da junta.
                            *   **Desgaste da Ferramenta:** Ferramentas com desgaste irregular ou que não mantêm a calibração.
                            *   **Fatores Ambientais:** Variações de temperatura ou umidade que afetam a ferramenta ou os componentes.
                        *   **Implicação:** Maior probabilidade de apertos fora das especificações, elevando o risco de falhas em campo. A prioridade é a **investigação da causa raiz** e a implementação de ações corretivas para reduzir a variabilidade.

                    **Em resumo, a análise conjunta da forma da curva média e da largura do envelope permite não apenas diagnosticar a presença de problemas, mas também direcionar a investigação para a natureza da falha (sistemática vs. aleatória) e a otimização contínua do processo de aperto.**
                    """)

                    st.markdown("""
                    ### Entendendo o Fator do Desvio Padrão para o Envelope (Fator Sigma)

                    O "fator do desvio padrão" que você ajusta no slider determina a amplitude do envelope de variação em torno da curva média. Este fator, geralmente representado por múltiplos de **sigma (σ)**, que é o desvio padrão da distribuição dos dados em cada ponto do ângulo, é uma métrica fundamental na estatística e no Controle Estatístico de Processo (CEP).

                    Assumindo que a distribuição dos valores de torque em cada ponto de ângulo ao longo da curva média se aproxima de uma **distribuição normal (gaussiana)**, os múltiplos de sigma têm um significado probabilístico direto:

                    *   **1-sigma (1σ)**: Se você definir o fator como 1.0, o envelope incluirá aproximadamente **68.27%** de todos os dados de torque para cada ângulo. Isso representa a variabilidade "central" do processo.
                    *   **2-sigma (2σ)**: Com um fator de 2.0, o envelope se expande para cobrir cerca de **95.45%** dos dados. Este é um nível de confiança comum para capturar a maioria da variabilidade natural de um processo. Qualquer ponto de dados fora desse envelope de 2-sigma já pode ser considerado um "desvio" significativo.
                    *   **3-sigma (3σ)**: Um fator de 3.0 engloba aproximadamente **99.73%** dos dados. Este é o limite tradicionalmente usado em gráficos de controle de qualidade (como os gráficos de controle de Shewhart) para definir os **Limites de Controle Naturais (LCN)** do processo. Se um ponto de dados cai fora dos limites de 3-sigma, é um forte indicativo da presença de uma **causa especial de variação**, ou seja, algo incomum aconteceu que merece investigação imediata, e não é apenas parte da variabilidade aleatória do processo.

                    **Impacto na Análise:**

                    A escolha do fator sigma para o envelope impacta diretamente sua percepção da estabilidade do processo:

                    *   Um **fator menor** (e.g., 1-sigma) tornará o envelope mais estreito, e mais curvas individuais podem parecer "fora" ou "marginais", mesmo que façam parte da variação normal. Isso pode levar a **falsos alarmes** e investigações desnecessárias.
                    *   Um **fator maior** (e.g., 3-sigma) criará um envelope mais amplo, que capturará quase toda a variabilidade natural do processo. Pontos que caem fora desse envelope são verdadeiramente anomalias e sinalizam problemas sérios no processo, requerendo **ação corretiva**.

                    Para um diagnóstico eficaz na engenharia de processos, o uso de 2-sigma ou 3-sigma é geralmente recomendado, pois eles fornecem um bom equilíbrio entre a sensibilidade para detectar desvios e a robustez contra falsos alarmes, auxiliando Diógenes na identificação de quando o processo está "fora de controle estatístico" e precisa de atenção.
                    """)
            else:
                st.info(f"Dados insuficientes para gerar a análise de curva média para '{stage_name}'.")

        # --- NOVOS GRÁFICOS DE VARIABILIDADE ---
        # Coletar dados para os novos gráficos
        max_torques_per_curve = []
        max_angles_per_curve = []

        for i, result in enumerate(st.session_state.processed_file_results):
            for label, fname, x_vals, y_vals in result['file_specific_curves_data']:
                if x_vals and y_vals:  # Garantir que há pontos
                    max_torques_per_curve.append({
                        "image_idx": i + 1,  # Índice baseado em 1 para o eixo X
                        "max_value": np.max(y_vals),
                        "curve_label": label,
                        "file_name": fname
                    })
                    max_angles_per_curve.append({
                        "image_idx": i + 1,
                        "max_value": np.max(x_vals),
                        "curve_label": label,
                        "file_name": fname
                    })

        # Gráfico de Variabilidade do Torque Máximo
        st.write("---")
        st.subheader(" Variabilidade do Torque Máximo por Imagem")
        fig_max_torque = go.Figure()

        max_torque_first_stage = [d for d in max_torques_per_curve if d["curve_label"] == "Primeiro estágio"]
        if max_torque_first_stage:
            fig_max_torque.add_trace(go.Scatter(
                x=[d["image_idx"] for d in max_torque_first_stage],
                y=[d["max_value"] for d in max_torque_first_stage],
                mode='lines+markers',
                name='Primeiro Estágio',
                marker=dict(color='deeppink', size=6),
                line=dict(width=2)
            ))

        max_torque_second_stage = [d for d in max_torques_per_curve if d["curve_label"] == "Segundo estágio"]
        if max_torque_second_stage:
            fig_max_torque.add_trace(go.Scatter(
                x=[d["image_idx"] for d in max_torque_second_stage],
                y=[d["max_value"] for d in max_torque_second_stage],
                mode='lines+markers',
                name='Segundo Estágio',
                marker=dict(color='royalblue', size=6),
                line=dict(width=2)
            ))

        fig_max_torque.update_layout(
            title="Torque Máximo Registrado por Imagem (Primeiro e Segundo Estágio)",
            xaxis_title="Número da Imagem",
            yaxis_title="Torque Máximo (Nm)",
            hovermode="x unified",
            height=500,
            xaxis_tickmode='array',  # Forçar ticks inteiros
            xaxis_tickvals=list(range(1, len(st.session_state.processed_file_results) + 1)) if len(
                st.session_state.processed_file_results) > 0 else [],
            yaxis_range=[y_min_default, y_max_default]  # Mantém o range do torque consistente
        )
        st.plotly_chart(fig_max_torque, use_container_width=True)

        with st.expander("🛠️ Diagnóstico: Estabilidade do Torque Máximo por Imagem"):
            st.markdown("""
            **Estabilidade do Torque Final:** Este gráfico oferece uma visão temporal da estabilidade do pico de torque atingido por cada aperto, considerando todas as imagens carregadas. É uma ferramenta essencial para monitorar a consistência do processo ao longo de múltiplos ciclos de aperto.

            *   **Flutuações acentuadas** entre imagens consecutivas podem sinalizar problemas como:
                *   **Degradação intermitente da ferramenta de aperto:** Desgaste irregular ou superaquecimento.
                *   **Variações significativas nas propriedades do material:** Lotes diferentes de fixações ou componentes com coeficientes de atrito inconsistentes.
                *   **Inconsistências no posicionamento do componente ou na sequência de aperto:** Introdução de desalinhamentos ou pré-cargas errôneas.
                *   **Acúmulo de contaminantes:** Presença de óleo, sujeira ou detritos nas roscas que alteram o atrito.
            *   A identificação de **tendências** (ascendentes ou descendentes) é vital, pois pode indicar desgaste progressivo da ferramenta, calibração inadequada ou um problema sistêmico que está evoluindo com o tempo.

            **Relevância para a Restrição de Falhas:** A otimização para um torque máximo consistente e dentro das especificações é um pré-requisito para o controle eficaz do ângulo de aperto e, consequentemente, para a prevenção de falhas. Um torque final muito baixo leva a uma pré-carga insuficiente e risco de afrouxamento; um torque muito alto pode causar deformação plástica da rosca, fadiga ou quebra do parafuso/componente. A minimização da variabilidade neste parâmetro contribui diretamente para a durabilidade e segurança do conjunto.
            """)

        # Gráfico de Variabilidade do Ângulo Máximo
        st.write("---")
        st.subheader(" Variabilidade do Ângulo Máximo por Imagem")
        fig_max_angle = go.Figure()

        max_angle_first_stage = [d for d in max_angles_per_curve if d["curve_label"] == "Primeiro estágio"]
        if max_angle_first_stage:
            fig_max_angle.add_trace(go.Scatter(
                x=[d["image_idx"] for d in max_angle_first_stage],
                y=[d["max_value"] for d in max_angle_first_stage],
                mode='lines+markers',
                name='Primeiro Estágio',
                marker=dict(color='deeppink', size=6),
                line=dict(width=2)
            ))

        max_angle_second_stage = [d for d in max_angles_per_curve if d["curve_label"] == "Segundo estágio"]
        if max_angle_second_stage:
            fig_max_angle.add_trace(go.Scatter(
                x=[d["image_idx"] for d in max_angle_second_stage],
                y=[d["max_value"] for d in max_angle_second_stage],
                mode='lines+markers',
                name='Segundo Estágio',
                marker=dict(color='royalblue', size=6),
                line=dict(width=2)
            ))

        fig_max_angle.update_layout(
            title="Ângulo Máximo Registrado por Imagem (Primeiro e Segundo Estágio)",
            xaxis_title="Número da Imagem",
            yaxis_title="Ângulo Máximo (°)",
            hovermode="x unified",
            height=500,
            xaxis_tickmode='array',  # Forçar ticks inteiros
            xaxis_tickvals=list(range(1, len(st.session_state.processed_file_results) + 1)) if len(
                st.session_state.processed_file_results) > 0 else [],
            yaxis_range=[x_min_default, x_max_default]  # Mantém o range do ângulo consistente (usando os valores de X)
        )
        st.plotly_chart(fig_max_angle, use_container_width=True)

        with st.expander("🛠️ Diagnóstico: Controle do Ângulo Final – Chave para Prevenção de Falhas"):
            st.markdown("""
            **Controle do Ângulo Final – Chave para Prevenção de Falhas:** Este gráfico é **diretamente alinhado** com o objetivo principal de restringir falhas através da otimização e potencial redução do ângulo final de aperto. A variabilidade no ângulo máximo (o ponto de parada do processo de aperto angular ou de torque-ângulo) é um indicador crítico da precisão e consistência do controle angular do seu processo.

            *   **Grandes variações** no ângulo máximo entre os apertos podem resultar em:
                *   **Aperto Excessivo:** Se o ângulo é consistentemente muito alto, pode levar a uma deformação plástica indesejada da junta ou do elemento de fixação, resultando em fadiga precoce do material ou até mesmo falha imediata por ruptura. Isso é particularmente problemático em aplicações onde a integridade estrutural e a resiliência a ciclos de carga são cruciais.
                *   **Aperto Insuficiente:** Se o ângulo é muito baixo para o torque ou carga axial desejados, pode levar a pré-cargas inadequadas, resultando em afrouxamento da junta sob vibração ou carga dinâmica. Isso compromete a estabilidade do conjunto e pode levar a falhas de componentes interligados.

            **Para otimizar e reduzir o ângulo final de aperto**, é imperativo que a dispersão neste gráfico seja minimizada. Isso pode exigir uma investigação aprofundada de:
            *   **Ajustes na lógica de controle da ferramenta:** Refinamento dos parâmetros de controle PID ou algoritmos de parada.
            *   **Inspeção de folgas no sistema de fixação:** Eliminação de movimentos indesejados antes do início do aperto efetivo.
            *   **Reavaliação da rigidez da junta:** Variações na compressibilidade da junta podem levar a diferentes ângulos para o mesmo torque.
            *   **Compensação de atrito:** Implementação de estratégias para mitigar a influência do atrito variável.

            A capacidade de atingir consistentemente um ângulo máximo menor, mantendo os requisitos de torque e pré-carga dentro dos limites de engenharia, é um diferencial significativo para a **robustez e longevidade do conjunto**. Este gráfico serve como um KPI (Key Performance Indicator) fundamental para a engenharia de processo, sinalizando quando e onde intervenções são necessárias para alcançar um controle de aperto de alta precisão.
            """)

        st.write("---")
        st.subheader("Distribuição de Métricas Chave para Diagnóstico")

        stats_data_for_plots = []
        for label, fname, x_vals, y_vals, _ in all_curves:
            if len(x_vals) > 0 and len(y_vals) > 0:
                stats_data_for_plots.append({
                    "Curva": f"{label} ({fname})",
                    "Estágio": label,
                    "Torque Máximo": np.max(y_vals),
                    "Ângulo Mínimo": np.min(x_vals),
                    "Ângulo Máximo": np.max(x_vals)
                })
        df_stats_for_plots = pd.DataFrame(stats_data_for_plots)

        if not df_stats_for_plots.empty:
            metrics = ["Torque Máximo", "Ângulo Mínimo", "Ângulo Máximo"]
            for metric in metrics:
                fig_box = go.Figure()

                for stage in df_stats_for_plots['Estágio'].unique():
                    stage_data = df_stats_for_plots[df_stats_for_plots['Estágio'] == stage]
                    fig_box.add_trace(go.Box(
                        y=stage_data[metric],
                        name=stage,
                        boxpoints='all',
                        jitter=0.3,
                        pointpos=-1.8
                    ))

                fig_box.update_layout(
                    title=f'Distribuição de {metric} por Estágio',
                    yaxis_title=metric,
                    height=500,
                    showlegend=False,
                    yaxis_range=[y_min_default if "Torque" in metric else x_min_default if "Ângulo" in metric else None,
                                 y_max_default if "Torque" in metric else x_max_default if "Ângulo" in metric else None]
                )
                st.plotly_chart(fig_box, use_container_width=True)

                with st.expander(f"🛠️ Diagnóstico: Distribuição Estatística de {metric}"):
                    if "Torque" in metric:
                        st.markdown(f"""
                        **Robustez da Distribuição do {metric}:** Os box plots oferecem uma análise estatística visual da dispersão do {metric} para cada estágio de aperto.
                        *   **Caixas compactas** com bigodes curtos indicam um processo altamente repetitivo e controlado, com baixa variabilidade. Isso se traduz em maior confiança na pré-carga final da junta.
                        *   **Caixas alongadas ou assimétricas** sinalizam maior variabilidade ou tendências específicas (e.g., um "rabo" longo para torques mais altos ou mais baixos). Isso exige investigação da causa raiz, como flutuações na ferramenta, material ou condições da junta.
                        *   A presença de **"outliers"** (pontos isolados fora dos bigodes) para o {metric} sinaliza eventos anômalos que requerem investigação imediata – estes são os apertos mais propensos a falhas, seja por sub-aperto crítico ou sobre-aperto destrutivo.

                        **Relevância para a Prevenção de Falhas:** Uma distribuição bem controlada do torque máximo garante que a pré-carga da junta esteja consistentemente dentro dos limites de engenharia, prevenindo tanto o afrouxamento quanto a falha por excesso de estresse.
                        """)
                    elif "Ângulo" in metric:
                        st.markdown(f"""
                        **Robustez da Distribuição do {metric}:** Os box plots são cruciais para entender a variabilidade do {metric}, especialmente o 'Ângulo Máximo', que é vital para o controle da pré-carga e para evitar falhas.
                        *   **Caixas compactas** com bigodes curtos demonstram um processo de controle angular de alta precisão. Uma menor dispersão do ângulo máximo indica que a ferramenta de aperto está atingindo consistentemente o ponto final desejado, o que é fundamental para a durabilidade da junta.
                        *   **Caixas alongadas ou a presença de outliers** para o {metric} indicam instabilidade no controle angular. Isso pode resultar em:
                            *   **Ângulos finais excessivos:** Risco de deformação permanente, fadiga ou mesmo ruptura dos componentes, especialmente em juntas sensíveis à compressão.
                            *   **Ângulos finais insuficientes:** Implicando em pré-cargas abaixo do ideal, o que pode levar ao afrouxamento da junta sob vibração ou carga dinâmica.

                        **Foco na Redução de Ângulo para Restringir Falhas:** Um foco especial deve ser dado à distribuição do 'Ângulo Máximo'. Uma distribuição concentrada e com valores médios/medianos mais baixos (desde que o torque necessário seja atingido e a pré-carga mínima seja garantida) demonstra progresso no objetivo de reduzir o ângulo final. Se a caixa for alongada ou assimétrica, indica que o processo de controle do ângulo precisa de ajustes para maior uniformidade e precisão, o que é fundamental para evitar a fadiga por excesso de aperto ou falhas por falta de aperto. A estabilidade no ângulo de início e fim da curva de aperto reflete diretamente na previsibilidade da pré-carga.
                        """)
        else:
            st.info("Dados insuficientes para gerar gráficos de distribuição de métricas.")

        st.write("---")
        st.subheader("Análises Estatísticas por Curva (Tabela)")
        stats_data = []
        torque_max_global = []
        angle_min_global = []
        angle_max_global = []

        for label, fname, x_vals, y_vals, color in all_curves:
            if len(x_vals) > 0 and len(y_vals) > 0:
                torque_max_global.append(np.max(y_vals))
                angle_min_global.append(np.min(x_vals))
                angle_max_global.append(np.max(x_vals))

                stats_data.append({
                    "Curva": f"{label} ({fname})",
                    "Torque Máximo (Nm)": f"{np.max(y_vals):.2f}",
                    "Ângulo Mínimo (°)": f"{np.min(x_vals):.2f}",
                    "Ângulo Máximo (°)": f"{np.max(x_vals):.2f}",
                    "Torque Médio (Nm)": f"{np.mean(y_vals):.2f}",
                    "Desvio Padrão Torque (Nm)": f"{np.std(y_vals):.2f}"
                })
            else:
                stats_data.append({
                    "Curva": f"{label} ({fname})",
                    "Torque Máximo (Nm)": "N/A",
                    "Ângulo Mínimo (°)": "N/A",
                    "Ângulo Máximo (°)": "N/A",
                    "Torque Médio (Nm)": "N/A",
                    "Desvio Padrão Torque (Nm)": "N/A"
                })

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Torque Máximo Global", f"{np.max(torque_max_global):.2f} Nm" if torque_max_global else "N/A")
        with col2:
            st.metric("Ângulo Mínimo Global", f"{np.min(angle_min_global):.2f} °" if angle_min_global else "N/A")
        with col3:
            st.metric("Ângulo Máximo Global", f"{np.max(angle_max_global):.2f} °" if angle_max_global else "N/A")

        df_stats = pd.DataFrame(stats_data)
        st.dataframe(df_stats, use_container_width=True)

        if not df_stats.empty:
            csv = df_stats.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Baixar Estatísticas (CSV)",
                data=csv,
                file_name='estatisticas_torque_angulo.csv',
                mime='text/csv',
            )
    else:
        st.info(
            "Nenhuma curva processada para análise. Por favor, faça o upload e processamento de imagens na aba 'Visualização'.")
