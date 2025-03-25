import streamlit as st
import pandas as pd
import requests
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pprint
from datetime import datetime, timedelta
import os
import tempfile

# Configuración de API
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Dashboard de KPI",
    page_icon="📊",
    layout="wide"
)

# Inicializar estado de sesión para persistencia de datos
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
if "filtered_data" not in st.session_state:
    st.session_state.filtered_data = False
if "kpi_results" not in st.session_state:
    st.session_state.kpi_results = {}
if "available_kpis" not in st.session_state:
    st.session_state.available_kpis = []
if "available_filters" not in st.session_state:
    st.session_state.available_filters = {}

# Funciones de API
def load_data(source_type, source, mapping_path, service_line):
    """Cargar datos a través de la API"""
    try:
        response = requests.post(
            f"{API_URL}/load-data",
            json={
                "source_type": source_type,
                "source": source,
                "mapping_path": mapping_path,
                "service_line": service_line
            }
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error de API: {str(e)}")
        if hasattr(e, 'response') and e.response:
            st.error(f"Respuesta: {e.response.text}")
        return None

def get_available_filters():
    """Obtener filtros disponibles de la API"""
    try:
        response = requests.get(f"{API_URL}/available-filters")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error de API: {str(e)}")
        return None

def apply_filters(filters):
    """Aplicar filtros a través de la API"""
    try:
        response = requests.post(
            f"{API_URL}/apply-filters",
            json=filters
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error de API: {str(e)}")
        return None

def get_available_kpis():
    """Obtener KPIs disponibles de la API"""
    try:
        response = requests.get(f"{API_URL}/available-kpis")
        response.raise_for_status()
        return response.json()["kpis"]
    except requests.exceptions.RequestException as e:
        st.error(f"Error de API: {str(e)}")
        return []

def calculate_kpi(kpi_name):
    """Calcular un KPI a través de la API"""
    try:
        response = requests.get(f"{API_URL}/calculate-kpi/{kpi_name}")
        response.raise_for_status()
        return response.json()["result"]
    except requests.exceptions.RequestException as e:
        st.error(f"Error de API: {str(e)}")
        return None

def generate_ai_kpi(user_prompt, service_line, mapping_path):
    """Generar un KPI utilizando IA a través de la API"""
    try:
        response = requests.post(
            f"{API_URL}/generate-ai-kpi",
            json={
                "user_prompt": user_prompt,
                "service_line": service_line,
                "mapping_path": mapping_path
            }
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error de API: {str(e)}")
        return None

def get_raw_data(limit=100):
    """Obtener muestra de datos crudos de la API"""
    try:
        response = requests.get(f"{API_URL}/raw-data?limit={limit}")
        response.raise_for_status()
        return response.json()["data"]
    except requests.exceptions.RequestException as e:
        st.error(f"Error de API: {str(e)}")
        return None

def upload_file_to_api(file):
    """Subir un archivo a la API"""
    try:
        files = {"file": (file.name, file.getvalue())}
        response = requests.post(f"{API_URL}/upload-file", files=files)
        response.raise_for_status()
        return response.json()["path"]
    except requests.exceptions.RequestException as e:
        st.error(f"Error de API: {str(e)}")
        return None

def get_unique_values(column):
    """Obtener valores únicos para una columna específica"""
    try:
        response = requests.get(f"{API_URL}/unique-values/{column}")
        response.raise_for_status()
        return response.json()["values"]
    except requests.exceptions.RequestException as e:
        st.error(f"Error al obtener valores únicos: {str(e)}")
        return []

# Funciones de visualización
def visualize_kpi(kpi_name, kpi_result):
    """Visualizar resultados de KPI con gráficos apropiados"""
    if kpi_result is None:
        st.warning(f"No hay datos para visualizar para {kpi_name}")
        return
    
    # Manejo especial de KPIs específicos
    if kpi_name == "ResumenEstadístico":
        df_stats = pd.DataFrame.from_dict(kpi_result)
        st.dataframe(df_stats)
        return
    
    if kpi_name == "MatrizCorrelación":
        # Convertir diccionario a DataFrame para correlación
        corr_df = pd.DataFrame.from_dict(kpi_result)
        
        # Crear mapa de calor con plotly
        fig = px.imshow(
            corr_df,
            text_auto=True,
            color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1,
            title="Matriz de Correlación"
        )
        st.plotly_chart(fig, use_container_width=True)
        return
    
    if kpi_name == "DistribuciónNumérica":
        # Crear subplots para cada variable numérica
        if not kpi_result:
            st.warning("No hay datos numéricos para visualizar")
            return
        
        # Crear una fila de gráficos para cada variable
        for i, (col, data) in enumerate(kpi_result.items()):
            fig = go.Figure()
            
            # Añadir histograma
            fig.add_trace(go.Bar(
                x=[(data['limites'][i] + data['limites'][i+1])/2 for i in range(len(data['limites'])-1)],
                y=data['frecuencias'],
                name="Frecuencia"
            ))
            
            # Añadir líneas para estadísticas
            fig.add_vline(x=data['promedio'], line_dash="dash", line_color="red", annotation_text="Media")
            fig.add_vline(x=data['mediana'], line_dash="dash", line_color="green", annotation_text="Mediana")
            
            fig.update_layout(
                title=f"Distribución de {col}",
                xaxis_title=col,
                yaxis_title="Frecuencia"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        return
    
    if kpi_name == "TendenciaTemporal":
        if not kpi_result:
            st.warning("No se encontraron columnas de fecha para análisis temporal")
            return
        
        for date_col, time_data in kpi_result.items():
            # Convertir a DataFrame para graficar
            df_time = pd.DataFrame({
                'Fecha': list(time_data.keys()),
                'Cantidad': list(time_data.values())
            })
            
            fig = px.line(
                df_time, 
                x='Fecha', 
                y='Cantidad',
                title=f"Tendencia temporal por {date_col}",
                markers=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        return
    
    if kpi_name == "ConteoPorCategoría":
        if not kpi_result:
            st.warning("No hay datos categóricos para visualizar")
            return
        
        # Para cada categoría crear un gráfico
        for col, counts in kpi_result.items():
            df_cat = pd.DataFrame({
                'Categoría': list(counts.keys()),
                'Conteo': list(counts.values())
            }).sort_values('Conteo', ascending=False)
            
            fig = px.bar(
                df_cat,
                x='Categoría',
                y='Conteo',
                title=f"Distribución de {col}",
                text_auto=True
            )
            
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        return
    
    # Resto del código de visualización existente para tipos genéricos
    st.write("Tipo de resultado:", type(kpi_result).__name__)
    
    # Para tipos de datos simples (numéricos, etc.)
    if isinstance(kpi_result, (int, float)):
        # Mostrar como indicador grande
        fig = go.Figure(go.Indicator(
            mode="number",
            value=kpi_result,
            title={"text": kpi_name},
            number={"font": {"size": 50}}
        ))
        st.plotly_chart(fig, use_container_width=True)
        return
    
    # Para diccionarios (datos agrupados)
    if isinstance(kpi_result, dict):
        # Check para diccionarios anidados
        if any(isinstance(v, dict) for v in kpi_result.values()):
            # Mostrar diccionario anidado como gráfico de árbol
            df_list = []
            for outer_key, inner_dict in kpi_result.items():
                if isinstance(inner_dict, dict):
                    for inner_key, value in inner_dict.items():
                        # Solo incluir valores numéricos
                        if isinstance(value, (int, float)) and not pd.isna(value):
                            df_list.append({
                                "Categoría": str(outer_key),
                                "Subcategoría": str(inner_key),
                                "Valor": float(value)
                            })
            
            if df_list:
                df = pd.DataFrame(df_list)
                try:
                    fig = px.treemap(
                        df, 
                        path=['Categoría', 'Subcategoría'], 
                        values='Valor',
                        title=kpi_name
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error al crear gráfico de árbol: {str(e)}")
                    # Intentar con otro tipo de gráfico
                    try:
                        fig = px.sunburst(
                            df, 
                            path=['Categoría', 'Subcategoría'], 
                            values='Valor',
                            title=kpi_name
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except:
                        # Mostrar tabla como último recurso
                        st.write("Mostrando como tabla:")
                        st.dataframe(df)
            else:
                st.write("No hay datos válidos para visualizar")
                st.write(kpi_result)
        else:
            # Diccionario simple
            # Filtrar solo valores numéricos
            filtered_dict = {k: v for k, v in kpi_result.items() if isinstance(v, (int, float)) and not pd.isna(v)}
            
            if filtered_dict:
                df = pd.DataFrame({"Categoría": list(filtered_dict.keys()), 
                                  "Valor": list(filtered_dict.values())})
                
                # Ordenar para mejor visualización
                df = df.sort_values("Valor", ascending=False)
                
                try:
                    fig = px.bar(
                        df, 
                        x="Categoría", 
                        y="Valor",
                        title=kpi_name,
                        labels={"Valor": "Valor", "Categoría": ""},
                        text_auto=True
                    )
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error al crear gráfico de barras: {str(e)}")
                    # Mostrar tabla como último recurso
                    st.write("Mostrando como tabla:")
                    st.dataframe(df)
            else:
                st.write("No hay valores numéricos para visualizar")
                st.write(kpi_result)
    
    # Para listas, matrices o DataFrames
    elif isinstance(kpi_result, (list, np.ndarray, pd.Series, pd.DataFrame)):
        # Convertir a DataFrame si no lo es
        if not isinstance(kpi_result, pd.DataFrame):
            try:
                df = pd.DataFrame(kpi_result)
            except:
                st.write("No se puede visualizar como gráfico. Mostrando datos crudos:")
                st.write(kpi_result)
                return
        else:
            df = kpi_result
        
        # Mostrar como tabla interactiva
        st.dataframe(df)
        
        # Si tiene 2 columnas, intentar gráfico de dispersión
        if len(df.columns) == 2 and df.select_dtypes(include=['number']).shape[1] == 2:
            numeric_cols = df.select_dtypes(include=['number']).columns
            fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], title=kpi_name)
            st.plotly_chart(fig, use_container_width=True)
    else:
        # Para cualquier otro tipo de datos
        st.write("Tipo de dato no soportado para visualización directa. Mostrando valor:")
        st.write(kpi_result)
        
    # En todos los casos, ofrecer la descarga de los datos
    if isinstance(kpi_result, dict) or isinstance(kpi_result, (list, np.ndarray, pd.Series, pd.DataFrame)):
        try:
            # Convertir a JSON para descarga
            if isinstance(kpi_result, (pd.DataFrame, pd.Series)):
                json_str = kpi_result.to_json(orient="records")
            else:
                json_str = json.dumps(kpi_result)
            
            st.download_button(
                label="Descargar datos",
                data=json_str,
                file_name=f"{kpi_name.replace(' ', '_')}.json",
                mime="application/json"
            )
        except Exception as e:
            st.warning(f"No se pueden descargar los datos: {str(e)}")

# Añadir función para consultas de chat
def chat_query(query_text):
    """Enviar consulta de chat a la API"""
    try:
        response = requests.post(
            f"{API_URL}/chat-query",
            json={"query": query_text}
        )
        response.raise_for_status()
        return response.json()["response"]
    except requests.exceptions.RequestException as e:
        st.error(f"Error al enviar consulta: {str(e)}")
        return None

# Diseño de la UI
st.title("📊 Panel de Análisis de KPI")

# Barra lateral para carga de datos y filtrado
with st.sidebar:
    st.header("Fuente de Datos")
    
    source_type = st.selectbox(
        "Tipo de Fuente",
        ["excel", "csv", "json", "db", "api"],
        index=0
    )
    
    # Opción de subida de archivo
    uploaded_file = None
    use_upload = st.checkbox("Subir archivo en lugar de usar ruta", value=True)
    
    if use_upload:
        if source_type == "excel":
            uploaded_file = st.file_uploader("Subir archivo Excel", type=['xlsx', 'xls'])
            if uploaded_file:
                source = uploaded_file.name
                st.success(f"Archivo subido: {source}")
            else:
                st.info("Por favor suba un archivo Excel")
        elif source_type == "csv":
            uploaded_file = st.file_uploader("Subir archivo CSV", type=['csv'])
            if uploaded_file:
                source = uploaded_file.name
                st.success(f"Archivo subido: {source}")
            else:
                st.info("Por favor suba un archivo CSV")
        elif source_type == "json":
            uploaded_file = st.file_uploader("Subir archivo JSON", type=['json'])
            if uploaded_file:
                source = uploaded_file.name
                st.success(f"Archivo subido: {source}")
            else:
                st.info("Por favor suba un archivo JSON")
    else:
        source = st.text_input("Ruta/URL de Fuente", "Estimulaciones_edit.xlsx")
    
    mapping_path = st.text_input(
        "Ruta de Mapeo",
        "mappings/well_stimulation_mapping.json"
    )
    
    # Permitir subir archivo de mapeo también
    upload_mapping = st.checkbox("Subir archivo de mapeo", value=False)
    uploaded_mapping = None
    
    if upload_mapping:
        uploaded_mapping = st.file_uploader("Subir archivo JSON de mapeo", type=['json'])
        if uploaded_mapping:
            mapping_path = uploaded_mapping.name
            st.success(f"Mapeo subido: {mapping_path}")
    
    service_line = st.text_input("Línea de Servicio", "Stimulation")
    
    if st.button("Cargar Datos"):
        with st.spinner("Cargando datos..."):
            # Manejar subidas de archivos si están presentes
            if uploaded_file:
                # Subir el archivo a la API
                uploaded_path = upload_file_to_api(uploaded_file)
                if uploaded_path:
                    source = uploaded_path
                    st.success(f"Archivo subido a: {source}")
                else:
                    st.error("Error al subir el archivo a la API")
                    source = None
            
            # Manejar subida de archivo de mapeo
            if uploaded_mapping:
                uploaded_mapping_path = upload_file_to_api(uploaded_mapping)
                if uploaded_mapping_path:
                    mapping_path = uploaded_mapping_path
                    st.success(f"Archivo de mapeo subido a: {mapping_path}")
                else:
                    st.error("Error al subir el archivo de mapeo a la API")
            
            # Llamar a la API para cargar datos
            if source is not None:
                result = load_data(source_type, source, mapping_path, service_line)
                
                if result:
                    st.session_state.data_loaded = True
                    st.success(f"Cargadas {result['rows']} filas con {result['columns']} columnas")
                    
                    # Obtener filtros y KPIs disponibles
                    st.session_state.available_filters = get_available_filters()
                    st.session_state.available_kpis = get_available_kpis()
                    
                    # Reiniciar otros valores de estado de sesión
                    st.session_state.filtered_data = False
                    st.session_state.kpi_results = {}

    # Mostrar filtros solo si los datos están cargados
    if st.session_state.data_loaded and st.session_state.available_filters:
        st.header("Filtros")
        
        filters = {}
        
        # Filtros categóricos
        if "categorical" in st.session_state.available_filters and st.session_state.available_filters["categorical"]:
            st.subheader("Filtros Categóricos")
            categorical_filters = {}
            
            for col in st.session_state.available_filters["categorical"]:
                try:
                    unique_values = ["Todos"] + get_unique_values(col)
                    selected = st.selectbox(f"{col}", options=unique_values)
                    
                    if selected != "Todos":
                        categorical_filters[col] = selected
                except Exception as e:
                    st.error(f"Error al obtener valores únicos para {col}: {str(e)}")
            
            if categorical_filters:
                filters["categorical"] = categorical_filters
        
        # Filtros numéricos
        if "numerical" in st.session_state.available_filters and st.session_state.available_filters["numerical"]:
            st.subheader("Filtros Numéricos")
            numerical_filters = {}
            
            for category, cols in st.session_state.available_filters["numerical"].items():
                st.markdown(f"**{category}**")
                
                for col in cols:
                    # Por simplicidad, permitir un rango min/max
                    min_val = st.number_input(f"Mín {col}", value=0.0)
                    max_val = st.number_input(f"Máx {col}", value=1000000.0)
                    
                    if min_val != 0.0 or max_val != 1000000.0:
                        numerical_filters[col] = [min_val, max_val]
            
            if numerical_filters:
                filters["numerical"] = numerical_filters
        
        # Filtros de fecha
        if "date" in st.session_state.available_filters and st.session_state.available_filters["date"]:
            st.subheader("Filtros de Fecha")
            date_filters = {}
            
            for col in st.session_state.available_filters["date"]:
                # Rango de fecha predeterminado (últimos 6 meses)
                today = datetime.now()
                six_months_ago = today - timedelta(days=180)
                
                start_date = st.date_input(f"Inicio {col}", six_months_ago)
                end_date = st.date_input(f"Fin {col}", today)
                
                if start_date != six_months_ago or end_date != today:
                    date_filters[col] = [
                        start_date.strftime("%Y-%m-%d"),
                        end_date.strftime("%Y-%m-%d")
                    ]
            
            if date_filters:
                filters["date"] = date_filters
        
        if filters and st.button("Aplicar Filtros"):
            with st.spinner("Aplicando filtros..."):
                result = apply_filters(filters)
                
                if result:
                    st.session_state.filtered_data = True
                    st.success(f"Filtrado de {result['rows_before']} a {result['rows_after']} filas")
                    
                    # Reiniciar resultados de KPI ya que los datos han cambiado
                    st.session_state.kpi_results = {}

# Área de contenido principal
if st.session_state.data_loaded:
    # Selección de KPI
    st.header("Análisis de KPI")
    
    # Añadir nueva tab para chat
    tab1, tab2, tab3 = st.tabs(["KPIs Predefinidos", "KPIs Personalizados (IA)", "Consultas de Chat (IA)"])
    
    with tab1:
        if st.session_state.available_kpis:
            selected_kpis = st.multiselect(
                "Seleccionar KPIs para Calcular",
                options=st.session_state.available_kpis
            )
            
            if selected_kpis and st.button("Calcular KPIs Seleccionados"):
                with st.spinner("Calculando KPIs..."):
                    for kpi_name in selected_kpis:
                        st.session_state.kpi_results[kpi_name] = calculate_kpi(kpi_name)
        else:
            st.info("No hay KPIs predefinidos disponibles para esta línea de servicio")
    
    with tab2:
        st.subheader("Generar KPI Personalizado con IA")
        
        user_prompt = st.text_area(
            "Describa el KPI que desea crear",
            "Calcular el costo total en USD utilizado por arrendamiento"
        )
        
        if st.button("Generar KPI"):
            with st.spinner("Generando KPI con IA..."):
                ai_kpi = generate_ai_kpi(user_prompt, service_line, mapping_path)
                
                if ai_kpi:
                    st.success("¡KPI Generado Exitosamente!")
                    
                    # Mostrar detalles del KPI
                    st.subheader("Detalles del KPI")
                    st.write(f"**Nombre:** {ai_kpi['name']}")
                    st.write(f"**Descripción:** {ai_kpi['description']}")
                    
                    with st.expander("Ver Fórmula del KPI"):
                        st.code(ai_kpi['formula'])
                    
                    # Mostrar resultado de prueba si está disponible
                    if "test_result" in ai_kpi:
                        st.subheader("Resultado de Prueba del KPI")
                        try:
                            visualize_kpi(ai_kpi['name'], ai_kpi['test_result'])
                        except Exception as e:
                            st.error(f"Error al visualizar KPI: {str(e)}")
                            st.write("Resultado sin procesar:")
                            st.write(ai_kpi['test_result'])
    
    with tab3:
        st.subheader("Consulta los datos usando lenguaje natural")
        
        # Ejemplos de consultas
        st.markdown("""
        **Ejemplos de consultas que puedes hacer:**
        - ¿Cuántos registros hay en total y cómo están distribuidos por categoría?
        - ¿Cuál es el promedio y la desviación estándar de los valores numéricos?
        - ¿Existe alguna correlación importante entre las variables numéricas?
        - Muestra un resumen de las operaciones por locación
        - ¿Cuál es la tendencia de los datos a lo largo del tiempo?
        - ¿Qué productos químicos se utilizan más frecuentemente?
        - ¿Cómo se comparan los valores entre diferentes pozos?
        """)
        
        # Historial de chat (mantener en estado de sesión)
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        # Mostrar historial de chat
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.chat_message("user").write(message["content"])
            else:
                st.chat_message("assistant").write(message["content"])
        
        # Input para nueva consulta
        user_input = st.chat_input("Escribe tu consulta aquí...")
        
        if user_input:
            # Mostrar mensaje del usuario
            st.chat_message("user").write(user_input)
            
            # Añadir a historial
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            # Obtener respuesta
            with st.spinner("Analizando datos..."):
                response = chat_query(user_input)
            
            if response:
                # Mostrar respuesta
                st.chat_message("assistant").write(response)
                
                # Añadir a historial
                st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # Botón para limpiar historial
        if st.button("Limpiar historial de chat"):
            st.session_state.chat_history = []
            st.rerun()

    # Mostrar Resultados de KPI
    if st.session_state.kpi_results:
        st.header("Resultados de KPI")
        
        # Crear un diseño de 2 columnas para visualizaciones de KPI
        cols = st.columns(2)
        
        for i, (kpi_name, kpi_result) in enumerate(st.session_state.kpi_results.items()):
            with cols[i % 2]:
                st.subheader(kpi_name)
                try:
                    visualize_kpi(kpi_name, kpi_result)
                except Exception as e:
                    st.error(f"Error al visualizar KPI: {str(e)}")
                    st.write("Resultado sin procesar:")
                    pprint.pprint(kpi_result)
    
    # Vista previa de datos sin procesar
    with st.expander("Vista Previa de Datos Sin Procesar"):
        if st.button("Cargar Muestra de Datos Sin Procesar"):
            raw_data = get_raw_data(limit=100)
            if raw_data:
                st.dataframe(pd.DataFrame(raw_data))

else:
    # Mostrar instrucciones si los datos no están cargados
    st.info("👈 Comience cargando datos usando la barra lateral")
    
    st.markdown("""
    ## Cómo Empezar
    
    1. Seleccione su tipo de fuente de datos (Excel, CSV, etc.)
    2. Suba su archivo o ingrese la ruta del archivo
    3. Proporcione la ruta del archivo de mapeo o súbalo
    4. Especifique la línea de servicio
    5. Haga clic en "Cargar Datos"
    
    Una vez que los datos estén cargados, puede:
    - Aplicar filtros para delimitar su análisis
    - Calcular KPIs predefinidos
    - Generar KPIs personalizados usando IA
    - Visualizar resultados con gráficos interactivos
    """) 