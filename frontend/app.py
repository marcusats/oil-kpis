import streamlit as st
import pandas as pd
import pprint
from datetime import datetime, timedelta
import os
import tempfile
import json

# Import modules
from services.api_client import (
    load_data, 
    get_available_filters, 
    apply_filters,
    get_available_kpis,
    calculate_kpi,
    generate_ai_kpi,
    get_raw_data,
    upload_file_to_api,
    get_unique_values,
    chat_query,
    interpret_kpi
)
from visualizations.charts import visualize_kpi

# Streamlit page configuration
st.set_page_config(
    page_title="Dashboard de KPI",
    page_icon="📊",
    layout="wide"
)

# Initialize session state for data persistence
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

# Main UI
st.title("📊 Panel de Análisis de KPI")

# Sidebar for data loading and filtering
with st.sidebar:
    st.header("Fuente de Datos")
    
    source_type = st.selectbox(
        "Tipo de Fuente",
        ["excel", "csv", "json", "db", "api"],
        index=0
    )
    
    # File upload option
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
    
    # Allow uploading mapping file as well
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
            # Handle file uploads if present
            if uploaded_file:
                # Upload the file to the API
                uploaded_path = upload_file_to_api(uploaded_file)
                if uploaded_path:
                    source = uploaded_path
                    st.success(f"Archivo subido a: {source}")
                else:
                    st.error("Error al subir el archivo a la API")
                    source = None
            
            # Handle mapping file upload
            if uploaded_mapping:
                uploaded_mapping_path = upload_file_to_api(uploaded_mapping)
                if uploaded_mapping_path:
                    mapping_path = uploaded_mapping_path
                    st.success(f"Archivo de mapeo subido a: {mapping_path}")
                else:
                    st.error("Error al subir el archivo de mapeo a la API")
            
            # Call the API to load data
            if source is not None:
                result = load_data(source_type, source, mapping_path, service_line)
                
                if result:
                    st.session_state.data_loaded = True
                    st.success(f"Cargadas {result['rows']} filas con {result['columns']} columnas")
                    
                    # Get available filters and KPIs
                    st.session_state.available_filters = get_available_filters()
                    st.session_state.available_kpis = get_available_kpis()
                    
                    # Reset other session state values
                    st.session_state.filtered_data = False
                    st.session_state.kpi_results = {}

    # Display filters only if data is loaded
    if st.session_state.data_loaded and st.session_state.available_filters:
        st.header("Filtros")
        
        filters = {}
        
        # Categorical filters
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
        
        # Numerical filters
        if "numerical" in st.session_state.available_filters and st.session_state.available_filters["numerical"]:
            st.subheader("Filtros Numéricos")
            numerical_filters = {}
            
            for category, cols in st.session_state.available_filters["numerical"].items():
                st.markdown(f"**{category}**")
                
                for col in cols:
                    # For simplicity, allow a min/max range
                    min_val = st.number_input(f"Mín {col}", value=0.0)
                    max_val = st.number_input(f"Máx {col}", value=1000000.0)
                    
                    if min_val != 0.0 or max_val != 1000000.0:
                        numerical_filters[col] = [min_val, max_val]
            
            if numerical_filters:
                filters["numerical"] = numerical_filters
        
        # Date filters
        if "date" in st.session_state.available_filters and st.session_state.available_filters["date"]:
            st.subheader("Filtros de Fecha")
            date_filters = {}
            
            for col in st.session_state.available_filters["date"]:
                # Default date range (last 6 months)
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
                    
                    # Reset KPI results since data has changed
                    st.session_state.kpi_results = {}

# Main content area
if st.session_state.data_loaded:
    # KPI Analysis
    st.header("Análisis de KPI")
    
    # Add tabs for different types of analysis
    tab1, tab2, tab3 = st.tabs(["KPIs Predefinidos", "KPIs Personalizados (IA)", "Consultas de Chat (IA)"])
    
    with tab1:
        st.subheader("KPIs Predefinidos")
        
        # Load available KPIs if not already loaded
        if not st.session_state.available_kpis or st.button("Recargar KPIs Disponibles"):
            with st.spinner("Cargando KPIs disponibles..."):
                st.session_state.available_kpis = get_available_kpis()
        
        # Display available KPIs
        if st.session_state.available_kpis:
            # Create a mapping of display names to KPI ids
            kpi_options = {}
            for kpi in st.session_state.available_kpis:
                kpi_options[kpi.get("name", "KPI sin nombre")] = kpi.get("id", "")
            
            selected_kpi_names = st.multiselect(
                "Seleccionar KPIs para Calcular",
                options=list(kpi_options.keys())
            )
            
            if selected_kpi_names and st.button("Calcular KPIs Seleccionados"):
                with st.spinner("Calculando KPIs..."):
                    for kpi_name in selected_kpi_names:
                        # Get the actual KPI ID from the mapping
                        kpi_id = kpi_options[kpi_name]
                        # Almacenar directamente el resultado sin envolverlo en un diccionario adicional
                        st.session_state.kpi_results[kpi_id] = calculate_kpi(kpi_id)
        else:
            st.warning("No hay KPIs disponibles. Asegúrese de que los datos estén cargados.")
        
        # Display calculated KPIs
        if st.session_state.kpi_results:
            st.markdown("---")
            st.subheader("Resultados de KPIs Calculados")
            
            # Create tabs for each calculated KPI
            kpi_tabs = st.tabs(list(st.session_state.kpi_results.keys()))
            
            for i, (kpi_id, kpi_result) in enumerate(st.session_state.kpi_results.items()):
                with kpi_tabs[i]:
                    if kpi_result:
                        # Visualizar directamente con el resultado sin envolverlo en un diccionario
                        visualize_kpi(kpi_id, kpi_result)
                        
                        # Add interpretation button
                        if st.button("Interpretar Resultados", key=f"interpret_{kpi_id}"):
                            with st.spinner("Generando interpretación..."):
                                interpretation = interpret_kpi(kpi_id, kpi_result)
                                if interpretation:
                                    st.subheader("Interpretación de IA")
                                    st.markdown(interpretation)
                                else:
                                    st.warning("No se pudo generar interpretación.")
                        
                        # Add delete button
                        if st.button("Eliminar KPI", key=f"delete_{kpi_id}"):
                            del st.session_state.kpi_results[kpi_id]
                            st.rerun()
                    else:
                        st.warning(f"No hay resultados para el KPI {kpi_id}.")
        else:
            st.info("Calcule KPIs para ver resultados.")
    
    with tab2:
        st.subheader("Generar KPI Personalizado con IA")
        
        # Add examples of KPIs
        st.markdown("""
        **Ejemplos de KPIs que puedes crear:**
        - Calcular el costo promedio por pozo y mostrarlo como un gráfico de barras, ordenado de mayor a menor costo
        - Mostrar la eficiencia de cada operación dividiendo resultado por costo, visualizar como gráfico de barras
        - Crear un gráfico de líneas mostrando tendencias de producción por mes
        - Calcular el porcentaje de operaciones exitosas por tipo de pozo
        - Visualizar la relación entre profundidad y costo de estimulación en un gráfico de dispersión
        """)
        
        user_prompt = st.text_area(
            "Describa el KPI que desea crear",
            "Calcular el costo promedio por pozo y mostrarlo como un gráfico de barras, ordenado de mayor a menor costo."
        )
        
        if st.button("Generar KPI"):
            with st.spinner("Generando KPI con IA..."):
                ai_kpi = generate_ai_kpi(user_prompt, service_line, mapping_path)
                
                if ai_kpi:
                    st.success("¡KPI Generado Exitosamente!")
                    
                    # Show KPI details
                    st.subheader("Detalles del KPI")
                    st.write(f"**Nombre:** {ai_kpi['name']}")
                    st.write(f"**Descripción:** {ai_kpi['description']}")
                    
                    with st.expander("Ver Fórmula del KPI"):
                        st.code(ai_kpi['formula'])
                    
                    # Show test result if available
                    if "test_result" in ai_kpi:
                        st.subheader("Resultado del KPI")
                        
                        test_result = ai_kpi['test_result']
                        if test_result.get('status') == 'success':
                            # Create graph based on visualization preferences
                            try:
                                # Get visualization settings
                                viz_settings = ai_kpi.get('visualization', {
                                    'type': 'bar',
                                    'x_axis': 'Index',
                                    'y_axis': 'Value',
                                    'title': f"{ai_kpi['name']} - Resultados"
                                })
                                
                                # Get data from test result
                                data_type = test_result.get('type', 'other')
                                data = test_result.get('data', {})
                                
                                if data is None or (isinstance(data, dict) and len(data) == 0) or (isinstance(data, list) and len(data) == 0):
                                    st.warning("El KPI se ejecutó correctamente pero no produjo datos para visualizar.")
                                    st.write("Esto puede ocurrir si el filtrado eliminó todos los registros o si no hay datos que cumplan con los criterios.")
                                    
                                    # Show the formula again with suggestions
                                    st.subheader("Sugerencias")
                                    st.write("Revise la fórmula del KPI y considere:")
                                    st.write("1. Verificar los nombres de columnas utilizados")
                                    st.write("2. Asegurarse de que existan datos que cumplan con los filtros")
                                    st.write("3. Modificar el KPI para hacer menos restrictivos los criterios")
                                elif data_type == 'scalar':
                                    # For a single value, show a big number
                                    st.metric(label=viz_settings.get('title', 'Resultado'), value=data)
                                
                                elif data_type == 'series':
                                    # For a series, create a graph based on the series
                                    
                                    # Convert to DataFrame for easier plotting
                                    index = test_result.get('index', [])
                                    df_plot = pd.DataFrame({
                                        'x': index,
                                        'y': list(data.values())
                                    })
                                    
                                    # Create the plot based on visualization type
                                    chart_type = viz_settings.get('type', 'bar')
                                    
                                    if chart_type == 'bar':
                                        st.bar_chart(df_plot.set_index('x'))
                                    elif chart_type == 'line':
                                        st.line_chart(df_plot.set_index('x'))
                                    elif chart_type == 'area':
                                        st.area_chart(df_plot.set_index('x'))
                                    else:
                                        # Default to bar chart
                                        st.bar_chart(df_plot.set_index('x'))
                                    
                                elif data_type == 'dataframe':
                                    # For a dataframe, show a table and try to create a chart
                                    st.dataframe(pd.DataFrame(data))
                                    
                                    # Try to create a plot if there's not too many columns
                                    cols = test_result.get('columns', [])
                                    if len(cols) < 10:
                                        try:
                                            df_plot = pd.DataFrame(data)
                                            if chart_type == 'bar':
                                                st.bar_chart(df_plot)
                                            elif chart_type == 'line':
                                                st.line_chart(df_plot)
                                            elif chart_type == 'area':
                                                st.area_chart(df_plot)
                                            else:
                                                # Default to bar chart
                                                st.bar_chart(df_plot)
                                        except Exception as e:
                                            st.warning(f"No se pudo crear gráfico automático: {str(e)}")
                                
                                elif data_type == 'list':
                                    # For a list, create a simple series and plot
                                    df_plot = pd.DataFrame({
                                        'values': data
                                    })
                                    
                                    if chart_type == 'bar':
                                        st.bar_chart(df_plot)
                                    elif chart_type == 'line':
                                        st.line_chart(df_plot)
                                    elif chart_type == 'area':
                                        st.area_chart(df_plot)
                                    else:
                                        # Default to bar chart
                                        st.bar_chart(df_plot)
                                
                                elif data_type == 'dict':
                                    # For a dictionary, plot keys and values
                                    df_plot = pd.DataFrame({
                                        'x': list(data.keys()),
                                        'y': list(data.values())
                                    })
                                    
                                    if chart_type == 'bar':
                                        st.bar_chart(df_plot.set_index('x'))
                                    elif chart_type == 'line':
                                        st.line_chart(df_plot.set_index('x'))
                                    elif chart_type == 'area':
                                        st.area_chart(df_plot.set_index('x'))
                                    else:
                                        # Default to bar chart
                                        st.bar_chart(df_plot.set_index('x'))
                                
                                else:
                                    # Unknown type, just show the raw data
                                    st.write(data)
                                
                                # Add interpretation if available
                                if "interpretation" in ai_kpi:
                                    st.subheader("Interpretación")
                                    st.write(ai_kpi["interpretation"])
                                
                            except Exception as e:
                                st.error(f"Error al visualizar KPI: {str(e)}")
                                st.write("Resultado sin procesar:")
                                st.write(test_result.get('data'))
                        else:
                            # Error in test result
                            st.error(f"Error al ejecutar la fórmula del KPI: {test_result.get('message', 'Error desconocido')}")
                            
                            # Add a debug expander with raw result
                            with st.expander("Detalles del error"):
                                st.write(test_result)
                                
        # Add manual KPI creation option
        st.markdown("---")
        st.subheader("Crear KPI Manual (Sin IA)")
        st.markdown("Si hay problemas con la generación de KPI con IA, puedes crear un KPI manual:")
        
        with st.expander("Crear KPI Manual"):
            manual_name = st.text_input("Nombre del KPI", "Costo Promedio por Pozo")
            manual_desc = st.text_area("Descripción", "Calcula el costo promedio por cada pozo y lo muestra ordenado de mayor a menor.")
            
            # Show available columns to help the user
            try:
                raw_data = get_raw_data(limit=5)
                if raw_data:
                    st.markdown("**Columnas disponibles:**")
                    columns = pd.DataFrame(raw_data).columns.tolist()
                    st.write(", ".join(columns))
            except:
                st.warning("No se pudieron cargar las columnas disponibles.")
            
            # Crear ejemplos de KPIs comunes
            kpi_examples = {
                "Costo Promedio por Pozo": """# Simple, straightforward implementation that should work with any dataset
import pandas as pd
import numpy as np

# Find cost and well columns
cost_cols = [col for col in df.columns if any(term in str(col).lower() for term in ['cost', 'costo'])]
well_cols = [col for col in df.columns if any(term in str(col).lower() for term in ['pozo', 'well'])]

# If we found appropriate columns, use them
if cost_cols and well_cols:
    # Use the first matching columns found
    cost_col = cost_cols[0]
    well_col = well_cols[0]
    
    # Group by well and calculate average cost
    result = df.groupby(well_col)[cost_col].mean().sort_values(ascending=False)
else:
    # Fallback: try to use the first numeric column as cost and first string column as well
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    string_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if numeric_cols and string_cols:
        result = df.groupby(string_cols[0])[numeric_cols[0]].mean().sort_values(ascending=False)
    else:
        # Create a simple series as fallback
        result = pd.Series({'No suitable columns found': 0})
""",
                "Tendencia de Producción Mensual": """# Tendencia de producción mensual robusta que funciona con cualquier dataset
import pandas as pd
import numpy as np
from datetime import datetime

# 1. Buscar columnas relevantes
date_cols = [col for col in df.columns if any(term in str(col).lower() for term in ['fecha', 'date', 'time'])]
prod_cols = [col for col in df.columns if any(term in str(col).lower() for term in ['prod', 'volume', 'output', 'bbl'])]

# Verificar si encontramos las columnas necesarias
if date_cols and prod_cols:
    # Usar las primeras columnas encontradas
    date_col = date_cols[0]
    prod_col = prod_cols[0]
    
    # Crear copia de trabajo y limpiar valores nulos
    work_df = df[[date_col, prod_col]].copy()
    work_df = work_df.dropna()
    
    # Convertir fecha a datetime si no lo es
    if not pd.api.types.is_datetime64_dtype(work_df[date_col]):
        try:
            work_df[date_col] = pd.to_datetime(work_df[date_col], errors='coerce')
            work_df = work_df.dropna()
        except:
            # Si la conversión falla, crear una columna de fecha simulada
            work_df['fecha_simulada'] = pd.date_range(start='2023-01-01', periods=len(work_df), freq='D')
            date_col = 'fecha_simulada'
    
    # Extraer mes y año
    work_df['mes_año'] = work_df[date_col].dt.strftime('%Y-%m')
    
    # Agrupar por mes y calcular producción promedio
    result = work_df.groupby('mes_año')[prod_col].mean()
    
    # Ordenar por fecha
    try:
        result = result.reindex(sorted(result.index))
    except:
        pass

else:
    # Plan B: Usar columnas numéricas y crear una tendencia temporal simulada
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if numeric_cols:
        # Crear un índice de fecha simulado
        fake_dates = pd.date_range(start='2023-01-01', periods=12, freq='M')
        fake_dates_str = [d.strftime('%Y-%m') for d in fake_dates]
        
        # Tomar muestras del dataframe para cada mes simulado
        values = []
        chunk_size = max(1, len(df) // 12)
        
        for i in range(min(12, len(df) // chunk_size)):
            chunk = df.iloc[i*chunk_size:(i+1)*chunk_size]
            values.append(chunk[numeric_cols[0]].mean())
        
        # Crear una Serie con las fechas simuladas y los valores promedio
        result = pd.Series(values, index=fake_dates_str[:len(values)])
    else:
        # Última opción: crear datos de ejemplo
        fake_dates = pd.date_range(start='2023-01-01', periods=12, freq='M')
        fake_dates_str = [d.strftime('%Y-%m') for d in fake_dates]
        fake_values = np.random.normal(1000, 100, size=12)
        result = pd.Series(fake_values, index=fake_dates_str)
""",
                "Distribución de Costos por Categoría": """# Distribución de costos por categoría
import pandas as pd
import numpy as np

# Buscar columnas relevantes
cost_cols = [col for col in df.columns if any(term in str(col).lower() for term in ['cost', 'costo', 'price', 'valor'])]
category_cols = [col for col in df.columns if any(term in str(col).lower() for term in ['categ', 'type', 'tipo', 'clase', 'class'])]

# Verificar si encontramos las columnas necesarias
if cost_cols and category_cols:
    # Usar las primeras columnas encontradas
    cost_col = cost_cols[0]
    cat_col = category_cols[0]
    
    # Agrupar por categoría y sumar costos
    result = df.groupby(cat_col)[cost_col].sum().sort_values(ascending=False)
else:
    # Plan B: usar la primera columna numérica y la primera categórica
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if numeric_cols and cat_cols:
        result = df.groupby(cat_cols[0])[numeric_cols[0]].sum().sort_values(ascending=False)
    else:
        # Crear datos de ejemplo
        result = pd.Series({
            'Categoría A': 1200,
            'Categoría B': 850,
            'Categoría C': 650,
            'Categoría D': 450,
            'Categoría E': 300
        })
"""
            }
            
            # Selección de KPI predefinido
            kpi_template = st.selectbox("Seleccionar plantilla de KPI", 
                                        list(kpi_examples.keys()),
                                        index=0)
            
            manual_formula = st.text_area("Fórmula (código Python usando pandas)", 
                                          kpi_examples[kpi_template])
            
            chart_type = st.selectbox("Tipo de Gráfico", ["bar", "line", "area", "scatter", "pie"])
            
            if st.button("Crear KPI Manual"):
                # Create a manual KPI structure (same as would be returned by the AI)
                manual_kpi = {
                    "name": manual_name,
                    "description": manual_desc,
                    "formula": manual_formula,
                    "interpretation": "Interpretación manual no disponible. Añade tu propio análisis.",
                    "service_line": service_line,
                    "visualization": {
                        "type": chart_type,
                        "x_axis": "Índice",
                        "y_axis": "Valor",
                        "title": manual_name
                    }
                }
                
                # Test the formula
                with st.spinner("Ejecutando fórmula..."):
                    try:
                        # Get the raw data through the API
                        raw_data = get_raw_data(limit=100)
                        
                        if raw_data:
                            # Create a DataFrame
                            test_df = pd.DataFrame(raw_data)
                            
                            # Set up execution environment
                            exec_globals = {
                                'df': test_df,
                                'pd': pd,
                                'np': np,
                                'result': None
                            }
                            
                            # Execute the formula
                            exec(manual_formula, exec_globals)
                            
                            # Get the result
                            result = exec_globals.get('result')
                            
                            if result is not None:
                                if isinstance(result, pd.Series):
                                    # Create a visualization
                                    st.success("¡KPI ejecutado correctamente!")
                                    st.subheader("Resultados del KPI")
                                    
                                    # Convert to DataFrame for easier plotting
                                    df_plot = pd.DataFrame({
                                        'x': result.index,
                                        'y': result.values
                                    })
                                    
                                    # Create chart based on selected type
                                    if chart_type == 'bar':
                                        st.bar_chart(df_plot.set_index('x'))
                                    elif chart_type == 'line':
                                        st.line_chart(df_plot.set_index('x'))
                                    elif chart_type == 'area':
                                        st.area_chart(df_plot.set_index('x'))
                                    else:
                                        # Default to bar chart
                                        st.bar_chart(df_plot.set_index('x'))
                                else:
                                    st.warning("La fórmula se ejecutó pero no devolvió una Serie de pandas.")
                                    st.write("Resultado:", result)
                            else:
                                st.error("La fórmula no produjo ningún resultado.")
                        else:
                            st.error("No se pudieron obtener datos para probar la fórmula.")
                    except Exception as e:
                        st.error(f"Error al ejecutar la fórmula: {str(e)}")
    
    with tab3:
        st.subheader("Consulta los datos usando lenguaje natural")
        
        # Query examples
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
        
        # Chat history (keep in session state)
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        # Show chat history
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.chat_message("user").write(message["content"])
            else:
                st.chat_message("assistant").write(message["content"])
        
        # Input for new query
        user_input = st.chat_input("Escribe tu consulta aquí...")
        
        if user_input:
            # Show user message
            st.chat_message("user").write(user_input)
            
            # Add to history
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            # Get response
            with st.spinner("Analizando datos..."):
                response = chat_query(user_input)
            
            if response:
                # Show response
                st.chat_message("assistant").write(response)
                
                # Add to history
                st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # Button to clear history
        if st.button("Limpiar historial de chat"):
            st.session_state.chat_history = []
            st.rerun()

    # Raw data preview
    with st.expander("Vista Previa de Datos Sin Procesar"):
        if st.button("Cargar Muestra de Datos Sin Procesar"):
            raw_data = get_raw_data(limit=100)
            if raw_data:
                st.dataframe(pd.DataFrame(raw_data))

else:
    # Show instructions if data is not loaded
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