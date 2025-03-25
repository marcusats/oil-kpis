import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

# Add utility functions for KPI visualization
def safe_parse_json(json_str):
    """Parse JSON safely, handling potential errors"""
    try:
        return json.loads(json_str)
    except (TypeError, json.JSONDecodeError):
        return {}

def ensure_dataframe(data):
    """Ensure data is a pandas DataFrame"""
    if isinstance(data, pd.DataFrame):
        return data
    elif isinstance(data, dict):
        return pd.DataFrame.from_dict(data)
    elif isinstance(data, list):
        return pd.DataFrame(data)
    elif isinstance(data, pd.Series):
        return data.to_frame()
    else:
        return pd.DataFrame({"data": [data]})

def visualize_kpi(kpi_id, kpi_result):
    """
    Visualize the KPI result based on its type.
    """
    # Extract the result data from the API response if needed
    # Si kpi_result ya tiene una estructura con "result", extraerlo
    # Si no, usar directamente kpi_result
    result = kpi_result.get("result", kpi_result)
    
    # Handle different KPI types
    if kpi_id == "ResumenEstadístico":
        visualize_statistics(result)
    elif kpi_id == "MatrizCorrelación":
        visualize_correlation_matrix(result)
    elif kpi_id == "TendenciaTemporal":
        visualize_time_trend(result)
    elif kpi_id == "ConteoPorCategoría":
        visualize_category_counts(result)
    elif kpi_id == "DistribuciónNumérica":
        visualize_numeric_distribution(result)
    elif kpi_id == "Production_Analysis":
        visualize_production_analysis(result)
    elif kpi_id == "PorcentajeQuímicos":
        visualize_chemical_percentage(result)
    elif kpi_id == "UsoQuímicosPorLease":
        visualize_chemical_usage_by_lease(result)
    elif kpi_id == "TotalActividades":
        visualize_total_activities(result)
    elif kpi_id == "IngresosPorIntervención":
        visualize_revenue_by_intervention(result)
    else:
        # Default visualization for unknown KPI types
        st.json(result)

def visualize_statistics(result):
    # Implementation for statistics visualization
    if "statistics" in result:
        for col, stats in result["statistics"].items():
            st.subheader(f"Estadísticas para {col}")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Media", f"{stats.get('mean', 'N/A'):.2f}" if stats.get('mean') is not None else "N/A")
                st.metric("Mediana", f"{stats.get('50%', 'N/A'):.2f}" if stats.get('50%') is not None else "N/A")
                st.metric("Desv. Estándar", f"{stats.get('std', 'N/A'):.2f}" if stats.get('std') is not None else "N/A")
            
            with col2:
                st.metric("Mínimo", f"{stats.get('min', 'N/A'):.2f}" if stats.get('min') is not None else "N/A")
                st.metric("Máximo", f"{stats.get('max', 'N/A'):.2f}" if stats.get('max') is not None else "N/A")
                st.metric("Conteo", stats.get("count", 0))
            
            # Add outlier information
            if "outliers" in stats:
                outliers = stats["outliers"]
                st.info(f"Valores atípicos: {outliers.get('count', 0)} ({outliers.get('percentage', 0):.2f}%)")

def visualize_correlation_matrix(result):
    # Implementation for correlation matrix visualization
    if "formatted" in result:
        correlations = result["formatted"]
        if correlations:
            # Create a DataFrame for visualization
            df_corr = pd.DataFrame(correlations)
            
            # Check if we have the raw correlation matrix for heatmap
            if "matrix" in result and "columns" in result:
                st.subheader("Matriz de Correlación")
                
                # Get the correlation matrix and column names
                matrix_data = result["matrix"]
                columns = list(matrix_data.keys())
                
                # Convert dictionary matrix to 2D array for heatmap
                corr_matrix = []
                for col1 in columns:
                    row = []
                    for col2 in columns:
                        row.append(matrix_data[col1][col2])
                    corr_matrix.append(row)
                
                # Create a heatmap using Plotly
                fig = go.Figure(data=go.Heatmap(
                    z=corr_matrix,
                    x=columns,
                    y=columns,
                    colorscale='RdBu_r',  # Red-Blue scale, reversed (red for negative, blue for positive)
                    zmid=0,  # Center the color scale at 0
                    text=corr_matrix,  # Show values on hover
                    texttemplate="%{text:.2f}",  # Format to 2 decimals and show directly on cells
                    textfont={"size":10},
                    hovertemplate='%{y} vs %{x}<br>Correlación: %{z:.3f}<extra></extra>',
                    colorbar=dict(title='Correlación')
                ))
                
                fig.update_layout(
                    title='Matriz de Correlación',
                    height=700,
                    width=800,
                    xaxis=dict(tickangle=45),
                )
                
                # Show the plot
                st.plotly_chart(fig, use_container_width=True)
            else:
                # If raw matrix is not available, try to reconstruct it from formatted correlations
                st.subheader("Matriz de Correlación")
                
                # Extract unique column names
                all_cols = sorted(list(set(
                    [row["column1"] for row in correlations] + 
                    [row["column2"] for row in correlations]
                )))
                
                # Create an empty correlation matrix
                matrix_size = len(all_cols)
                corr_matrix = np.ones((matrix_size, matrix_size))  # Diagonal will be 1
                
                # Fill the matrix with correlation values
                col_to_idx = {col: i for i, col in enumerate(all_cols)}
                
                for item in correlations:
                    i = col_to_idx[item["column1"]]
                    j = col_to_idx[item["column2"]]
                    corr_matrix[i, j] = item["correlation"]
                    corr_matrix[j, i] = item["correlation"]  # Symmetric matrix
                
                # Create a heatmap using Plotly
                fig = go.Figure(data=go.Heatmap(
                    z=corr_matrix,
                    x=all_cols,
                    y=all_cols,
                    colorscale='RdBu_r',  # Red-Blue scale, reversed
                    zmid=0,  # Center the color scale at 0
                    text=corr_matrix,
                    texttemplate="%{text:.2f}",  # Format to 2 decimals and show directly on cells
                    textfont={"size":10},
                    hovertemplate='%{y} vs %{x}<br>Correlación: %{z:.3f}<extra></extra>',
                    colorbar=dict(title='Correlación')
                ))
                
                fig.update_layout(
                    title='Matriz de Correlación',
                    height=700,
                    width=800,
                    xaxis=dict(tickangle=45),
                )
                
                # Show the plot
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No hay suficientes datos para calcular correlaciones.")

def visualize_time_trend(result):
    # Implementation for time trend visualization
    if "trends" in result:
        for col_name, trend_data in result["trends"].items():
            st.subheader(f"Tendencia temporal para {col_name}")
            
            if "error" in trend_data:
                st.error(trend_data["error"])
                continue
                
            if "data_points" in trend_data:
                data_points = trend_data["data_points"]
                
                # Create a DataFrame for visualization
                df_trend = pd.DataFrame(data_points)
                
                # Create a line chart
                fig = px.line(df_trend, x="date", y="value", markers=True,
                             title=f"Tendencia de {col_name} a lo largo del tiempo")
                
                # Add trendline if available
                if "trend" in trend_data and trend_data["trend"]:
                    trend = trend_data["trend"]
                    direction = trend.get("direction", "flat")
                    strength = trend.get("strength", 0)
                    
                    # Add the trend line
                    x = np.arange(len(data_points))
                    y = trend.get("slope", 0) * x + trend.get("intercept", 0)
                    
                    fig.add_trace(go.Scatter(x=df_trend["date"], y=y,
                                            mode="lines", name="Tendencia",
                                            line=dict(color="red", dash="dash")))
                    
                    # Add a note about the trend
                    direction_text = "ascendente" if direction == "up" else "descendente" if direction == "down" else "estable"
                    strength_text = "fuerte" if strength > 0.7 else "moderada" if strength > 0.3 else "débil"
                    
                    st.markdown(f"**Tendencia {direction_text} {strength_text}** (R² = {trend.get('r_squared', 0):.3f})")
                    
                    if trend.get("significant", False):
                        st.success("Esta tendencia es estadísticamente significativa (p < 0.05).")
                    else:
                        st.warning("Esta tendencia no es estadísticamente significativa (p >= 0.05).")
                
                # Show the plot
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No hay suficientes datos para visualizar la tendencia.")

def visualize_category_counts(result):
    # Implementation for category counts visualization
    for col, data in result.items():
        if "counts" in data:
            st.subheader(f"Distribución de {col}")
            
            counts = data["counts"]
            df_counts = pd.DataFrame(counts)
            
            # Create a bar chart
            fig = px.bar(df_counts, x="category", y="count", 
                         text="percentage",
                         title=f"Distribución de {col} (Total: {data.get('total', 0)})")
            
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            
            # Show the plot
            st.plotly_chart(fig, use_container_width=True)
            
            # For large number of categories, show a summary table
            if len(counts) > 10:
                top_5 = pd.DataFrame(counts[:5])
                with st.expander("Ver top 5 categorías en tabla"):
                    st.dataframe(top_5)

def visualize_numeric_distribution(result):
    # Implementation for numeric distribution visualization
    if "distributions" in result:
        for col, data in result["distributions"].items():
            st.subheader(f"Distribución de {col}")
            
            if "error" in data:
                st.error(data["error"])
                continue
                
            if "histogram" in data:
                hist_data = data["histogram"]
                
                # Create a histogram
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=hist_data["bin_centers"],
                    y=hist_data["counts"],
                    name="Frecuencia",
                    marker_color="lightblue"
                ))
                
                fig.update_layout(
                    title=f"Histograma de {col}",
                    xaxis_title=col,
                    yaxis_title="Frecuencia",
                    bargap=0.2
                )
                
                # Show the plot
                st.plotly_chart(fig, use_container_width=True)
                
                # Show statistics
                if "statistics" in data:
                    stats = data["statistics"]
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Media", f"{stats.get('mean', 'N/A'):.2f}" if stats.get('mean') is not None else "N/A")
                        st.metric("Skewness", f"{stats.get('skew', 'N/A'):.2f}" if stats.get('skew') is not None else "N/A")
                    
                    with col2:
                        st.metric("Mediana", f"{stats.get('median', 'N/A'):.2f}" if stats.get('median') is not None else "N/A")
                        st.metric("Kurtosis", f"{stats.get('kurtosis', 'N/A'):.2f}" if stats.get('kurtosis') is not None else "N/A")
                    
                    with col3:
                        st.metric("Desv. Est.", f"{stats.get('std', 'N/A'):.2f}" if stats.get('std') is not None else "N/A")
            else:
                st.warning("No hay suficientes datos para visualizar la distribución.")

def visualize_production_analysis(result):
    # Implementation for production analysis visualization
    st.subheader("Análisis de Producción")
    
    # Create metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if "average_production" in result:
            st.metric("Producción Promedio", f"{result['average_production']:.2f} BBL")
    
    with col2:
        if "max_production" in result:
            st.metric("Producción Máxima", f"{result['max_production']:.2f} BBL")
    
    with col3:
        if "min_production" in result:
            st.metric("Producción Mínima", f"{result['min_production']:.2f} BBL")

# Nuevas funciones de visualización para KPIs de Stimulation

def visualize_chemical_percentage(result):
    """Visualizar el porcentaje de químicos utilizados"""
    if "error" in result:
        st.warning(result["message"])
        return
        
    st.subheader("Porcentaje de Químicos Utilizados")
    
    if "visualization_data" in result:
        viz_data = result["visualization_data"]
        labels = viz_data.get("labels", [])
        values = viz_data.get("values", [])
        
        if labels and values:
            # Crear gráfico de pastel
            fig = px.pie(
                names=labels, 
                values=values,
                title="Distribución de Químicos",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Mostrar tabla de datos
            if "percentage_breakdown" in result:
                df = pd.DataFrame({
                    "Químico": labels,
                    "Porcentaje (%)": values,
                    "Volumen Total": result.get("total_volume", 0)
                })
                st.dataframe(df)
        else:
            st.info("No hay datos suficientes para visualizar el porcentaje de químicos")

def visualize_chemical_usage_by_lease(result):
    """Visualizar el uso de químicos por lease"""
    if "error" in result:
        st.warning(result["message"])
        return
        
    st.subheader("Uso de Químicos por Lease")
    
    if "visualization_data" in result:
        viz_data = result["visualization_data"]
        leases = viz_data.get("leases", [])
        chemicals = viz_data.get("chemicals", [])
        values = viz_data.get("values", [])
        
        if leases and chemicals and values:
            # Crear un heatmap
            fig = go.Figure(data=go.Heatmap(
                z=values,
                x=chemicals,
                y=leases,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Volumen")
            ))
            
            fig.update_layout(
                title="Uso de Químicos por Lease",
                xaxis_title="Químicos",
                yaxis_title="Lease"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Mostrar también un gráfico de barras apiladas
            st.subheader("Uso Total por Lease")
            
            # Crear un DataFrame para el gráfico de barras
            data = []
            for i, lease in enumerate(leases):
                for j, chemical in enumerate(chemicals):
                    data.append({
                        "Lease": lease,
                        "Químico": chemical,
                        "Volumen": values[i][j]
                    })
            
            df = pd.DataFrame(data)
            
            fig2 = px.bar(
                df, 
                x="Lease", 
                y="Volumen", 
                color="Químico",
                title="Volumen de Químicos por Lease",
                barmode="stack"
            )
            
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No hay datos suficientes para visualizar el uso de químicos por lease")

def visualize_total_activities(result):
    """Visualizar el número total de actividades con un componente visual atractivo"""
    if "error" in result:
        st.warning(result["message"])
        return
    
    # Determinar el tipo de visualización
    display_type = result.get("display_type", "single_number")
    total = result.get("total_activities", 0)
    
    # Estilo CSS para el número grande
    st.markdown("""
    <style>
    .big-number {
        font-size: 80px;
        font-weight: bold;
        text-align: center;
        color: #1E88E5;
        margin: 20px 0;
    }
    .number-label {
        font-size: 24px;
        text-align: center;
        color: #616161;
        margin: 10px 0 30px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Mostrar el número total de actividades
    st.markdown(f'<div class="big-number">{total}</div>', unsafe_allow_html=True)
    st.markdown('<div class="number-label">Actividades Totales</div>', unsafe_allow_html=True)
    
    # Si tenemos datos de tendencia, mostrar un gráfico
    if display_type == "number_with_trend" and "visualization_data" in result:
        viz_data = result["visualization_data"]
        labels = viz_data.get("labels", [])
        values = viz_data.get("values", [])
        
        if labels and values:
            # Crear un gráfico de línea para mostrar la tendencia
            st.subheader("Tendencia Mensual de Actividades")
            
            fig = go.Figure()
            
            # Añadir gráfico de barras
            fig.add_trace(go.Bar(
                x=labels,
                y=values,
                name="Actividades",
                marker_color="#1E88E5"
            ))
            
            # Añadir línea de tendencia
            fig.add_trace(go.Scatter(
                x=labels,
                y=values,
                mode='lines+markers',
                name='Tendencia',
                line=dict(color='#FFC107', width=3)
            ))
            
            fig.update_layout(
                xaxis_title="Mes",
                yaxis_title="Número de Actividades",
                hovermode="x"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calcular métricas adicionales
            if len(values) > 1:
                # Calcular cambio porcentual desde el primer mes
                first_value = values[0]
                last_value = values[-1]
                if first_value > 0:
                    percent_change = ((last_value - first_value) / first_value) * 100
                    
                    # Mostrar métricas de cambio
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Primer Mes", values[0], delta=None)
                    
                    with col2:
                        st.metric("Último Mes", values[-1], delta=f"{percent_change:.1f}%")
                    
                    with col3:
                        avg = sum(values) / len(values)
                        st.metric("Promedio Mensual", f"{avg:.1f}")

def visualize_revenue_by_intervention(result):
    """Visualizar los ingresos por tipo de intervención"""
    if "error" in result:
        st.warning(result["message"])
        return
    
    st.subheader("Ingresos por Tipo de Intervención")
    
    if "visualization_data" in result:
        viz_data = result["visualization_data"]
        labels = viz_data.get("labels", [])
        values = viz_data.get("values", [])
        
        if labels and values:
            # Crear un gráfico de barras horizontal
            df = pd.DataFrame({
                "Tipo de Intervención": labels,
                "Ingresos": values
            })
            
            fig = px.bar(
                df,
                y="Tipo de Intervención",
                x="Ingresos",
                orientation='h',
                title="Ingresos por Tipo de Intervención",
                color="Ingresos",
                color_continuous_scale=px.colors.sequential.Blues
            )
            
            # Formatear los valores en el eje X para mostrar moneda
            fig.update_layout(
                xaxis_title="Ingresos ($)",
                yaxis_title="",
                xaxis=dict(tickprefix="$", tickformat=",.0f")
            )
            
            # Añadir etiquetas con valores
            fig.update_traces(texttemplate="$%{x:,.0f}", textposition="outside")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Mostrar métricas de resumen
            col1, col2 = st.columns(2)
            
            with col1:
                total_revenue = sum(values)
                st.metric("Ingresos Totales", f"${total_revenue:,.2f}")
            
            with col2:
                if len(values) > 0:
                    avg_revenue = total_revenue / len(values)
                    st.metric("Ingreso Promedio por Tipo", f"${avg_revenue:,.2f}")
        else:
            st.info("No hay datos suficientes para visualizar los ingresos por tipo de intervención") 