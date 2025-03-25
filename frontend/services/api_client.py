import requests
import streamlit as st
import pandas as pd
import json

# Configuración de API
API_URL = "http://localhost:8000"

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
        st.info("Enviando solicitud al servidor... esto puede tardar unos momentos.")
        response = requests.post(
            f"{API_URL}/generate-ai-kpi",
            json={
                "user_prompt": user_prompt,
                "service_line": service_line,
                "mapping_path": mapping_path
            },
            timeout=120  # Increase timeout to 2 minutes for complex generation
        )
        
        try:
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as http_err:
            # Try to extract error details from the response
            try:
                error_info = response.json()
                error_detail = error_info.get('detail', str(http_err))
                st.error(f"Error del servidor: {error_detail}")
            except:
                st.error(f"Error de API: {str(http_err)}")
            return None
            
    except requests.exceptions.RequestException as e:
        st.error(f"Error de conexión con la API: {str(e)}")
        st.warning("Asegúrate de que el servidor backend esté ejecutándose y que la API esté accesible.")
        return None
    except Exception as e:
        st.error(f"Error inesperado: {str(e)}")
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

def interpret_kpi(kpi_name, kpi_result):
    """Obtener interpretación de IA para un KPI específico"""
    try:
        response = requests.post(
            f"{API_URL}/interpret-kpi",
            json={
                "kpi_name": kpi_name,
                "kpi_result": kpi_result
            }
        )
        response.raise_for_status()
        return response.json()["interpretation"]
    except requests.exceptions.RequestException as e:
        st.error(f"Error al obtener interpretación de IA: {str(e)}")
        return "No se pudo obtener la interpretación. Por favor intente nuevamente más tarde." 