import os
import logging
import openai
import json
import traceback
from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from backend.dataset import get_dataframe

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Create router
router = APIRouter()

class KpiInterpretRequest(BaseModel):
    kpi_name: str
    kpi_result: Any

class ChatQueryRequest(BaseModel):
    query: str

def get_data_for_query(query, data_summary):
    # This function would normally extract relevant data based on the query
    # For now, we'll just return a sample
    return "Sample data extracted for the query"

@router.post("/interpret-kpi")
async def interpret_kpi(request: KpiInterpretRequest):
    """
    Generate an AI interpretation of KPI results.
    """
    logger.info(f"Starting interpret_kpi for {request.kpi_name}")
    
    # Check if we have a loaded dataset
    df = get_dataframe()
    if df is None:
        return {"interpretation": "No dataset loaded. Please load a dataset first."}
    
    # Get OpenAI API key from environment
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        logger.error("OpenAI API key not found in environment")
        return {"interpretation": "API key not found. Please check your environment setup."}
    
    logger.info(f"API Key found with length: {len(api_key)}")
    
    # Set up the OpenAI configuration
    openai.api_key = api_key
    
    # Prepare dataset context
    dataset_context = df.head(10).to_string()
    
    # Special case for the correlation matrix to simplify results
    if request.kpi_name == "MatrizCorrelación":
        # Filter strong correlations to simplify the response
        try:
            correlation_data = json.loads(request.kpi_result)
            strong_correlations = {}
            
            for var1, values in correlation_data.items():
                for var2, value in values.items():
                    if var1 != var2 and abs(value) > 0.7:
                        if var1 not in strong_correlations:
                            strong_correlations[var1] = {}
                        strong_correlations[var1][var2] = value
            
            request.kpi_result = json.dumps(strong_correlations)
        except Exception as e:
            logger.error(f"Error processing correlation matrix: {e}")
    
    # Generate prompt for the KPI interpretation
    prompt = f"""
    Eres un experto en análisis de datos para la industria petrolera. Por favor, responde siempre en español.
    
    Analiza el siguiente resultado de KPI para '{request.kpi_name}' y proporciona una interpretación detallada:
    
    {request.kpi_result}
    
    Considera esta muestra del conjunto de datos para contexto adicional:
    {dataset_context}
    
    Proporciona una interpretación detallada que incluya:
    1. ¿Qué significa este valor de KPI en el contexto de las operaciones petroleras?
    2. ¿Cuáles son las implicaciones para la eficiencia operativa?
    3. ¿Hay alguna recomendación basada en este valor?
    4. ¿Cómo se compara con estándares o expectativas de la industria?
    
    Tu respuesta debería ser específica, hacer referencia a valores concretos de datos y proporcionar sugerencias de acción.
    """
    
    logger.info(f"Generated prompt with length: {len(prompt)}")
    
    try:
        # Call OpenAI API using the v0.28.0 style
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Eres un experto en análisis de datos para la industria petrolera. Por favor, responde siempre en español."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.5
        )
        
        # Extract the interpretation from the response
        interpretation = response.choices[0].message.content.strip()
        logger.info(f"Generated interpretation with length: {len(interpretation)}")
        
        return {"interpretation": interpretation}
    except Exception as e:
        logger.error(f"Error calling OpenAI API: {e}")
        
        # Provide a fallback interpretation
        fallback_message = f"""
        Lo siento, no pude generar una interpretación detallada para el KPI '{request.kpi_name}' en este momento.
        
        Aquí tienes una interpretación genérica:
        
        El valor KPI proporcionado representa métricas importantes para tus operaciones petroleras.
        Para obtener un análisis más detallado, asegúrate de que tu clave API de OpenAI esté correctamente configurada
        e intenta nuevamente más tarde.
        
        Detalles del error: {str(e)}
        """
        
        return {"interpretation": fallback_message.strip()}

@router.post("/chat-query")
async def chat_query(request: ChatQueryRequest):
    """
    Generate an AI response to a user query about the dataset.
    """
    logger.info(f"Starting chat_query for: {request.query}")
    
    # Get the dataset
    df = get_dataframe()
    if df is None:
        return {"response": "No dataset loaded. Please load a dataset first."}
    
    # Get OpenAI API key from environment
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        logger.error("OpenAI API key not found in environment")
        return {"response": "API key not found. Please check your environment setup."}
    
    logger.info(f"API Key found with length: {len(api_key)}")
    
    # Prepare dataset context
    data_summary = df.describe().to_string()
    
    # Get additional data relevant to the query
    query_data = get_data_for_query(request.query, data_summary)
    
    # Generate prompt for the chat query
    prompt = f"""
    Eres un experto en análisis de datos para la industria petrolera. Por favor, responde siempre en español.
    
    Consulta del usuario: {request.query}
    
    Resumen del conjunto de datos:
    {data_summary}
    
    Datos relevantes:
    {query_data}
    
    Por favor, proporciona una respuesta detallada a la consulta del usuario basada en los datos proporcionados.
    Tu respuesta debería ser específica, hacer referencia a valores concretos de datos cuando sea aplicable, 
    y proporcionar sugerencias de acción en el contexto de las operaciones petroleras.
    """
    
    logger.info(f"Generated prompt with length: {len(prompt)}")
    
    try:
        # Set up the OpenAI configuration
        openai.api_key = api_key
        
        # Call OpenAI API using the v0.28.0 style
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Eres un experto en análisis de datos para la industria petrolera. Por favor, responde siempre en español."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.5
        )
        
        # Extract the AI response from the response
        ai_response = response.choices[0].message.content.strip()
        logger.info(f"Generated AI response with length: {len(ai_response)}")
        
        return {"response": ai_response}
    except Exception as e:
        logger.error(f"Error calling OpenAI API: {e}")
        
        # Provide a fallback response
        fallback_message = f"""
        Lo siento, no pude generar una respuesta detallada a tu consulta en este momento.
        
        Aquí tienes un análisis básico basado en los datos disponibles:
        
        El conjunto de datos contiene varias métricas relevantes para las operaciones petroleras. Para obtener un
        análisis más detallado de tu consulta específica, asegúrate de que tu clave API de OpenAI esté
        correctamente configurada e intenta nuevamente más tarde.
        
        Detalles del error: {str(e)}
        """
        
        return {"response": fallback_message.strip()}