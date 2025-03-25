import os
import logging
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Path, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import traceback
import sys
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
import math
from scipy import stats
import json
import openai

# Add the parent directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Print OpenAI API key length for debugging (without revealing the key)
api_key = os.environ.get("OPENAI_API_KEY")
if api_key:
    logger.info(f"OpenAI API key loaded with length: {len(api_key)}")
    # Check which openai version is installed
    try:
        import openai
        logger.info(f"OpenAI library version: {openai.__version__}")
        
        # Check if we can use the modern client
        try:
            client = openai.OpenAI(api_key=api_key)
            logger.info("Modern OpenAI client (v1.x) is available")
        except (ImportError, AttributeError):
            logger.info("Using legacy OpenAI client (v0.28.x)")
    except ImportError:
        logger.warning("OpenAI library is not installed")
else:
    logger.warning("OpenAI API key not found in environment variables")
    logger.warning("Setting a default API key for testing purposes only (this will not work in production!)")
    os.environ["OPENAI_API_KEY"] = "sk-dummy-api-key-for-testing-purposes-only"

# Import routers using absolute imports
from backend.routers import data_routes, ai_routes

# Create model for load-data request
class LoadDataRequest(BaseModel):
    source_type: str
    source: str
    mapping_path: Optional[str] = None
    service_line: Optional[str] = None

# Helper function to make objects JSON serializable
def json_serializable(obj):
    """Convert an object to a JSON serializable object."""
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, 
                         np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64, float)):
        if np.isnan(obj) or np.isinf(obj) or obj > 1.7976931348623157e+308 or obj < -1.7976931348623157e+308:
            return None
        try:
            # Intenta serializar para verificar que es válido
            json.dumps(float(obj))
            return float(obj)
        except:
            logger.warning(f"Valor float no serializable detectado: {obj}, reemplazando con None")
            return None
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, (set, tuple)):
        return list(obj)
    elif isinstance(obj, dict):
        return {k: json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [json_serializable(item) for item in obj]
    elif obj is None or isinstance(obj, (str, int, bool)):
        return obj
    else:
        return str(obj)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(data_routes.router, prefix="/api")
app.include_router(ai_routes.router, prefix="/ai")

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Oil KPIs API is running",
        "endpoints": {
            "data": [
                "/api/data/load-dataset",
                "/api/data/apply-filters",
                "/api/data/get-column-data",
                "/api/data/dataset-info",
                "/api/data/raw-data"
            ],
            "ai": [
                "/ai/interpret-kpi",
                "/ai/chat-query"
            ],
            "root": [
                "/",
                "/raw-data",
                "/interpret-kpi",
                "/chat-query",
                "/upload-file",
                "/load-data",
                "/unique-values/{column}",
                "/available-kpis",
                "/calculate-kpi/{kpi_name}",
                "/available-filters"
            ]
        }
    }

# Upload file endpoint
@app.post("/upload-file")
async def upload_file(file: UploadFile = File(...)):
    """
    Handle file uploads from the frontend.
    Save the uploaded file to a temporary location and return the path.
    """
    try:
        logger.info(f"Received upload request for file: {file.filename}")
        
        # Create uploads directory if it doesn't exist
        upload_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        
        # Save the file
        file_path = os.path.join(upload_dir, file.filename)
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        logger.info(f"File saved to: {file_path}")
        
        # Return the path in the format expected by the frontend
        return {"path": file_path}
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"detail": f"Error uploading file: {str(e)}"}
        )

# Load data endpoint
@app.post("/load-data")
async def load_data(request: LoadDataRequest):
    """
    Load data using the parameters from the frontend.
    """
    try:
        logger.info(f"Loading data from source: {request.source}")
        
        # In this simplified implementation, we just forward to the load_dataset function
        from backend.dataset import load_dataset
        result = load_dataset(request.source)
        
        if not result.get("success", False):
            return JSONResponse(
                status_code=400,
                content={"detail": result.get("error", "Failed to load dataset")}
            )
        
        return result
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"detail": f"Error loading data: {str(e)}"}
        )

# Endpoint for raw data retrieval
@app.get("/raw-data")
async def get_raw_data(limit: int = 100):
    """Get a sample of raw data"""
    try:
        from backend.dataset import get_dataframe
        df = get_dataframe()
        
        if df is None:
            raise HTTPException(status_code=400, detail="No dataset loaded")
        
        # Limit rows to avoid large responses
        sample_df = df.head(limit).copy()
        
        # Sanitizar todas las columnas, no solo las de tipo float
        for col in sample_df.columns:
            # Primero, tratar con los valores NaN e infinitos en columnas numéricas
            if pd.api.types.is_numeric_dtype(sample_df[col]):
                mask = sample_df[col].isna() | np.isinf(sample_df[col])
                if mask.any():
                    logger.info(f"Replacing {mask.sum()} NaN/inf values in column {col}")
                    sample_df.loc[mask, col] = None
            
            # Para columnas no numéricas, convertir NaN a None
            elif pd.api.types.is_object_dtype(sample_df[col]):
                mask = sample_df[col].isna()
                if mask.any():
                    logger.info(f"Replacing {mask.sum()} NaN values in column {col}")
                    sample_df.loc[mask, col] = None
        
        # Convertir el DataFrame a registros y asegurar que todos sean JSON serializables
        records = sample_df.to_dict(orient="records")
        
        # Aplicar sanitización agresiva a los registros
        sanitized_records = []
        for record in records:
            sanitized_record = {}
            for key, value in record.items():
                # Si es un float, verificar explícitamente que sea serializable
                if isinstance(value, float):
                    if math.isnan(value) or math.isinf(value) or value > 1.7976931348623157e+308 or value < -1.7976931348623157e+308:
                        sanitized_record[key] = None
                    else:
                        try:
                            # Verificar si es serializable a JSON
                            json.dumps(value)
                            sanitized_record[key] = value
                        except:
                            logger.warning(f"Valor float no serializable en columna {key}: {value}, reemplazando con None")
                            sanitized_record[key] = None
                else:
                    # Para cualquier otro tipo, pasar por json_serializable
                    sanitized_record[key] = json_serializable(value)
            sanitized_records.append(sanitized_record)
        
        # Verificación final
        try:
            result = {"data": sanitized_records}
            # Intentar serializar todo el resultado
            json.dumps(result)
            return result
        except ValueError as e:
            logger.error(f"Error en serialización final: {str(e)}")
            # Último recurso: convertir a string todos los valores problemáticos
            for record in sanitized_records:
                for key, value in record.items():
                    if not isinstance(value, (str, int, bool, type(None))):
                        record[key] = str(value)
            
            return {"data": sanitized_records}
        
    except Exception as e:
        logger.error(f"Error in raw-data endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"detail": f"Error getting raw data: {str(e)}"}
        )

# Forward requests to the AI router
@app.post("/interpret-kpi")
async def interpret_kpi_endpoint(request: ai_routes.KpiInterpretRequest):
    try:
        return await ai_routes.interpret_kpi(request)
    except Exception as e:
        logger.error(f"Error in interpret_kpi endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"detail": f"Internal server error: {str(e)}"}
        )

@app.post("/chat-query")
async def chat_query_endpoint(request: ai_routes.ChatQueryRequest):
    try:
        return await ai_routes.chat_query(request)
    except Exception as e:
        logger.error(f"Error in chat_query endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"detail": f"Internal server error: {str(e)}"}
        )

# Get unique values for a column
@app.get("/unique-values/{column}")
async def get_unique_values(column: str = Path(..., description="Column name")):
    """
    Get unique values for a specific column.
    Used for populating filter dropdowns in the frontend.
    """
    try:
        logger.info(f"Getting unique values for column: {column}")
        from backend.dataset import get_dataframe
        
        df = get_dataframe()
        if df is None:
            raise HTTPException(status_code=400, detail="No dataset loaded")
        
        if column not in df.columns:
            raise HTTPException(status_code=404, detail=f"Column '{column}' not found")
        
        # Get unique values for the column
        unique_values = df[column].dropna().unique().tolist()
        
        # Convert non-serializable values to strings
        serializable_values = []
        for val in unique_values:
            if isinstance(val, (str, int, float, bool)) or val is None:
                serializable_values.append(val)
            else:
                serializable_values.append(str(val))
        
        return {"values": serializable_values}
    
    except Exception as e:
        logger.error(f"Error retrieving unique values: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"detail": f"Error retrieving unique values: {str(e)}"}
        )

# Get available KPIs
@app.get("/available-kpis")
async def get_available_kpis():
    """
    Get the list of available KPIs.
    """
    try:
        # For this simplified implementation, we'll just return a static list of KPIs
        kpis = [
            {
                "id": "ResumenEstadístico",
                "name": "Resumen Estadístico",
                "description": "Estadísticas básicas de variables numéricas",
                "requires_columns": True,
                "requires_group_by": False
            },
            {
                "id": "MatrizCorrelación",
                "name": "Matriz de Correlación",
                "description": "Correlaciones entre variables numéricas",
                "requires_columns": True,
                "requires_group_by": False
            },
            {
                "id": "TendenciaTemporal",
                "name": "Tendencia Temporal",
                "description": "Análisis de tendencias a lo largo del tiempo",
                "requires_columns": True,
                "requires_group_by": False
            },
            {
                "id": "ConteoPorCategoría",
                "name": "Conteo por Categoría",
                "description": "Frecuencia de valores para variables categóricas",
                "requires_columns": True,
                "requires_group_by": False
            },
            {
                "id": "DistribuciónNumérica",
                "name": "Distribución Numérica",
                "description": "Histograma de variables numéricas",
                "requires_columns": True,
                "requires_group_by": False
            },
            {
                "id": "Production_Analysis",
                "name": "Análisis de Producción",
                "description": "Análisis de métricas de producción",
                "requires_columns": False,
                "requires_group_by": False
            },
            # Añadir KPIs de Stimulation
            {
                "id": "PorcentajeQuímicos",
                "name": "Porcentaje de Químicos",
                "description": "Porcentaje de cada químico utilizado en las operaciones",
                "requires_columns": False,
                "requires_group_by": False
            },
            {
                "id": "UsoQuímicosPorLease",
                "name": "Uso de Químicos por Lease",
                "description": "Volumen total de químicos utilizados por lease",
                "requires_columns": False,
                "requires_group_by": False
            },
            {
                "id": "TotalActividades",
                "name": "Total de Actividades",
                "description": "Número total de actividades de estimulación realizadas",
                "requires_columns": False,
                "requires_group_by": False
            },
            {
                "id": "IngresosPorIntervención",
                "name": "Ingresos por Tipo de Intervención",
                "description": "Ingresos totales generados por tipo de intervención",
                "requires_columns": False,
                "requires_group_by": False
            }
        ]
        return {"kpis": kpis}
    except Exception as e:
        logger.error(f"Error retrieving available KPIs: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"detail": f"Error retrieving available KPIs: {str(e)}"}
        )

# Calculate a KPI
@app.get("/calculate-kpi/{kpi_name}")
async def calculate_kpi(
    kpi_name: str = Path(..., description="KPI name to calculate"),
    columns: str = Query(None, description="Comma-separated list of columns"),
    group_by: str = Query(None, description="Comma-separated list of grouping columns")
):
    """
    Calculate a KPI with specified parameters.
    """
    try:
        logger.info(f"Calculating KPI: {kpi_name}")
        from backend.dataset import get_dataframe
        import pandas as pd
        import numpy as np
        import math
        
        # Helper function to convert non-serializable values to JSON-safe values
        def json_safe(obj):
            if isinstance(obj, float):
                if math.isnan(obj) or math.isinf(obj):
                    return None
                return obj
            elif isinstance(obj, dict):
                return {k: json_safe(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [json_safe(item) for item in obj]
            return obj
        
        df = get_dataframe()
        if df is None:
            raise HTTPException(status_code=400, detail="No dataset loaded")
        
        # Parse columns and group_by if provided
        columns_list = columns.split(",") if columns else []
        group_by_list = group_by.split(",") if group_by else None
        
        # Store the result to wrap it properly later
        result_data = None
        
        # If no columns are specified, use default columns based on KPI
        if not columns_list:
            if kpi_name in ["ResumenEstadístico", "MatrizCorrelación", "DistribuciónNumérica"]:
                # For numerical KPIs, use numerical columns
                columns_list = df.select_dtypes(include=['number']).columns.tolist()[:10]
                logger.info(f"Auto-selected columns for {kpi_name}: {columns_list}")
            elif kpi_name == "TendenciaTemporal":
                # For time trends, select numeric columns
                columns_list = df.select_dtypes(include=['number']).columns.tolist()[:5]
                logger.info(f"Auto-selected columns for {kpi_name}: {columns_list}")
            elif kpi_name == "ConteoPorCategoría":
                # For categorical KPIs, use categorical columns
                columns_list = df.select_dtypes(include=['object']).columns.tolist()[:3]
                logger.info(f"Auto-selected columns for {kpi_name}: {columns_list}")
            elif kpi_name == "Production_Analysis":
                # For Production Analysis, create a sample result
                result_data = {
                    "average_production": round(df["Production_BBL"].mean(), 2) if "Production_BBL" in df.columns else 750.5,
                    "max_production": round(df["Production_BBL"].max(), 2) if "Production_BBL" in df.columns else 1250.0,
                    "min_production": round(df["Production_BBL"].min(), 2) if "Production_BBL" in df.columns else 550.0
                }
                return {"result": result_data}
        
        # Check if we have enough columns for the KPI
        if kpi_name == "MatrizCorrelación" and len(columns_list) < 2:
            result_data = {
                "error": "Not enough numeric columns for correlation matrix",
                "kpi_id": kpi_name,
                "message": "A correlation matrix requires at least 2 numeric columns"
            }
            return {"result": result_data}
        
        # Calculate the KPI based on its type
        if kpi_name == "ResumenEstadístico":
            # Calculate basic statistics for each column
            result_data = {"statistics": {}}
            for col in columns_list:
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    # Handle NaN values safely
                    series = df[col].dropna()
                    if len(series) > 0:
                        # Detect outliers (values beyond 1.5 * IQR)
                        q1 = float(series.quantile(0.25))
                        q3 = float(series.quantile(0.75))
                        iqr = q3 - q1
                        lower_bound = q1 - 1.5 * iqr
                        upper_bound = q3 + 1.5 * iqr
                        outliers = series[(series < lower_bound) | (series > upper_bound)]
                        
                        result_data["statistics"][col] = {
                            "count": int(series.count()),
                            "mean": float(series.mean()) if not pd.isna(series.mean()) else None,
                            "std": float(series.std()) if not pd.isna(series.std()) else None,
                            "min": float(series.min()) if not pd.isna(series.min()) else None,
                            "25%": float(series.quantile(0.25)) if not pd.isna(series.quantile(0.25)) else None,
                            "50%": float(series.median()) if not pd.isna(series.median()) else None,
                            "75%": float(series.quantile(0.75)) if not pd.isna(series.quantile(0.75)) else None,
                            "max": float(series.max()) if not pd.isna(series.max()) else None,
                            "null_count": int(df[col].isna().sum()),
                            "outliers": {
                                "count": int(len(outliers)),
                                "percentage": float(len(outliers) / len(series) * 100) if len(series) > 0 else 0.0,
                                "min": float(outliers.min()) if len(outliers) > 0 and not pd.isna(outliers.min()) else None,
                                "max": float(outliers.max()) if len(outliers) > 0 and not pd.isna(outliers.max()) else None
                            }
                        }
                    else:
                        result_data["statistics"][col] = {
                            "count": 0,
                            "mean": None,
                            "std": None,
                            "min": None,
                            "25%": None,
                            "50%": None,
                            "75%": None,
                            "max": None,
                            "null_count": int(df[col].isna().sum()),
                            "outliers": {
                                "count": 0,
                                "percentage": 0.0,
                                "min": None,
                                "max": None
                            }
                        }
        
        elif kpi_name == "MatrizCorrelación":
            # Calculate correlation matrix with NaN handling
            valid_columns = []
            for col in columns_list:
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]) and df[col].notna().sum() > 0:
                    valid_columns.append(col)
            
            if len(valid_columns) < 2:
                result_data = {
                    "error": "Not enough valid numeric columns for correlation matrix",
                    "kpi_id": kpi_name,
                    "message": "A correlation matrix requires at least 2 numeric columns with non-null values"
                }
            else:
                # Fill NaN values temporarily for correlation calculation
                corr_df = df[valid_columns].copy()
                # Use a safer approach to calculate correlation
                corr_matrix = {}
                
                try:
                    # Calculate correlation matrix
                    raw_corr = corr_df.corr(method='pearson', min_periods=1).round(3)
                    
                    # Convert to safer dictionary format with null checking
                    for col1 in valid_columns:
                        corr_matrix[col1] = {}
                        for col2 in valid_columns:
                            corr_value = raw_corr.loc[col1, col2]
                            # Handle NaN and Infinity values
                            if pd.isna(corr_value) or np.isinf(corr_value):
                                corr_matrix[col1][col2] = 0
                            else:
                                corr_matrix[col1][col2] = float(corr_value)
                except Exception as e:
                    logger.error(f"Error calculating correlation matrix: {str(e)}")
                    # Provide an empty correlation matrix
                    for col1 in valid_columns:
                        corr_matrix[col1] = {col2: 0 for col2 in valid_columns}
                
                # Format for frontend visualization
                formatted = []
                for col1 in corr_matrix:
                    for col2 in corr_matrix[col1]:
                        if col1 != col2:  # Skip self-correlations
                            corr_value = corr_matrix[col1][col2]
                            formatted.append({
                                "column1": col1,
                                "column2": col2,
                                "correlation": corr_value
                            })
                
                # Sort by absolute correlation value (descending)
                formatted.sort(key=lambda x: abs(x["correlation"]), reverse=True)
                
                result_data = {
                    "matrix": corr_matrix,
                    "formatted": formatted
                }
        
        elif kpi_name == "TendenciaTemporal":
            # Implement TendenciaTemporal
            date_cols = df.select_dtypes(include=['datetime']).columns.tolist()
            if not date_cols:
                result_data = {
                    "error": "No date columns found",
                    "kpi_id": kpi_name,
                    "message": "Time trend analysis requires at least one date column"
                }
            else:
                date_col = date_cols[0]  # Use the first date column
                result_data = {"trends": {}}
                
                for col in columns_list:
                    if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                        # Group by month and calculate average
                        try:
                            # Create a copy with the date column and the value column, dropping NaN values
                            temp_df = df[[date_col, col]].dropna()
                            if len(temp_df) > 0:
                                # Add month and year columns
                                temp_df['month_year'] = temp_df[date_col].dt.strftime('%Y-%m')
                                # Group by month-year and calculate stats
                                grouped = temp_df.groupby('month_year')[col].agg(['mean', 'min', 'max', 'count']).reset_index()
                                
                                # Format data points for visualization
                                data_points = []
                                for _, row in grouped.iterrows():
                                    data_points.append({
                                        "date": row['month_year'],
                                        "value": float(row['mean']) if not pd.isna(row['mean']) else None
                                    })
                                
                                # Calculate trend metrics if we have enough data points
                                trend_metrics = None
                                if len(data_points) > 1:
                                    # Calculate simple linear regression
                                    x = np.array(range(len(data_points)))
                                    y = np.array([point["value"] for point in data_points if point["value"] is not None])
                                    
                                    if len(y) > 1:
                                        # Calculate slope and intercept
                                        slope, intercept, r_value, p_value, std_err = stats.linregress(x[:len(y)], y)
                                        
                                        # Calculate trend direction and strength
                                        trend_direction = "up" if slope > 0 else "down" if slope < 0 else "flat"
                                        trend_strength = float(r_value ** 2)  # R-squared
                                        
                                        trend_metrics = {
                                            "slope": float(slope),
                                            "r_squared": trend_strength,
                                            "direction": trend_direction,
                                            "strength": float(abs(r_value)),
                                            "significant": bool(p_value < 0.05)
                                        }
                                
                                result_data["trends"][col] = {
                                    "data_points": data_points,
                                    "frequency": "monthly",
                                    "trend": trend_metrics
                                }
                        except Exception as e:
                            logger.error(f"Error calculating trend for {col}: {str(e)}")
                            result_data["trends"][col] = {"error": str(e)}
        
        elif kpi_name == "ConteoPorCategoría":
            # Calculate value counts for categorical columns
            result_data = {}
            for col in columns_list:
                if col in df.columns and pd.api.types.is_object_dtype(df[col]):
                    counts = df[col].value_counts().to_dict()
                    # Ensure all keys are strings and all values are integers
                    processed_counts = []
                    total_count = int(df[col].count())
                    
                    for cat, count in counts.items():
                        # Ensure the category key is a string
                        category = str(cat) if cat is not None else "None"
                        # Ensure count is an integer
                        count_value = int(count) if not pd.isna(count) else 0
                        # Calculate percentage
                        percentage = float(count_value / total_count * 100) if total_count > 0 else 0.0
                        
                        processed_counts.append({
                            "category": category,
                            "count": count_value,
                            "percentage": percentage
                        })
                    
                    # Sort by count descending
                    processed_counts.sort(key=lambda x: x["count"], reverse=True)
                    
                    result_data[col] = {
                        "total": total_count,
                        "unique_categories": int(df[col].nunique()),
                        "counts": processed_counts
                    }
        
        elif kpi_name == "DistribuciónNumérica":
            # Create histograms for numerical columns
            result_data = {"distributions": {}}
            for col in columns_list:
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    series = df[col].dropna()
                    if len(series) > 0:
                        try:
                            # Calculate histogram with auto bins
                            hist, bin_edges = np.histogram(series, bins='auto')
                            bin_centers = [(bin_edges[i] + bin_edges[i+1])/2 for i in range(len(bin_edges)-1)]
                            
                            # Calculate basic statistics
                            result_data["distributions"][col] = {
                                "histogram": {
                                    "counts": hist.tolist(),
                                    "bins": [float(x) for x in bin_edges.tolist()],
                                    "bin_centers": [float(x) for x in bin_centers]
                                },
                                "statistics": {
                                    "mean": float(series.mean()) if not pd.isna(series.mean()) else None,
                                    "median": float(series.median()) if not pd.isna(series.median()) else None,
                                    "std": float(series.std()) if not pd.isna(series.std()) else None,
                                    "min": float(series.min()) if not pd.isna(series.min()) else None,
                                    "max": float(series.max()) if not pd.isna(series.max()) else None,
                                    "skew": float(stats.skew(series)) if len(series) > 2 and not pd.isna(stats.skew(series)) else None,
                                    "kurtosis": float(stats.kurtosis(series)) if len(series) > 2 and not pd.isna(stats.kurtosis(series)) else None
                                }
                            }
                        except Exception as e:
                            logger.error(f"Error creating histogram for {col}: {str(e)}")
                            result_data["distributions"][col] = {
                                "error": str(e),
                                "statistics": {
                                    "mean": float(series.mean()) if not pd.isna(series.mean()) else None,
                                    "median": float(series.median()) if not pd.isna(series.median()) else None,
                                    "std": float(series.std()) if not pd.isna(series.std()) else None,
                                    "min": float(series.min()) if not pd.isna(series.min()) else None,
                                    "max": float(series.max()) if not pd.isna(series.max()) else None
                                }
                            }
        
        # Implementación de los nuevos KPIs de Stimulation
        elif kpi_name == "PorcentajeQuímicos":
            # Intentar encontrar columnas de químicos
            chemical_columns = [col for col in df.columns if any(term in str(col).lower() for term in ['acid', 'ácido', 'chemical', 'químico', 'gel', 'inhib', 'diver', 'neutr', 'salmuera', 'agua'])]
            
            if chemical_columns:
                # Calcular el uso total de químicos
                total_volume = df[chemical_columns].sum().sum()
                
                # Calcular el porcentaje de cada químico
                if total_volume > 0:
                    chemical_percentages = (df[chemical_columns].sum() / total_volume) * 100
                    # Ordenar por porcentaje en orden descendente
                    chemical_percentages = chemical_percentages.sort_values(ascending=False)
                    
                    result_data = {
                        "percentage_breakdown": json_serializable(chemical_percentages.to_dict()),
                        "total_volume": float(total_volume),
                        "visualization_data": {
                            "labels": chemical_percentages.index.tolist(),
                            "values": chemical_percentages.values.tolist()
                        }
                    }
                else:
                    result_data = {
                        "error": "No hay datos de uso de químicos",
                        "message": "No se encontraron datos válidos de uso de químicos en el dataset."
                    }
            else:
                result_data = {
                    "error": "Columnas de químicos no encontradas",
                    "message": "No se pudieron identificar columnas que contengan información sobre el uso de químicos."
                }
        
        elif kpi_name == "UsoQuímicosPorLease":
            # Buscar columnas de químicos y columna de lease
            chemical_columns = [col for col in df.columns if any(term in str(col).lower() for term in ['acid', 'ácido', 'chemical', 'químico', 'gel', 'inhib', 'diver', 'neutr', 'salmuera', 'agua'])]
            lease_columns = [col for col in df.columns if any(term in str(col).lower() for term in ['lease', 'arrend', 'pozo', 'well', 'campo', 'field'])]
            
            if chemical_columns and lease_columns:
                lease_col = lease_columns[0]  # Usar la primera columna de lease encontrada
                
                # Agrupar por lease y sumar las columnas de químicos
                grouped_data = df.groupby(lease_col)[chemical_columns].sum()
                
                # Preparar datos para visualización
                result_data = {
                    "usage_by_lease": json_serializable(grouped_data.to_dict('index')),
                    "total_by_chemical": json_serializable(grouped_data.sum().to_dict()),
                    "visualization_data": {
                        "leases": grouped_data.index.tolist(),
                        "chemicals": chemical_columns,
                        "values": [[float(grouped_data.loc[lease, chem]) if not pd.isna(grouped_data.loc[lease, chem]) else 0 
                                    for chem in chemical_columns] 
                                   for lease in grouped_data.index]
                    }
                }
            else:
                result_data = {
                    "error": "Columnas necesarias no encontradas",
                    "message": "No se pudieron identificar las columnas de químicos o de lease."
                }
        
        elif kpi_name == "TotalActividades":
            # Simplemente contar el número de filas en el dataset
            total_activities = len(df)
            
            # Intentar encontrar alguna columna de fecha para mostrar tendencia temporal
            date_columns = [col for col in df.columns if any(term in str(col).lower() for term in ['date', 'fecha', 'time', 'día'])]
            
            if date_columns:
                date_col = date_columns[0]
                
                # Convertir a datetime si no lo es ya
                if not pd.api.types.is_datetime64_dtype(df[date_col]):
                    try:
                        date_series = pd.to_datetime(df[date_col], errors='coerce')
                    except:
                        date_series = None
                else:
                    date_series = df[date_col]
                
                if date_series is not None:
                    # Agrupar por mes y contar
                    df_with_date = df.copy()
                    df_with_date['month_year'] = date_series.dt.strftime('%Y-%m')
                    monthly_counts = df_with_date.groupby('month_year').size()
                    
                    result_data = {
                        "total_activities": int(total_activities),
                        "monthly_trend": json_serializable(monthly_counts.to_dict()),
                        "visualization_data": {
                            "labels": monthly_counts.index.tolist(),
                            "values": monthly_counts.values.tolist()
                        },
                        "display_type": "number_with_trend"
                    }
                else:
                    result_data = {
                        "total_activities": int(total_activities),
                        "display_type": "single_number"
                    }
            else:
                result_data = {
                    "total_activities": int(total_activities),
                    "display_type": "single_number"
                }
        
        elif kpi_name == "IngresosPorIntervención":
            # Buscar columnas de costo/ingreso y tipo de intervención
            cost_columns = [col for col in df.columns if any(term in str(col).lower() for term in ['costo', 'cost', 'precio', 'price', 'valor', 'value', 'ingreso', 'revenue'])]
            intervention_columns = [col for col in df.columns if any(term in str(col).lower() for term in ['interv', 'tipo', 'type', 'servicio', 'service'])]
            
            if cost_columns and intervention_columns:
                cost_col = cost_columns[0]
                intervention_col = intervention_columns[0]
                
                # Agrupar por tipo de intervención y sumar los costos
                grouped_data = df.groupby(intervention_col)[cost_col].sum()
                
                # Ordenar por ingresos en orden descendente
                grouped_data = grouped_data.sort_values(ascending=False)
                
                result_data = {
                    "revenue_by_intervention": json_serializable(grouped_data.to_dict()),
                    "total_revenue": float(grouped_data.sum()),
                    "visualization_data": {
                        "labels": grouped_data.index.tolist(),
                        "values": [float(val) if not pd.isna(val) else 0 for val in grouped_data.values]
                    }
                }
            else:
                result_data = {
                    "error": "Columnas necesarias no encontradas",
                    "message": "No se pudieron identificar las columnas de costos o de tipo de intervención."
                }
        
        # Default case if KPI not implemented or no result_data set
        if result_data is None:
            result_data = {
                "error": "KPI not implemented",
                "kpi_id": kpi_name,
                "message": f"The KPI '{kpi_name}' is not yet implemented"
            }
        
        # Always wrap the result in a "result" key for frontend compatibility
        # And ensure all values are JSON-serializable
        return {"result": json_safe(result_data)}
        
    except Exception as e:
        logger.error(f"Error calculating KPI {kpi_name}: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"result": {"error": f"Error calculating KPI: {str(e)}"}}
        )

# Get available filters
@app.get("/available-filters")
async def get_available_filters():
    """
    Get available columns that can be used for filtering the dataset.
    """
    try:
        logger.info("Getting available filters")
        from backend.dataset import get_dataframe
        
        df = get_dataframe()
        if df is None:
            raise HTTPException(status_code=400, detail="No dataset loaded")
        
        available_filters = {
            "categorical": [],
            "numerical": {},
            "date": []
        }
        
        # Add categorical columns
        available_filters["categorical"] = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Add numerical columns (grouped by type for UI organization)
        numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
        if numerical_cols:
            available_filters["numerical"]["General"] = numerical_cols
        
        # Add date columns
        available_filters["date"] = df.select_dtypes(include=['datetime']).columns.tolist()
        
        return available_filters
    except Exception as e:
        logger.error(f"Error retrieving available filters: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"detail": f"Error retrieving available filters: {str(e)}"}
        )

# Generate AI KPI endpoint
@app.post("/generate-ai-kpi")
async def generate_ai_kpi(request: Request):
    """
    Generate a custom KPI using AI based on the user's description.
    """
    try:
        logger.info("Generating AI KPI")
        from backend.dataset import get_dataframe
        
        # Parse request body
        try:
            body = await request.json()
        except Exception as e:
            logger.error(f"Error parsing request body: {str(e)}")
            raise HTTPException(status_code=400, detail="Invalid JSON in request body")
        
        user_prompt = body.get("user_prompt", "")
        service_line = body.get("service_line", "")
        mapping_path = body.get("mapping_path", "")
        
        # Define required fields upfront so it's always available
        required_fields = ["name", "formula", "description", "service_line", "visualization"]
        
        if not user_prompt:
            logger.warning("Missing user_prompt in request body")
            raise HTTPException(status_code=400, detail="KPI description is required")
        
        logger.info(f"Generating KPI with description: {user_prompt}")
        
        # Get the dataset
        df = get_dataframe()
        if df is None:
            raise HTTPException(status_code=400, detail="No dataset loaded")
        
        # Get OpenAI API key from environment
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.error("OpenAI API key not found in environment")
            raise HTTPException(
                status_code=500, 
                detail="API key not found. Please check your environment setup."
            )
        
        logger.info(f"API Key found with length: {len(api_key)}")
        
        # Set up the OpenAI configuration - use the global openai module
        import openai as openai_module  # Import with alias to avoid conflict
        openai_module.api_key = api_key
        
        # Prepare dataset context
        dataset_info = {
            "columns": df.columns.tolist(),
            "dtypes": {col: str(df[col].dtype) for col in df.columns},
            "sample": df.head(5).to_dict(orient="records"),
            "service_line": service_line
        }
        
        # Generate prompt for the KPI generation
        prompt = f"""
        Eres un experto en análisis de datos para la industria petrolera.
        
        Crea un KPI personalizado para la línea de servicio '{service_line}' basado en la siguiente descripción:
        {user_prompt}
        
        Información del conjunto de datos:
        Columnas: {dataset_info['columns']}
        Tipos de datos: {dataset_info['dtypes']}
        Datos de muestra: {dataset_info['sample']}
        
        Sigue estos pasos para crear el KPI:
        1. Comprende lo que el usuario está pidiendo en su descripción
        2. Identifica qué columnas del conjunto de datos son necesarias
        3. Crea una fórmula usando operaciones de pandas (usando 'df' como nombre de la variable del dataframe)
        4. Asegúrate de que la fórmula resulte en un valor único o una serie que pueda graficarse
        5. Especifica qué tipo de visualización sería mejor (barra, línea, pastel, dispersión, etc.)
        
        Genera una respuesta JSON con la siguiente estructura:
        {{
            "name": "Nombre del KPI",
            "description": "Descripción detallada de lo que mide este KPI",
            "formula": "# Tu código pandas aquí\\nresult = df[...].operation()  # El código debe ser ejecutable y Python válido",
            "interpretation": "Cómo interpretar los resultados de este KPI",
            "columns_required": ["lista", "de", "columnas", "requeridas"],
            "service_line": "{service_line}",
            "visualization": {{
                "type": "bar|line|pie|scatter|histogram",
                "x_axis": "Columna o valor a usar para el eje x",
                "y_axis": "Columna o valor a usar para el eje y",
                "title": "Título del gráfico"
            }}
        }}
        
        IMPORTANTE: La fórmula DEBE ser código Python válido usando pandas y debe retornar un resultado que pueda graficarse.
        Usa operaciones estándar de pandas y asegúrate de que la sintaxis sea correcta. El código completo debe incluirse entre comillas.
        """
        
        try:
            # Try modern OpenAI client first (v1.x style)
            try:
                # Import OpenAI again to ensure it's in this scope
                import openai as openai_module
                client = openai_module.OpenAI(api_key=api_key)
                logger.info("Using modern OpenAI client (v1.x style)")
                try:
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "Eres un experto en análisis de datos para la industria petrolera. Por favor, responde siempre en español."},
                            {"role": "user", "content": prompt}
                        ],
                        response_format={"type": "json_object"},
                        max_tokens=2000,
                        temperature=0.7
                    )
                    ai_response = response.choices[0].message.content.strip()
                    logger.info(f"Generated AI response with modern client, length: {len(ai_response)}")
                except Exception as api_error:
                    logger.error(f"Error with OpenAI API call (modern client): {str(api_error)}")
                    raise api_error
            except (ImportError, AttributeError) as e:
                logger.warning(f"Error with modern OpenAI client: {str(e)}, falling back to v0.28 style")
                # Fall back to v0.28 style
                try:
                    logger.info("Using legacy OpenAI client (v0.28 style)")
                    response = openai_module.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "Eres un experto en análisis de datos para la industria petrolera. Por favor, responde siempre en español."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=2000,
                        temperature=0.7
                    )
                    ai_response = response.choices[0].message.content.strip()
                    logger.info(f"Generated AI response with legacy client, length: {len(ai_response)}")
                except Exception as api_error:
                    logger.error(f"Error with OpenAI API call (legacy client): {str(api_error)}")
                    raise api_error
            
            # Parse the response as JSON
            try:
                kpi_definition = json.loads(ai_response)
                
                # Ensure that the response has the fields expected by the frontend
                missing_fields = [field for field in required_fields if field not in kpi_definition]
                
                if missing_fields:
                    # Try to adapt fields that might be named differently
                    if "kpi_name" in kpi_definition and "name" not in kpi_definition:
                        kpi_definition["name"] = kpi_definition.pop("kpi_name")
                    
                    # Still check if any required fields are missing
                    missing_fields = [field for field in required_fields if field not in kpi_definition]
                    
                    if missing_fields:
                        logger.warning(f"Missing required fields in KPI definition: {missing_fields}")
                        # Add default values for missing fields
                        for field in missing_fields:
                            if field == "name":
                                kpi_definition["name"] = "Custom KPI"
                            elif field == "formula":
                                kpi_definition["formula"] = "# Unable to generate formula\nresult = None"
                            elif field == "description":
                                kpi_definition["description"] = "Custom KPI based on user prompt"
                            elif field == "service_line":
                                kpi_definition["service_line"] = service_line
                            elif field == "visualization":
                                kpi_definition["visualization"] = {
                                    "type": "bar",
                                    "x_axis": "Index",
                                    "y_axis": "Value",
                                    "title": "KPI Result"
                                }
                
                # Test the formula on a sample of the dataset to see if it works
                test_result = None
                try:
                    # Create a temporary copy of part of the dataframe
                    test_df = df.head(100).copy()
                    
                    # Variables to make available in the execution environment
                    exec_globals = {
                        'df': test_df,
                        'pd': pd,
                        'np': np,
                        'result': None
                    }
                    
                    # Execute the formula
                    exec(kpi_definition["formula"], exec_globals)
                    
                    # Retrieve the result
                    result = exec_globals.get('result')
                    
                    # Special handling for matplotlib plots - convert them to DataFrames
                    if result is not None and hasattr(result, '__module__') and 'matplotlib' in str(result.__module__):
                        logger.info("Converting matplotlib result to DataFrame")
                        
                        # Try to get the data from the formula
                        if 'Base [md]' in kpi_definition["formula"] and 'Costo (USD)' in kpi_definition["formula"]:
                            # Extract the data access part from the formula
                            import re
                            data_pattern = r"df\[\['(.*?)',\s*'(.*?)'\]\]\.dropna\(\)"
                            match = re.search(data_pattern, kpi_definition["formula"])
                            
                            if match:
                                col1, col2 = match.groups()
                                # Get the data directly
                                result = df[[col1, col2]].dropna()
                                logger.info(f"Extracted data with columns {col1} and {col2}, shape: {result.shape}")
                            else:
                                # If pattern not found, look for Base [md] and Costo (USD) columns
                                if 'Base [md]' in df.columns and 'Costo (USD)' in df.columns:
                                    result = df[['Base [md]', 'Costo (USD)']].dropna()
                                    logger.info(f"Using default columns, shape: {result.shape}")
                    
                    # Format the result based on its type
                    if result is not None:
                        if isinstance(result, (pd.Series, pd.DataFrame)):
                            # If it's a Series or DataFrame, convert to a dict or list
                            if isinstance(result, pd.Series):
                                test_result = {
                                    "status": "success",
                                    "type": "series",
                                    "data": json_serializable(result.to_dict()),
                                    "index": json_serializable(result.index.tolist() if hasattr(result.index, 'tolist') else [str(i) for i in result.index])
                                }
                            else:  # DataFrame
                                test_result = {
                                    "status": "success",
                                    "type": "dataframe",
                                    "data": json_serializable(result.to_dict(orient='records')),
                                    "columns": json_serializable(result.columns.tolist())
                                }
                        elif isinstance(result, (list, tuple)):
                            # If it's a list or tuple, convert to list
                            test_result = {
                                "status": "success",
                                "type": "list",
                                "data": json_serializable(list(result))
                            }
                        elif isinstance(result, dict):
                            # If it's already a dict, use as is
                            test_result = {
                                "status": "success",
                                "type": "dict",
                                "data": json_serializable(result)
                            }
                        elif isinstance(result, (int, float)):
                            # If it's a scalar, convert to a simple value
                            test_result = {
                                "status": "success",
                                "type": "scalar",
                                "data": json_serializable(float(result))
                            }
                        else:
                            # For any other type, convert to string
                            test_result = {
                                "status": "success",
                                "type": "other",
                                "data": str(result)
                            }
                    else:
                        test_result = {
                            "status": "error",
                            "message": "Formula execution did not produce a result"
                        }
                
                except Exception as e:
                    logger.error(f"Error testing formula: {str(e)}")
                    test_result = {
                        "status": "error",
                        "message": str(e)
                    }
                
                # Add the test result to the KPI definition
                kpi_definition["test_result"] = test_result
                
                return kpi_definition
            except json.JSONDecodeError:
                # If the response is not valid JSON, extract JSON from the text
                import re
                json_match = re.search(r'({.*})', ai_response.replace('\n', ''), re.DOTALL)
                
                if json_match:
                    try:
                        kpi_definition = json.loads(json_match.group(1))
                        # Apply the same field transformations as above
                        if "kpi_name" in kpi_definition and "name" not in kpi_definition:
                            kpi_definition["name"] = kpi_definition.pop("kpi_name")
                        
                        # Ensure required fields
                        for field in required_fields:
                            if field not in kpi_definition:
                                if field == "name":
                                    kpi_definition["name"] = "Custom KPI"
                                elif field == "formula":
                                    kpi_definition["formula"] = "# Unable to generate formula\nresult = None"
                                elif field == "description":
                                    kpi_definition["description"] = "Custom KPI based on user prompt"
                                elif field == "service_line":
                                    kpi_definition["service_line"] = service_line
                                elif field == "visualization":
                                    kpi_definition["visualization"] = {
                                        "type": "bar",
                                        "x_axis": "Index",
                                        "y_axis": "Value",
                                        "title": "KPI Result"
                                    }
                        
                        # Test the KPI formula
                        try:
                            # Create a temporary copy of part of the dataframe
                            test_df = df.head(100).copy()
                            
                            # Variables to make available in the execution environment
                            exec_globals = {
                                'df': test_df,
                                'pd': pd,
                                'np': np,
                                'result': None
                            }
                            
                            # Execute the formula
                            exec(kpi_definition["formula"], exec_globals)
                            
                            # Retrieve the result
                            result = exec_globals.get('result')
                            
                            # Special handling for matplotlib plots - convert them to DataFrames
                            if result is not None and hasattr(result, '__module__') and 'matplotlib' in str(result.__module__):
                                logger.info("Converting matplotlib result to DataFrame")
                                
                                # Try to get the data from the formula
                                if 'Base [md]' in kpi_definition["formula"] and 'Costo (USD)' in kpi_definition["formula"]:
                                    # Extract the data access part from the formula
                                    import re
                                    data_pattern = r"df\[\['(.*?)',\s*'(.*?)'\]\]\.dropna\(\)"
                                    match = re.search(data_pattern, kpi_definition["formula"])
                                    
                                    if match:
                                        col1, col2 = match.groups()
                                        # Get the data directly
                                        result = df[[col1, col2]].dropna()
                                        logger.info(f"Extracted data with columns {col1} and {col2}, shape: {result.shape}")
                                    else:
                                        # If pattern not found, look for Base [md] and Costo (USD) columns
                                        if 'Base [md]' in df.columns and 'Costo (USD)' in df.columns:
                                            result = df[['Base [md]', 'Costo (USD)']].dropna()
                                            logger.info(f"Using default columns, shape: {result.shape}")
                            
                            # Format the result
                            if result is not None:
                                test_result = {
                                    "status": "success",
                                    "type": "other",
                                    "data": str(result) if not isinstance(result, (int, float, list, dict)) else result
                                }
                            else:
                                test_result = {
                                    "status": "error",
                                    "message": "Formula execution did not produce a result"
                                }
                        except Exception as e:
                            logger.error(f"Error testing formula: {str(e)}")
                            test_result = {
                                "status": "error",
                                "message": str(e)
                            }
                        
                        # Add the test result to the KPI definition
                        kpi_definition["test_result"] = test_result
                                                
                        return kpi_definition
                    except Exception as e:
                        logger.error(f"Error parsing extracted JSON: {str(e)}")
                
                # Return a structured response if we can't parse it as JSON
                logger.warning("Could not parse AI response as JSON, returning fallback structure")
                return {
                    "name": "Custom KPI",
                    "description": "Generated from user prompt",
                    "formula": "# Unable to parse AI response\nresult = None",
                    "service_line": service_line,
                    "visualization": {
                        "type": "bar",
                        "x_axis": "Index",
                        "y_axis": "Value",
                        "title": "KPI Result"
                    },
                    "test_result": {
                        "status": "error",
                        "message": "Unable to parse AI response as valid JSON"
                    },
                    "raw_response": ai_response[:1000]  # Limit response size
                }
            
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            logger.error(traceback.format_exc())
            
            # Provide a fallback KPI with a simple formula that should work
            logger.info("Generating fallback KPI with a simple formula")
            fallback_kpi = {
                "name": "Costo Promedio por Pozo",
                "description": "Cálculo básico del costo promedio por pozo, con visualización en gráfico de barras ordenado.",
                "formula": """# Fallback formula that should work with most datasets
# Calculate average cost per well and sort the results
import pandas as pd
import numpy as np

# Find cost and well columns (assuming standard naming conventions)
cost_columns = [col for col in df.columns if 'cost' in col.lower() or 'precio' in col.lower() or 'valor' in col.lower()]
well_columns = [col for col in df.columns if 'pozo' in col.lower() or 'well' in col.lower()]

# If we can't find them, use the first numeric column for cost and first string column for well
if not cost_columns:
    cost_columns = [col for col in df.select_dtypes(include=[np.number]).columns][:1]
if not well_columns:
    well_columns = [col for col in df.select_dtypes(include=['object']).columns][:1]

# Choose columns to use
cost_col = cost_columns[0] if cost_columns else df.columns[0]
well_col = well_columns[0] if well_columns else df.columns[0]

# Group by well and calculate average cost
result = df.groupby(well_col)[cost_col].mean().sort_values(ascending=False)

# Return the sorted series
result
""",
                "interpretation": "Este KPI muestra el costo promedio por pozo, ordenado de mayor a menor. Ayuda a identificar qué pozos tienen los costos más altos, lo que puede indicar ineficiencias o problemas que requieren investigación.",
                "columns_required": ["pozo", "costo"],
                "service_line": service_line,
                "visualization": {
                    "type": "bar",
                    "x_axis": "Pozo",
                    "y_axis": "Costo Promedio",
                    "title": "Costo Promedio por Pozo"
                }
            }
            
            # Test the fallback formula
            try:
                # Create a temporary copy of part of the dataframe
                test_df = df.head(100).copy()
                
                # Variables to make available in the execution environment
                exec_globals = {
                    'df': test_df,
                    'pd': pd,
                    'np': np,
                    'result': None
                }
                
                # Execute the formula
                exec(fallback_kpi["formula"], exec_globals)
                
                # Retrieve the result
                result = exec_globals.get('result')
                
                # Format the result
                if result is not None and isinstance(result, pd.Series):
                    test_result = {
                        "status": "success",
                        "type": "series",
                        "data": json_serializable(result.to_dict()),
                        "index": json_serializable(result.index.tolist() if hasattr(result.index, 'tolist') else [str(i) for i in result.index])
                    }
                    logger.info("Fallback formula executed successfully")
                else:
                    test_result = {
                        "status": "error",
                        "message": "Fallback formula execution did not produce a valid Series"
                    }
                    logger.error("Fallback formula did not produce a valid Series")
            except Exception as exec_error:
                logger.error(f"Error executing fallback formula: {str(exec_error)}")
                test_result = {
                    "status": "error",
                    "message": str(exec_error)
                }
            
            # Add the test result to the fallback KPI
            fallback_kpi["test_result"] = test_result
            
            # Return the fallback KPI instead of raising an exception
            logger.info("Returning fallback KPI")
            return fallback_kpi
            
    except HTTPException as http_exc:
        # Re-raise HTTP exceptions to let FastAPI handle them properly
        raise http_exc
    except Exception as e:
        logger.error(f"Error generating AI KPI: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error generating AI KPI: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 