from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import json
import importlib
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import tempfile
import openai
import numpy as np
from dotenv import load_dotenv

# Import your classes from the notebook
from rapidfuzz import process, fuzz
from ai_integration.chatgpt_kpi_assistant import ai_fallback_column_mapping
from ai_integration.chatgpt_kpi_assistant import ai_suggest_column_category
from ai_integration.chatgpt_kpi_assistant import generate_ai_kpi

# Cargar variables de entorno del archivo .env
load_dotenv()

# Obtener la clave de API de OpenAI de las variables de entorno
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Configurar el cliente de OpenAI con la clave
openai.api_key = OPENAI_API_KEY

# Añadir validación para verificar si la clave está disponible
if not OPENAI_API_KEY:
    print("⚠️ ADVERTENCIA: No se ha configurado OPENAI_API_KEY en el archivo .env")
    print("Las funciones de IA no estarán disponibles. Por favor configura la clave en el archivo .env.")

# DataHandler, FilterHandler, and KPIEngine classes
class DataHandler:
    def __init__(self, source_type, source, mapping_path):
        self.source_type = source_type
        self.source = source
        self.mapping_path = mapping_path
        
        # Cargar mapeo de columnas desde archivo JSON
        try:
            with open(mapping_path, "r") as f:
                self.mapping = json.load(f)
        except Exception as e:
            self.mapping = {}
            print(f"Error al cargar archivo de mapeo: {str(e)}")
    
    def load_data(self):
        """Cargar datos desde fuente especificada"""
        try:
            if self.source_type.lower() == "excel":
                # Verificar si el archivo existe
                if not os.path.exists(self.source):
                    return f"Archivo no encontrado: {self.source}"
                
                # Intentar cargar todas las hojas
                try:
                    # Primero intentar cargar la primera hoja
                    df = pd.read_excel(self.source)
                    print(f"Columnas cargadas: {df.columns.tolist()}")
                    return df
                except Exception as excel_error:
                    # Si falla, intentar con parámetros específicos
                    print(f"Error al cargar Excel normalmente: {str(excel_error)}")
                    try:
                        # Intentar con engine='openpyxl'
                        df = pd.read_excel(self.source, engine='openpyxl')
                        print(f"Cargado con openpyxl. Columnas: {df.columns.tolist()}")
                        return df
                    except Exception as openpyxl_error:
                        print(f"Error con openpyxl: {str(openpyxl_error)}")
                        try:
                            # Última opción: especificar sheet_name='Sheet1'
                            df = pd.read_excel(self.source, sheet_name=0)
                            print(f"Cargado con sheet_name=0. Columnas: {df.columns.tolist()}")
                            return df
                        except Exception as sheet_error:
                            return f"Error al cargar Excel: {str(sheet_error)}"
            elif self.source_type.lower() == "csv":
                if not os.path.exists(self.source):
                    return f"Archivo no encontrado: {self.source}"
                return pd.read_csv(self.source)
            elif self.source_type.lower() == "json":
                if not os.path.exists(self.source):
                    return f"Archivo no encontrado: {self.source}"
                with open(self.source, "r") as f:
                    data = json.load(f)
                return pd.DataFrame(data)
            elif self.source_type.lower() == "db":
                # Implementar conexión a base de datos (ejemplo con SQLAlchemy)
                # from sqlalchemy import create_engine
                # engine = create_engine(self.source)
                # return pd.read_sql(self.query, engine)
                return f"Conexiones a bases de datos aún no implementadas"
            elif self.source_type.lower() == "api":
                # Implementar obtención de datos de API
                # import requests
                # response = requests.get(self.source)
                # data = response.json()
                # return pd.DataFrame(data)
                return f"Fuentes de datos API aún no implementadas"
            else:
                return f"Tipo de fuente no soportado: {self.source_type}"
        except Exception as e:
            return f"Error al cargar datos: {str(e)}"

class FilterHandler:
    def __init__(self, df, mapping_path):
        self.df = df
        
        # Load column mapping from JSON file
        try:
            with open(mapping_path, "r") as f:
                self.mapping = json.load(f)
        except Exception as e:
            self.mapping = {}
            print(f"Error loading mapping file: {str(e)}")
            
        # Initialize column types
        self.categorical_cols = []
        self.numerical_cols = {}
        self.date_cols = []
        
        self._initialize_column_types()
    
    def _initialize_column_types(self):
        """Identify column types from mapping and data"""
        # From mapping
        if self.mapping:
            for col_type, cols in self.mapping.get("columns", {}).items():
                if col_type == "categorical":
                    self.categorical_cols.extend(cols)
                elif col_type in ["numerical", "metric"]:
                    self.numerical_cols[col_type] = cols
                elif col_type == "date":
                    self.date_cols.extend(cols)
        
        # Auto-detect from dataframe
        for col in self.df.columns:
            if col not in self.categorical_cols and col not in self.date_cols and not any(col in cols for cols in self.numerical_cols.values()):
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    if "metric" not in self.numerical_cols:
                        self.numerical_cols["metric"] = []
                    self.numerical_cols["metric"].append(col)
                elif pd.api.types.is_datetime64_dtype(self.df[col]):
                    self.date_cols.append(col)
                else:
                    self.categorical_cols.append(col)
    
    def get_available_filters(self):
        """Return available filter columns by type"""
        return {
            "categorical": self.categorical_cols,
            "numerical": self.numerical_cols,
            "date": self.date_cols
        }
    
    def apply_filters(self, filters):
        """Apply filters to dataframe"""
        filtered_df = self.df.copy()
        
        # Apply categorical filters
        for col, value in filters.get("categorical", {}).items():
            if col in filtered_df.columns:
                filtered_df = filtered_df[filtered_df[col] == value]
        
        # Apply numerical filters
        for col, range_vals in filters.get("numerical", {}).items():
            if col in filtered_df.columns and len(range_vals) == 2:
                filtered_df = filtered_df[(filtered_df[col] >= range_vals[0]) & 
                                          (filtered_df[col] <= range_vals[1])]
        
        # Apply date filters
        for col, date_range in filters.get("date", {}).items():
            if col in filtered_df.columns and len(date_range) == 2:
                # Convert string dates to datetime
                start_date = pd.to_datetime(date_range[0])
                end_date = pd.to_datetime(date_range[1])
                
                # Convert column to datetime if it's not already
                if not pd.api.types.is_datetime64_dtype(filtered_df[col]):
                    filtered_df[col] = pd.to_datetime(filtered_df[col])
                
                filtered_df = filtered_df[(filtered_df[col] >= start_date) & 
                                          (filtered_df[col] <= end_date)]
        
        return filtered_df

class BaseKPI:
    def __init__(self, name, description):
        self.name = name
        self.description = description
    
    def calculate(self, df):
        """Base method to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement calculate method")

class KPIRegistry:
    def __init__(self):
        self.kpis = {}
    
    def register(self, name, kpi_class):
        self.kpis[name] = kpi_class
    
    def get_kpi(self, name):
        return self.kpis.get(name)

class KPIEngine:
    def __init__(self, df, service_line, mapping_path):
        self.df = df
        self.service_line = service_line
        self.mapping_path = mapping_path
        
        # Load column mapping
        try:
            with open(mapping_path, "r") as f:
                self.mapping = json.load(f)
        except Exception as e:
            self.mapping = {}
            print(f"Error loading mapping file: {str(e)}")
        
        # Initialize KPI registry
        self.kpi_registry = KPIRegistry()
        
        # Register built-in KPIs based on service line
        self._register_built_in_kpis()
    
    def _register_built_in_kpis(self):
        """Register built-in KPIs based on service line"""
        if self.service_line.lower() == "stimulation":
            self.kpi_registry.register("TotalCost", self.calculate_total_cost)
            self.kpi_registry.register("CostPerLease", self.calculate_cost_per_lease)
            self.kpi_registry.register("ChemicalUsage", self.calculate_chemical_usage)
            self.kpi_registry.register("OperationalEfficiency", self.calculate_operational_efficiency)
        elif self.service_line.lower() == "drilling":
            self.kpi_registry.register("AverageDrillingTime", self.calculate_avg_drilling_time)
            self.kpi_registry.register("FootageDrilled", self.calculate_footage_drilled)
        # Add more service lines as needed
    
    def calculate_kpi(self, kpi_name):
        """Calculate a KPI by name"""
        kpi_func = self.kpi_registry.get_kpi(kpi_name)
        if kpi_func:
            return kpi_func(self.df)
        else:
            raise ValueError(f"KPI not found: {kpi_name}")
    
    def create_dynamic_kpi(self, formula_str):
        """Create a dynamic KPI from a formula string"""
        # Example implementation - would need to be more sophisticated in production
        def dynamic_kpi(df):
            # Create a safe local environment with dataframe
            local_vars = {"df": df, "pd": pd, "np": np}
            
            # Execute the formula in the safe environment
            result = eval(formula_str, {"__builtins__": {}}, local_vars)
            return result
        
        return dynamic_kpi
    
    # Built-in KPI implementations
    def calculate_total_cost(self, df):
        """Calculate total cost from dataframe"""
        cost_col = self.mapping.get("metrics", {}).get("cost", "Cost")
        if cost_col in df.columns:
            return df[cost_col].sum()
        return 0
    
    def calculate_cost_per_lease(self, df):
        """Calculate cost per lease from dataframe"""
        cost_col = self.mapping.get("metrics", {}).get("cost", "Cost") 
        lease_col = self.mapping.get("dimensions", {}).get("lease", "Lease")
        
        if cost_col in df.columns and lease_col in df.columns:
            return df.groupby(lease_col)[cost_col].sum().to_dict()
        return {}
    
    def calculate_chemical_usage(self, df):
        """Calculate chemical usage by lease"""
        lease_col = self.mapping.get("dimensions", {}).get("lease", "Lease")
        chemical_cols = self.mapping.get("metrics", {}).get("chemicals", [])
        
        if not chemical_cols:
            # Auto-detect chemical columns (example logic)
            chemical_cols = [col for col in df.columns if "chemical" in col.lower()]
        
        if lease_col in df.columns and chemical_cols:
            result = {}
            for chemical in chemical_cols:
                if chemical in df.columns:
                    result[chemical] = df.groupby(lease_col)[chemical].sum().to_dict()
            return result
        return {}
    
    def calculate_operational_efficiency(self, df):
        """Calculate operational efficiency metrics"""
        time_col = self.mapping.get("metrics", {}).get("time", "Time")
        volume_col = self.mapping.get("metrics", {}).get("volume", "Volume")
        
        if time_col in df.columns and volume_col in df.columns:
            # Example: Volume processed per unit time
            return df[volume_col].sum() / df[time_col].sum()
        return 0
    
    def calculate_avg_drilling_time(self, df):
        """Calculate average drilling time"""
        time_col = self.mapping.get("metrics", {}).get("drilling_time", "DrillingTime")
        
        if time_col in df.columns:
            return df[time_col].mean()
        return 0
    
    def calculate_footage_drilled(self, df):
        """Calculate total footage drilled"""
        footage_col = self.mapping.get("metrics", {}).get("footage", "Footage")
        
        if footage_col in df.columns:
            return df[footage_col].sum()
        return 0

# Create API models
class DataSource(BaseModel):
    source_type: str
    source: str
    mapping_path: str
    service_line: str

class FilterRequest(BaseModel):
    categorical: Optional[Dict[str, str]] = {}
    numerical: Optional[Dict[str, List[float]]] = {}
    date: Optional[Dict[str, List[str]]] = {}

class AIKPIRequest(BaseModel):
    user_prompt: str
    service_line: str
    mapping_path: str

# Create FastAPI app
app = FastAPI(title="KPI Data Analysis API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to store state
_data_handler = None
_filter_handler = None
_kpi_engine = None
_df = None

@app.post("/load-data")
async def load_data(data_source: DataSource):
    """Load data from specified source"""
    global _data_handler, _filter_handler, _kpi_engine, _df
    
    try:
        _data_handler = DataHandler(
            source_type=data_source.source_type,
            source=data_source.source,
            mapping_path=data_source.mapping_path
        )
        
        print(f"Attempting to load data from: {data_source.source}")
        
        # Intentar cargar los datos con múltiples métodos
        if data_source.source_type.lower() == 'excel':
            try:
                # Primero intentar cargar normalmente
                _df = pd.read_excel(data_source.source)
                print(f"Successfully loaded Excel file with pandas read_excel, shape: {_df.shape}")
            except Exception as e1:
                print(f"Standard Excel loading failed: {str(e1)}")
                try:
                    # Intentar con engine='openpyxl'
                    _df = pd.read_excel(data_source.source, engine='openpyxl')
                    print(f"Successfully loaded with openpyxl engine, shape: {_df.shape}")
                except Exception as e2:
                    print(f"Loading with openpyxl failed: {str(e2)}")
                    try:
                        # Intentar listar todas las hojas y cargar la primera
                        xls = pd.ExcelFile(data_source.source, engine='openpyxl')
                        print(f"Available sheets: {xls.sheet_names}")
                        _df = pd.read_excel(xls, sheet_name=0)
                        print(f"Successfully loaded first sheet, shape: {_df.shape}")
                    except Exception as e3:
                        print(f"All Excel loading methods failed: {str(e3)}")
                        raise HTTPException(status_code=500, detail=f"Could not load Excel file. Error: {str(e3)}")
        else:
            _df = _data_handler.load_data()
            
        if isinstance(_df, str):  # Si devuelve un mensaje de error
            raise HTTPException(status_code=400, detail=_df)
        
        if _df.empty:
            raise HTTPException(status_code=400, detail="Loaded DataFrame is empty")
            
        print(f"Data loaded successfully. Shape: {_df.shape}")
        print(f"Columns: {_df.columns.tolist()}")
        
        # Inicializar manejador de filtros
        _filter_handler = FilterHandler(_df, data_source.mapping_path)
        
        # Inicializar motor de KPI
        _kpi_engine = KPIEngine(_df, data_source.service_line, data_source.mapping_path)
        
        # Agregar KPIs adicionales para este conjunto de datos
        register_additional_kpis()
        
        return {
            "success": True,
            "rows": len(_df),
            "columns": len(_df.columns),
            "column_names": _df.columns.tolist()
        }
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in load-data endpoint: {str(e)}\n{error_details}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/available-filters")
async def get_available_filters():
    """Get available filters from the loaded data"""
    global _filter_handler
    
    if _filter_handler is None:
        raise HTTPException(status_code=400, detail="Data not loaded. Call /load-data first")
    
    try:
        available_filters = _filter_handler.get_available_filters()
        return available_filters
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/apply-filters")
async def apply_filters(filters: FilterRequest):
    """Apply filters to the loaded data"""
    global _filter_handler, _kpi_engine, _df
    
    if _filter_handler is None:
        raise HTTPException(status_code=400, detail="Data not loaded. Call /load-data first")
    
    try:
        filter_dict = filters.dict(exclude_unset=True)
        filtered_df = _filter_handler.apply_filters(filter_dict)
        
        # Update KPI engine with filtered data
        _kpi_engine.df = filtered_df
        
        # Return basic info about the filtered data
        return {
            "success": True,
            "rows_before": len(_df),
            "rows_after": len(filtered_df),
            "filter_summary": filter_dict
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/available-kpis")
async def get_available_kpis():
    """Get all available KPIs"""
    global _kpi_engine
    
    if _kpi_engine is None:
        raise HTTPException(status_code=400, detail="Data not loaded. Call /load-data first")
    
    try:
        return {
            "kpis": list(_kpi_engine.kpi_registry.kpis.keys())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/calculate-kpi/{kpi_name}")
async def calculate_kpi(kpi_name: str):
    """Calculate a specific KPI"""
    global _kpi_engine
    
    if _kpi_engine is None:
        raise HTTPException(status_code=400, detail="Data not loaded. Call /load-data first")
    
    try:
        result = _kpi_engine.calculate_kpi(kpi_name)
        return {"kpi_name": kpi_name, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-ai-kpi")
async def create_ai_kpi(request: AIKPIRequest):
    """Generate a new KPI using AI"""
    try:
        ai_kpi = generate_ai_kpi(
            request.user_prompt,
            request.service_line,
            request.mapping_path
        )
        
        # Test the KPI if data is loaded
        if _kpi_engine:
            try:
                result = _kpi_engine.create_dynamic_kpi(ai_kpi["formula"])(_kpi_engine.df)
                ai_kpi["test_result"] = result
            except Exception as e:
                ai_kpi["test_error"] = str(e)
                
        return ai_kpi
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/raw-data")
async def get_raw_data(limit: int = 100):
    """Get a sample of the raw data"""
    global _kpi_engine
    
    if _kpi_engine is None or _kpi_engine.df is None:
        raise HTTPException(status_code=400, detail="Data not loaded. Call /load-data first")
    
    try:
        # Imprimir información sobre el DataFrame para depuración
        print(f"DataFrame info: {_kpi_engine.df.info()}")
        print(f"DataFrame columns: {_kpi_engine.df.columns.tolist()}")
        print(f"DataFrame shape: {_kpi_engine.df.shape}")
        print(f"DataFrame types: {_kpi_engine.df.dtypes}")
        
        # Verificar si hay valores no serializables
        problematic_columns = []
        for col in _kpi_engine.df.columns:
            try:
                # Intenta serializar a JSON para ver si hay problemas
                json.dumps(_kpi_engine.df[col].head(5).tolist())
            except:
                problematic_columns.append(col)
                print(f"Problematic column detected: {col}, type: {_kpi_engine.df[col].dtype}")
        
        if problematic_columns:
            print(f"Columns with serialization issues: {problematic_columns}")
        
        # Crear una copia segura del DataFrame para conversión
        df_sample = _kpi_engine.df.head(limit).reset_index(drop=True).copy()
        
        # Convertir todas las columnas a tipos serializables
        for col in df_sample.columns:
            # Manejar fechas
            if pd.api.types.is_datetime64_any_dtype(df_sample[col]):
                df_sample[col] = df_sample[col].astype(str)
            # Manejar objetos complejos
            elif not pd.api.types.is_numeric_dtype(df_sample[col]) and not pd.api.types.is_string_dtype(df_sample[col]):
                df_sample[col] = df_sample[col].astype(str)
            
            # Reemplazar NaN/None con valores que JSON pueda manejar
            if df_sample[col].isna().any():
                df_sample[col] = df_sample[col].fillna("null")
        
        # Convertir a diccionario y luego a string JSON y de vuelta a diccionario para asegurar serialización
        try:
            # Convertir a registros
            records = df_sample.to_dict(orient="records")
            
            # Intentar serializar a JSON y volver a diccionario para validar
            json_str = json.dumps(records)
            validated_records = json.loads(json_str)
            
            return {"data": validated_records}
        except Exception as json_error:
            print(f"JSON serialization error: {str(json_error)}")
            # Método alternativo: convertir manualmente
            safe_records = []
            for _, row in df_sample.iterrows():
                safe_row = {}
                for col in df_sample.columns:
                    try:
                        value = row[col]
                        # Convertir cualquier valor problemático a string
                        if not isinstance(value, (int, float, str, bool, type(None))):
                            value = str(value)
                        # Convertir NaN a None
                        if pd.isna(value):
                            value = None
                        safe_row[col] = value
                    except:
                        safe_row[col] = str(row[col])
                safe_records.append(safe_row)
            return {"data": safe_records}
    except Exception as e:
        # Mostrar el error completo con traza
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in raw-data endpoint: {str(e)}\n{error_details}")
        raise HTTPException(status_code=500, detail=f"Error processing data: {str(e)}. Check server logs for details.")

# Agregar un nuevo endpoint específico para obtener valores únicos
@app.get("/unique-values/{column}")
async def get_unique_values(column: str):
    """Get unique values for a specific column"""
    global _kpi_engine
    
    if _kpi_engine is None or _kpi_engine.df is None:
        raise HTTPException(status_code=400, detail="Data not loaded. Call /load-data first")
    
    try:
        if column not in _kpi_engine.df.columns:
            return {"values": []}
        
        # Obtener valores únicos sin NaN y convertir a tipos de datos simples
        unique_values = _kpi_engine.df[column].dropna().unique()
        
        # Convertir valores a tipos serializables
        result = []
        for val in unique_values:
            if isinstance(val, (int, float, bool, str)):
                result.append(val)
            else:
                result.append(str(val))
        
        return {"values": result}
    except Exception as e:
        # Agregar más detalles al error
        import traceback
        error_details = traceback.format_exc()
        print(f"Error en get_unique_values: {str(e)}\n{error_details}")
        raise HTTPException(status_code=500, detail=f"Error getting unique values: {str(e)}")

@app.post("/upload-file")
async def upload_file(file: UploadFile = File(...)):
    """Upload a file and save it temporarily on the server"""
    try:
        # Create temporary directory if it doesn't exist
        temp_dir = os.path.join(tempfile.gettempdir(), "kpi_dashboard")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Save the file
        file_path = os.path.join(temp_dir, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        return {"filename": file.filename, "path": file_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

@app.post("/chat-query")
async def chat_query(query: dict):
    """Process a chat query about the data"""
    global _kpi_engine
    
    if _kpi_engine is None or _kpi_engine.df is None:
        raise HTTPException(status_code=400, detail="Data not loaded. Call /load-data first")
    
    try:
        user_query = query.get("query", "")
        if not user_query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Create a safe copy of the DataFrame for analysis
        safe_df = _kpi_engine.df.copy()
        
        # Convert problematic columns to safe types
        for col in safe_df.columns:
            # Convert string 'null' to actual None
            if safe_df[col].dtype == 'object':
                safe_df[col] = safe_df[col].replace('null', None)
            
            # Ensure numeric columns have compatible types
            if safe_df[col].dtype.name.startswith(('float', 'int')):
                safe_df[col] = pd.to_numeric(safe_df[col], errors='coerce')
        
        query_lower = user_query.lower()
        
        # Specialized handling for chemical product queries
        if ("químic" in query_lower or "chemical" in query_lower or "product" in query_lower) and \
           ("frecuent" in query_lower or "más" in query_lower or "uso" in query_lower):
            
            # Find chemical-related columns
            chemical_cols = []
            for col in safe_df.columns:
                if any(term in col.lower() for term in ["químic", "chemical", "fluid", "fluido", "product", "producto"]):
                    chemical_cols.append(col)
                # Also look for common chemical names
                elif any(term in col.lower() for term in ["ácido", "acid", "foam", "espuma", "polímero", "polymer",
                                                         "surfactant", "surfactante", "aditivo", "additive"]):
                    chemical_cols.append(col)
            
            if chemical_cols:
                response_text = f"## Análisis de Productos Químicos\n\n"
                response_text += "### Productos químicos más utilizados en los datos:\n\n"
                
                # For each chemical column, analyze usage
                chemical_usage_data = []
                
                for col in chemical_cols:
                    if safe_df[col].dtype.name.startswith(('float', 'int')):
                        # For numeric columns (e.g. volumes), count where value > 0
                        non_zero = safe_df[safe_df[col] > 0][col].count()
                        avg_value = safe_df[safe_df[col] > 0][col].mean()
                        if non_zero > 0:
                            percentage = (non_zero/len(safe_df))*100
                            chemical_usage_data.append({
                                "name": col,
                                "count": int(non_zero),
                                "percentage": percentage,
                                "avg_value": avg_value,
                                "is_numeric": True
                            })
                    else:
                        # For categorical columns, get value counts
                        value_counts = safe_df[col].dropna().value_counts()
                        if not value_counts.empty:
                            for value, count in value_counts.items():
                                if value and str(value).lower() not in ["no", "none", "na", "n/a"]:
                                    percentage = (count/len(safe_df))*100
                                    chemical_usage_data.append({
                                        "name": f"{col}: {value}",
                                        "count": int(count),
                                        "percentage": percentage,
                                        "is_numeric": False
                                    })
                
                # Sort by frequency (most used first)
                chemical_usage_data.sort(key=lambda x: x["count"], reverse=True)
                
                if chemical_usage_data:
                    for item in chemical_usage_data[:10]:  # Show top 10
                        if item["is_numeric"]:
                            response_text += f"- **{item['name']}**: Usado en {item['count']} registros ({item['percentage']:.2f}% del total)"
                            response_text += f", con un promedio de {item['avg_value']:.2f} por registro donde se usa.\n"
                        else:
                            response_text += f"- **{item['name']}**: Presente en {item['count']} registros ({item['percentage']:.2f}% del total)\n"
                else:
                    response_text += "No se encontraron datos específicos sobre el uso de productos químicos en los registros analizados.\n\n"
                    response_text += "Las columnas analizadas fueron: " + ", ".join(chemical_cols) + "\n"
                
                return {"response": response_text}
            else:
                # No chemical columns found
                response_text = "## Análisis de Productos Químicos\n\n"
                response_text += "No se encontraron columnas específicas relacionadas con productos químicos en los datos.\n\n"
                response_text += "Columnas disponibles en el conjunto de datos:\n\n"
                
                # List first 10 columns to help user identify relevant ones
                for col in list(safe_df.columns)[:10]:
                    response_text += f"- {col}\n"
                
                if len(safe_df.columns) > 10:
                    response_text += f"- ... y {len(safe_df.columns) - 10} columnas más\n\n"
                
                response_text += "\nPuedes especificar qué columna contiene información sobre productos químicos para un mejor análisis."
                
                return {"response": response_text}
        
        # Try to use OpenAI for other queries if available
        try:
            if OPENAI_API_KEY:
                # Get column info and data sample to help OpenAI understand the dataset
                column_info = {col: str(safe_df[col].dtype) for col in safe_df.columns}
                data_sample = safe_df.head(5).to_dict(orient="records")
                
                prompt = f"""
                Analiza los siguientes datos y responde a la consulta del usuario de manera específica.
                
                Consulta del usuario: "{user_query}"
                
                Información sobre las columnas del dataset:
                {json.dumps(column_info, indent=2)}
                
                Muestra de los datos (primeras 5 filas):
                {json.dumps(data_sample, indent=2, default=str)}
                
                Proporciona una respuesta concisa, específica y relevante a la consulta del usuario.
                No incluyas un resumen general de todo el dataset a menos que sea relevante para la consulta.
                La respuesta debe estar en formato Markdown.
                """
                
                response = openai.ChatCompletion.create(
                    model="gpt-4",  # or "gpt-4" depending on your subscription
                    messages=[
                        {"role": "system", "content": "Eres un asistente de análisis de datos que proporciona respuestas precisas y específicas a consultas sobre datos petroleros."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1000,
                    temperature=0.5
                )
                
                return {"response": response.choices[0].message["content"]}
        except Exception as ai_error:
            print(f"Error using AI for chat query: {str(ai_error)}")
        
        # Continue with the rest of the function for other query types
        # ... existing code for other query types ...

        # Find relevant columns based on query
        relevant_columns = []
        query_topics = {
            "químic": ["químic", "chemical", "product", "fluido", "fluid"],
            "cost": ["costo", "cost", "precio", "price", "valor", "value"],
            "pozo": ["pozo", "well", "campo", "field"],
            "producción": ["producción", "production", "volumen", "volume"],
            "fecha": ["fecha", "date", "tiempo", "time", "period"]
        }
        
        for topic, keywords in query_topics.items():
            if any(keyword in query_lower for keyword in keywords):
                for col in safe_df.columns:
                    if any(keyword in col.lower() for keyword in keywords):
                        relevant_columns.append(col)
        
        # Generate a more specific response based on the query
        response_text = f"## Respuesta a: '{user_query}'\n\n"
        
        if relevant_columns:
            response_text += f"He encontrado columnas relevantes para tu consulta: {', '.join(relevant_columns)}\n\n"
            
            # Add specific information about relevant columns
            for col in relevant_columns[:3]:  # Limit to first 3 for brevity
                response_text += f"### {col}\n"
                
                if safe_df[col].dtype.name.startswith(('float', 'int')):
                    # Numeric column analysis
                    response_text += f"- **Promedio**: {safe_df[col].mean():.2f}\n"
                    response_text += f"- **Mínimo**: {safe_df[col].min():.2f}\n"
                    response_text += f"- **Máximo**: {safe_df[col].max():.2f}\n"
                    response_text += f"- **Mediana**: {safe_df[col].median():.2f}\n"
                else:
                    # Categorical column analysis
                    counts = safe_df[col].dropna().value_counts().head(5)
                    for value, count in counts.items():
                        percentage = round((count/len(safe_df))*100, 2)
                        response_text += f"- **{value}**: {count} registros ({percentage}%)\n"
                
                response_text += "\n"
        else:
            # Fallback to general data summary but with a more specific message
            response_text += f"No encontré columnas específicas relacionadas con tu consulta. Por favor, intenta ser más específico o usa otros términos.\n\n"
            response_text += f"Puedes preguntar sobre:\n- Productos químicos utilizados\n- Costos y precios\n- Pozos y campos\n- Producción y volumen\n- Fechas y periodos\n\n"
            response_text += f"El conjunto de datos tiene {len(safe_df)} registros y {len(safe_df.columns)} columnas. Algunas columnas importantes incluyen:\n\n"
            
            # Show first 5 columns as examples
            for col in list(safe_df.columns)[:5]:
                response_text += f"- {col}\n"
            
        return {"response": response_text}
            
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in chat query: {str(e)}\n{error_details}")
        
        # Return an informative error response instead of HTTP 500
        error_response = f"""
        Lo siento, ocurrió un error al procesar tu consulta.
        
        Detalles del error: {str(e)}
        
        Puedo ofrecerte estas alternativas:
        1. Intenta calcular KPIs predefinidos en la primera pestaña
        2. Explora los datos cargados en la sección de vista previa
        3. Intenta hacer una pregunta más simple o específica
        """
        
        return {"response": error_response}

# Asegurarse de cargar todos los módulos necesarios y registrar KPIs predefinidos adicionales
def register_additional_kpis():
    """Registrar KPIs adicionales útiles para cualquier conjunto de datos"""
    global _kpi_engine
    
    if _kpi_engine is None:
        return
    
    # Análisis Exploratorio Básico
    _kpi_engine.kpi_registry.kpis["ResumenEstadístico"] = lambda df: df.describe().to_dict()
    
    # Conteo por categorías
    def conteo_por_categoria(df):
        result = {}
        for col in df.select_dtypes(include=['object']).columns:
            counts = df[col].value_counts().head(10).to_dict()  # Top 10 categorías
            if len(counts) > 0:  # Solo incluir si hay datos válidos
                result[col] = counts
        return result
    
    _kpi_engine.kpi_registry.kpis["ConteoPorCategoría"] = conteo_por_categoria
    
    # Distribución de Valores Numéricos
    def distribucion_numerica(df):
        result = {}
        for col in df.select_dtypes(include=['number']).columns:
            # Crear 10 bins para histograma
            bins = 10
            try:
                hist, bin_edges = np.histogram(df[col].dropna(), bins=bins)
                # Convertir a formato serializable
                result[col] = {
                    "frecuencias": hist.tolist(),
                    "limites": bin_edges.tolist(),
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "promedio": float(df[col].mean()),
                    "mediana": float(df[col].median())
                }
            except:
                pass  # Ignorar columnas problemáticas
        return result
    
    _kpi_engine.kpi_registry.kpis["DistribuciónNumérica"] = distribucion_numerica
    
    # Análisis de Correlación
    def matriz_correlacion(df):
        # Obtener solo columnas numéricas
        num_df = df.select_dtypes(include=['number'])
        if len(num_df.columns) > 1:  # Solo si hay al menos 2 columnas numéricas
            # Calcular correlación y convertir a diccionario
            corr_matrix = num_df.corr().round(2).to_dict()
            return corr_matrix
        return {}
    
    _kpi_engine.kpi_registry.kpis["MatrizCorrelación"] = matriz_correlacion
    
    # Análisis de Tendencia Temporal (si hay columnas de fecha)
    def tendencia_temporal(df):
        # Buscar columnas de fecha
        date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        if not date_cols:
            # Intentar detectar columnas con fechas en formato string
            for col in df.select_dtypes(include=['object']).columns:
                try:
                    pd.to_datetime(df[col], errors='raise')
                    date_cols.append(col)
                except:
                    pass
        
        result = {}
        for date_col in date_cols:
            # Convertir a datetime si no lo es
            if df[date_col].dtype != 'datetime64[ns]':
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            
            # Agrupar por mes y contar registros
            df_temp = df.copy()
            df_temp['month'] = df_temp[date_col].dt.to_period('M')
            monthly_counts = df_temp.groupby('month').size()
            
            # Convertir índice a string para serialización
            result[date_col] = {str(k): int(v) for k, v in monthly_counts.items()}
        
        return result
    
    _kpi_engine.kpi_registry.kpis["TendenciaTemporal"] = tendencia_temporal

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True) 