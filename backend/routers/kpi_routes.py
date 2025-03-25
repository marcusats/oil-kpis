from fastapi import APIRouter, HTTPException, Depends
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import json
import os
import datetime
from scipy import stats
from .data_routes import get_dataframe, get_dataset_name

# Initialize router
router = APIRouter(prefix="/kpi", tags=["kpi"])


@router.get("/list-kpis")
async def list_kpis():
    """
    List all available KPIs with descriptions.
    """
    kpis = [
        {
            "id": "ResumenEstadístico",
            "name": "Resumen Estadístico",
            "description": "Muestra estadísticas descriptivas básicas para columnas numéricas, incluyendo media, mediana, min, max, y desviación estándar.",
            "requires_numeric": True,
            "min_columns": 1,
            "max_columns": 20
        },
        {
            "id": "MatrizCorrelación",
            "name": "Matriz de Correlación",
            "description": "Calcula la matriz de correlación entre columnas numéricas, mostrando la fuerza y dirección de las relaciones lineales.",
            "requires_numeric": True,
            "min_columns": 2,
            "max_columns": 20
        },
        {
            "id": "DistribuciónNumérica",
            "name": "Distribución Numérica",
            "description": "Muestra la distribución de valores en columnas numéricas mediante histogramas o gráficos de densidad.",
            "requires_numeric": True,
            "min_columns": 1,
            "max_columns": 5
        },
        {
            "id": "TendenciaTemporal",
            "name": "Tendencia Temporal",
            "description": "Analiza la evolución de variables numéricas a lo largo del tiempo, identificando patrones y tendencias.",
            "requires_numeric": True,
            "requires_date": True,
            "min_columns": 1,
            "max_columns": 5
        },
        {
            "id": "ConteoPorCategoría",
            "name": "Conteo por Categoría",
            "description": "Muestra la distribución de valores en columnas categóricas, contando la frecuencia de cada categoría.",
            "requires_categorical": True,
            "min_columns": 1,
            "max_columns": 3
        }
    ]
    
    return kpis


@router.post("/calculate")
@router.get("/calculate")
async def calculate_kpi(kpi_id: str, columns: List[str], group_by: Optional[List[str]] = None):
    """
    Calculate a KPI for the selected columns, optionally grouped by categorical columns.
    """
    _df = get_dataframe()
    
    if _df is None:
        raise HTTPException(status_code=400, detail="No dataset loaded")
    
    # Validate columns
    for col in columns:
        if col not in _df.columns:
            raise HTTPException(status_code=404, detail=f"Column '{col}' not found")
    
    # Validate group_by columns if provided
    if group_by:
        for col in group_by:
            if col not in _df.columns:
                raise HTTPException(status_code=404, detail=f"Group by column '{col}' not found")
    
    try:
        result = {}
        
        # Helper function to make values JSON-serializable
        def make_json_serializable(value):
            """
            Convert values like NaN, Infinity, -Infinity to JSON-serializable values.
            """
            if isinstance(value, float):
                if pd.isna(value) or np.isnan(value):
                    return None
                if np.isinf(value):
                    if value > 0:
                        return 1.0e38  # A very large value as a placeholder for Infinity
                    else:
                        return -1.0e38  # A very small value as a placeholder for -Infinity
            return value
        
        # Function to recursively process dictionaries and lists
        def process_data_structure(data):
            """
            Recursively process dictionaries and lists to make all values JSON-serializable.
            """
            if isinstance(data, dict):
                return {k: process_data_structure(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [process_data_structure(item) for item in data]
            elif isinstance(data, (np.ndarray, pd.Series)):
                return [process_data_structure(item) for item in data]
            else:
                return make_json_serializable(data)
        
        # Resumen Estadístico (Statistical Summary)
        if kpi_id == "ResumenEstadístico":
            # Basic validation
            numeric_cols = [col for col in columns if pd.api.types.is_numeric_dtype(_df[col])]
            if not numeric_cols:
                raise HTTPException(status_code=400, detail="No numeric columns selected for statistical summary")
            
            # Calculate statistics
            if group_by and len(group_by) > 0:
                # Grouped statistics
                result["grouped"] = True
                result["group_by"] = group_by
                result["groups"] = {}
                
                # Process each group
                for name, group in _df.groupby(group_by):
                    # Convert group name to tuple if it's not already
                    if not isinstance(name, tuple):
                        name = (name,)
                    
                    # Create group key
                    group_key = " | ".join([f"{group_by[i]}={str(name[i])}" for i in range(len(name))])
                    
                    # Calculate statistics for this group
                    group_stats = {}
                    for col in numeric_cols:
                        col_stats = {
                            "count": int(group[col].count()),
                            "mean": float(group[col].mean()),
                            "std": float(group[col].std()),
                            "min": float(group[col].min()),
                            "25%": float(group[col].quantile(0.25)),
                            "50%": float(group[col].median()),
                            "75%": float(group[col].quantile(0.75)),
                            "max": float(group[col].max())
                        }
                        
                        # Detect outliers (values beyond 1.5 * IQR)
                        q1 = group[col].quantile(0.25)
                        q3 = group[col].quantile(0.75)
                        iqr = q3 - q1
                        lower_bound = q1 - 1.5 * iqr
                        upper_bound = q3 + 1.5 * iqr
                        outliers = group[(group[col] < lower_bound) | (group[col] > upper_bound)][col]
                        
                        col_stats["outliers"] = {
                            "count": len(outliers),
                            "percentage": float(len(outliers) / len(group) * 100) if len(group) > 0 else 0.0,
                            "min": float(outliers.min()) if len(outliers) > 0 else None,
                            "max": float(outliers.max()) if len(outliers) > 0 else None
                        }
                        
                        group_stats[col] = col_stats
                    
                    result["groups"][group_key] = group_stats
            else:
                # Overall statistics (no grouping)
                result["grouped"] = False
                result["statistics"] = {}
                
                for col in numeric_cols:
                    col_stats = {
                        "count": int(_df[col].count()),
                        "mean": float(_df[col].mean()),
                        "std": float(_df[col].std()),
                        "min": float(_df[col].min()),
                        "25%": float(_df[col].quantile(0.25)),
                        "50%": float(_df[col].median()),
                        "75%": float(_df[col].quantile(0.75)),
                        "max": float(_df[col].max())
                    }
                    
                    # Detect outliers (values beyond 1.5 * IQR)
                    q1 = _df[col].quantile(0.25)
                    q3 = _df[col].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    outliers = _df[(_df[col] < lower_bound) | (_df[col] > upper_bound)][col]
                    
                    col_stats["outliers"] = {
                        "count": len(outliers),
                        "percentage": float(len(outliers) / len(_df) * 100),
                        "min": float(outliers.min()) if len(outliers) > 0 else None,
                        "max": float(outliers.max()) if len(outliers) > 0 else None
                    }
                    
                    result["statistics"][col] = col_stats
        
        # Matriz de Correlación (Correlation Matrix)
        elif kpi_id == "MatrizCorrelación":
            # Basic validation
            numeric_cols = [col for col in columns if pd.api.types.is_numeric_dtype(_df[col])]
            if len(numeric_cols) < 2:
                raise HTTPException(status_code=400, detail="At least 2 numeric columns required for correlation matrix")
            
            # Calculate correlation matrix
            if group_by and len(group_by) > 0:
                # Grouped correlation matrices
                result["grouped"] = True
                result["group_by"] = group_by
                result["groups"] = {}
                
                # Process each group
                for name, group in _df.groupby(group_by):
                    # Convert group name to tuple if it's not already
                    if not isinstance(name, tuple):
                        name = (name,)
                    
                    # Create group key
                    group_key = " | ".join([f"{group_by[i]}={str(name[i])}" for i in range(len(name))])
                    
                    # Calculate correlation matrix for this group
                    corr_matrix = group[numeric_cols].corr().round(4).fillna(0).to_dict()
                    
                    # Convert to format more suitable for frontend
                    formatted_corr = []
                    for col1 in numeric_cols:
                        for col2 in numeric_cols:
                            if col1 != col2 and abs(corr_matrix[col1][col2]) > 0.1:  # Show non-trivial correlations
                                formatted_corr.append({
                                    "column1": col1,
                                    "column2": col2,
                                    "correlation": corr_matrix[col1][col2]
                                })
                    
                    # Sort by absolute correlation value (descending)
                    formatted_corr.sort(key=lambda x: abs(x["correlation"]), reverse=True)
                    
                    result["groups"][group_key] = {
                        "matrix": corr_matrix,
                        "formatted": formatted_corr
                    }
            else:
                # Overall correlation matrix (no grouping)
                result["grouped"] = False
                
                # Calculate correlation matrix
                corr_matrix = _df[numeric_cols].corr().round(4).fillna(0).to_dict()
                
                # Convert to format more suitable for frontend
                formatted_corr = []
                for col1 in numeric_cols:
                    for col2 in numeric_cols:
                        if col1 != col2 and abs(corr_matrix[col1][col2]) > 0.1:  # Show non-trivial correlations
                            formatted_corr.append({
                                "column1": col1,
                                "column2": col2,
                                "correlation": corr_matrix[col1][col2]
                            })
                
                # Sort by absolute correlation value (descending)
                formatted_corr.sort(key=lambda x: abs(x["correlation"]), reverse=True)
                
                result["matrix"] = corr_matrix
                result["formatted"] = formatted_corr
        
        # Distribución Numérica (Numerical Distribution)
        elif kpi_id == "DistribuciónNumérica":
            # Basic validation
            numeric_cols = [col for col in columns if pd.api.types.is_numeric_dtype(_df[col])]
            if len(numeric_cols) < 1:
                raise HTTPException(status_code=400, detail="At least 1 numeric column required for numerical distribution")
            
            # Calculate distributions
            if group_by and len(group_by) > 0:
                # Grouped distributions
                result["grouped"] = True
                result["group_by"] = group_by
                result["groups"] = {}
                
                # Process each group
                for name, group in _df.groupby(group_by):
                    # Convert group name to tuple if it's not already
                    if not isinstance(name, tuple):
                        name = (name,)
                    
                    # Create group key
                    group_key = " | ".join([f"{group_by[i]}={str(name[i])}" for i in range(len(name))])
                    
                    # Calculate distribution for each column in this group
                    group_distributions = {}
                    for col in numeric_cols:
                        # Calculate histogram
                        hist, bin_edges = np.histogram(group[col].dropna(), bins='auto')
                        
                        # Calculate basic statistics
                        distribution = {
                            "histogram": {
                                "counts": hist.tolist(),
                                "bins": bin_edges.tolist(),
                                "bin_centers": [(bin_edges[i] + bin_edges[i+1])/2 for i in range(len(bin_edges)-1)]
                            },
                            "statistics": {
                                "mean": float(group[col].mean()),
                                "median": float(group[col].median()),
                                "std": float(group[col].std()),
                                "skew": float(stats.skew(group[col].dropna())) if len(group[col].dropna()) > 0 else 0,
                                "kurtosis": float(stats.kurtosis(group[col].dropna())) if len(group[col].dropna()) > 0 else 0
                            }
                        }
                        
                        group_distributions[col] = distribution
                    
                    result["groups"][group_key] = group_distributions
            else:
                # Overall distributions (no grouping)
                result["grouped"] = False
                result["distributions"] = {}
                
                for col in numeric_cols:
                    # Calculate histogram
                    hist, bin_edges = np.histogram(_df[col].dropna(), bins='auto')
                    
                    # Calculate basic statistics
                    distribution = {
                        "histogram": {
                            "counts": hist.tolist(),
                            "bins": bin_edges.tolist(),
                            "bin_centers": [(bin_edges[i] + bin_edges[i+1])/2 for i in range(len(bin_edges)-1)]
                        },
                        "statistics": {
                            "mean": float(_df[col].mean()),
                            "median": float(_df[col].median()),
                            "std": float(_df[col].std()),
                            "skew": float(stats.skew(_df[col].dropna())) if len(_df[col].dropna()) > 0 else 0,
                            "kurtosis": float(stats.kurtosis(_df[col].dropna())) if len(_df[col].dropna()) > 0 else 0
                        }
                    }
                    
                    result["distributions"][col] = distribution
        
        # Tendencia Temporal (Time Trend)
        elif kpi_id == "TendenciaTemporal":
            # Basic validation
            numeric_cols = [col for col in columns if pd.api.types.is_numeric_dtype(_df[col])]
            if len(numeric_cols) < 1:
                raise HTTPException(status_code=400, detail="At least 1 numeric column required for time trend")
            
            # Find date columns
            date_cols = [col for col in _df.columns if pd.api.types.is_datetime64_dtype(_df[col])]
            if len(date_cols) == 0:
                raise HTTPException(status_code=400, detail="No date column found for time trend analysis")
            
            # Use the first date column
            date_col = date_cols[0]
            
            # Create a copy of the dataframe with the date column
            trend_df = _df[[date_col] + numeric_cols].copy()
            
            # Sort by date
            trend_df = trend_df.sort_values(by=date_col)
            
            # Group by date periods if needed
            if group_by and len(group_by) > 0:
                # Grouped time trends
                result["grouped"] = True
                result["group_by"] = group_by
                result["groups"] = {}
                
                # Process each group
                for name, group in _df.groupby(group_by):
                    # Convert group name to tuple if it's not already
                    if not isinstance(name, tuple):
                        name = (name,)
                    
                    # Create group key
                    group_key = " | ".join([f"{group_by[i]}={str(name[i])}" for i in range(len(name))])
                    
                    # Create a copy of the group with the date column
                    group_trend_df = group[[date_col] + numeric_cols].copy()
                    
                    # Sort by date
                    group_trend_df = group_trend_df.sort_values(by=date_col)
                    
                    # Resample data - daily, weekly, monthly, or yearly depending on date range
                    date_range = (group_trend_df[date_col].max() - group_trend_df[date_col].min()).days
                    
                    if date_range <= 30:
                        # Daily for short ranges
                        resampled = group_trend_df.set_index(date_col).resample('D').mean()
                        freq = "daily"
                    elif date_range <= 180:
                        # Weekly for medium ranges
                        resampled = group_trend_df.set_index(date_col).resample('W').mean()
                        freq = "weekly"
                    elif date_range <= 730:
                        # Monthly for longer ranges
                        resampled = group_trend_df.set_index(date_col).resample('M').mean()
                        freq = "monthly"
                    else:
                        # Yearly for very long ranges
                        resampled = group_trend_df.set_index(date_col).resample('Y').mean()
                        freq = "yearly"
                    
                    # Reset index to get the date column back
                    resampled = resampled.reset_index()
                    
                    # Format for each numeric column
                    trend_data = {}
                    for col in numeric_cols:
                        # Format the trend data
                        data_points = []
                        for _, row in resampled.iterrows():
                            if not pd.isna(row[col]):
                                data_points.append({
                                    "date": row[date_col].strftime("%Y-%m-%d"),
                                    "value": float(row[col])
                                })
                        
                        # Calculate trend metrics
                        if len(data_points) > 1:
                            # Calculate simple linear regression
                            x = np.array(range(len(data_points)))
                            y = np.array([point["value"] for point in data_points])
                            
                            # Calculate slope and intercept
                            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                            
                            # Calculate trend direction and strength
                            trend_direction = "up" if slope > 0 else "down" if slope < 0 else "flat"
                            trend_strength = abs(r_value)
                            
                            trend_metrics = {
                                "slope": float(slope),
                                "r_squared": float(r_value ** 2),
                                "direction": trend_direction,
                                "strength": float(trend_strength),
                                "significant": bool(p_value < 0.05)
                            }
                        else:
                            trend_metrics = None
                        
                        trend_data[col] = {
                            "data_points": data_points,
                            "frequency": freq,
                            "trend": trend_metrics
                        }
                    
                    result["groups"][group_key] = trend_data
            else:
                # Overall time trends (no grouping)
                result["grouped"] = False
                result["date_column"] = date_col
                result["trends"] = {}
                
                # Resample data - daily, weekly, monthly, or yearly depending on date range
                date_range = (trend_df[date_col].max() - trend_df[date_col].min()).days
                
                if date_range <= 30:
                    # Daily for short ranges
                    resampled = trend_df.set_index(date_col).resample('D').mean()
                    freq = "daily"
                elif date_range <= 180:
                    # Weekly for medium ranges
                    resampled = trend_df.set_index(date_col).resample('W').mean()
                    freq = "weekly"
                elif date_range <= 730:
                    # Monthly for longer ranges
                    resampled = trend_df.set_index(date_col).resample('M').mean()
                    freq = "monthly"
                else:
                    # Yearly for very long ranges
                    resampled = trend_df.set_index(date_col).resample('Y').mean()
                    freq = "yearly"
                
                # Reset index to get the date column back
                resampled = resampled.reset_index()
                
                # Format for each numeric column
                for col in numeric_cols:
                    # Format the trend data
                    data_points = []
                    for _, row in resampled.iterrows():
                        if not pd.isna(row[col]):
                            data_points.append({
                                "date": row[date_col].strftime("%Y-%m-%d"),
                                "value": float(row[col])
                            })
                    
                    # Calculate trend metrics
                    if len(data_points) > 1:
                        # Calculate simple linear regression
                        x = np.array(range(len(data_points)))
                        y = np.array([point["value"] for point in data_points])
                        
                        # Calculate slope and intercept
                        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                        
                        # Calculate trend direction and strength
                        trend_direction = "up" if slope > 0 else "down" if slope < 0 else "flat"
                        trend_strength = abs(r_value)
                        
                        trend_metrics = {
                            "slope": float(slope),
                            "r_squared": float(r_value ** 2),
                            "direction": trend_direction,
                            "strength": float(trend_strength),
                            "significant": bool(p_value < 0.05)
                        }
                    else:
                        trend_metrics = None
                    
                    result["trends"][col] = {
                        "data_points": data_points,
                        "frequency": freq,
                        "trend": trend_metrics
                    }
        
        # Conteo por Categoría (Category Count)
        elif kpi_id == "ConteoPorCategoría":
            # Basic validation
            categorical_cols = [col for col in columns if not pd.api.types.is_numeric_dtype(_df[col])]
            if len(categorical_cols) < 1:
                raise HTTPException(status_code=400, detail="At least 1 categorical column required for category counts")
            
            # Calculate category counts
            if group_by and len(group_by) > 0:
                # Grouped category counts
                result["grouped"] = True
                result["group_by"] = group_by
                result["groups"] = {}
                
                # Process each group
                for name, group in _df.groupby(group_by):
                    # Convert group name to tuple if it's not already
                    if not isinstance(name, tuple):
                        name = (name,)
                    
                    # Create group key
                    group_key = " | ".join([f"{group_by[i]}={str(name[i])}" for i in range(len(name))])
                    
                    # Calculate counts for each column in this group
                    group_counts = {}
                    for col in categorical_cols:
                        value_counts = group[col].value_counts().head(20)  # Limit to top 20 categories
                        total_count = len(group)
                        
                        # Format the counts data
                        counts_data = []
                        for cat, count in value_counts.items():
                            counts_data.append({
                                "category": str(cat),
                                "count": int(count),
                                "percentage": float(count / total_count * 100)
                            })
                        
                        group_counts[col] = {
                            "total": total_count,
                            "unique_categories": int(group[col].nunique()),
                            "counts": counts_data
                        }
                    
                    result["groups"][group_key] = group_counts
            else:
                # Overall category counts (no grouping)
                result["grouped"] = False
                result["counts"] = {}
                
                for col in categorical_cols:
                    value_counts = _df[col].value_counts().head(20)  # Limit to top 20 categories
                    total_count = len(_df)
                    
                    # Format the counts data
                    counts_data = []
                    for cat, count in value_counts.items():
                        counts_data.append({
                            "category": str(cat),
                            "count": int(count),
                            "percentage": float(count / total_count * 100)
                        })
                    
                    result["counts"][col] = {
                        "total": total_count,
                        "unique_categories": int(_df[col].nunique()),
                        "counts": counts_data
                    }
        
        else:
            raise HTTPException(status_code=404, detail=f"KPI '{kpi_id}' not found")
        
        # Add metadata
        result["kpi_id"] = kpi_id
        result["columns"] = columns
        result["dataset_name"] = get_dataset_name()
        result["timestamp"] = datetime.datetime.now().isoformat()
        
        # Process result to make all values JSON-serializable
        result = process_data_structure(result)
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating KPI: {str(e)}") 