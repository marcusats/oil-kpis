import pandas as pd
import numpy as np
import json

def prepare_data_context(df):
    """Prepare data context for AI prompt"""
    # Initialize a dictionary to hold both text and structured data
    context = {
        "dataset_info": {
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns)
        },
        "text_summary": [],  # For text-based description (readable)
        "column_types": {},  # For structured data
        "statistics": {},    # For statistical data
        "sample_data": []    # For sample rows
    }
    
    try:
        # Basic dataset info
        context["text_summary"].append(f"- El conjunto de datos tiene {len(df)} registros y {len(df.columns)} columnas.")
        
        # Column information
        context["text_summary"].append("\n### Columnas:")
        for col in df.columns:
            col_type = str(df[col].dtype)
            num_null = df[col].isna().sum()
            context["text_summary"].append(f"- {col} (tipo: {col_type}, valores nulos: {num_null})")
            context["column_types"][col] = {
                "type": col_type,
                "null_count": int(num_null),
                "unique_count": int(df[col].nunique())
            }
        
        # Statistical summary for numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            context["text_summary"].append("\n### Resumen estadístico de columnas numéricas:")
            for col in numeric_cols[:10]:  # Limit to first 10 numeric columns
                try:
                    stats = df[col].describe()
                    context["text_summary"].append(f"- {col}: min={stats['min']:.2f}, max={stats['max']:.2f}, media={stats['mean']:.2f}, mediana={stats['50%']:.2f}")
                    context["statistics"][col] = {
                        "min": float(stats["min"]),
                        "max": float(stats["max"]),
                        "mean": float(stats["mean"]),
                        "median": float(stats["50%"]),
                        "std": float(stats["std"])
                    }
                except:
                    context["text_summary"].append(f"- {col}: [error al calcular estadísticas]")
        
        # Categorical columns info
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) > 0:
            context["text_summary"].append("\n### Información de columnas categóricas:")
            for col in cat_cols[:10]:  # Limit to first 10 categorical columns
                num_unique = df[col].nunique()
                context["text_summary"].append(f"- {col}: {num_unique} valores únicos")
                
                # Top 5 values for each categorical column
                if num_unique > 0 and num_unique < 100:  # Only show if not too many unique values
                    try:
                        top_values = df[col].value_counts().head(5)
                        value_info = ", ".join([f"{val}: {count}" for val, count in top_values.items()])
                        context["text_summary"].append(f"  Valores más comunes: {value_info}")
                        
                        # Store in structured format
                        context["statistics"][col] = {
                            "unique_count": int(num_unique),
                            "top_values": {str(val): int(count) for val, count in top_values.items()}
                        }
                    except:
                        pass
        
        # Date columns info if any
        date_cols = df.select_dtypes(include=['datetime']).columns
        if len(date_cols) > 0:
            context["text_summary"].append("\n### Información de columnas de fecha:")
            for col in date_cols:
                try:
                    min_date = df[col].min()
                    max_date = df[col].max()
                    context["text_summary"].append(f"- {col}: desde {min_date} hasta {max_date}")
                    
                    # Store in structured format
                    context["statistics"][col] = {
                        "min_date": str(min_date),
                        "max_date": str(max_date),
                        "date_range_days": (max_date - min_date).days
                    }
                except:
                    context["text_summary"].append(f"- {col}: [error al procesar fechas]")
                
        # Add correlation information for numeric columns
        if len(numeric_cols) >= 2:
            context["text_summary"].append("\n### Correlaciones importantes entre columnas numéricas:")
            try:
                corr_matrix = df[numeric_cols].corr().abs()
                
                # Get top 5 correlations (excluding self-correlations)
                corrs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        col1 = corr_matrix.columns[i]
                        col2 = corr_matrix.columns[j]
                        corr_val = corr_matrix.iloc[i, j]
                        corrs.append((col1, col2, corr_val))
                
                # Sort by correlation value
                corrs.sort(key=lambda x: x[2], reverse=True)
                
                # Add top correlations to context
                context["correlations"] = []
                for col1, col2, corr_val in corrs[:5]:
                    context["text_summary"].append(f"- {col1} y {col2}: {corr_val:.4f}")
                    context["correlations"].append({
                        "column1": str(col1),
                        "column2": str(col2),
                        "correlation": float(corr_val)
                    })
            except:
                context["text_summary"].append("  [Error al calcular correlaciones]")
        
        # Add sample data
        try:
            sample_rows = min(5, len(df))
            sample_data = df.head(sample_rows).to_dict(orient='records')
            
            # For each sample row, convert non-JSON serializable values to strings
            for i, row in enumerate(sample_data):
                for key, value in row.items():
                    if isinstance(value, (pd.Timestamp, pd.Period)):
                        sample_data[i][key] = str(value)
                    elif isinstance(value, (np.int64, np.int32)):
                        sample_data[i][key] = int(value)
                    elif isinstance(value, (np.float64, np.float32)):
                        sample_data[i][key] = float(value)
                    elif pd.isna(value):
                        sample_data[i][key] = None
            
            context["sample_data"] = sample_data
            context["text_summary"].append("\n### Muestra de datos:")
            for i, row in enumerate(sample_data[:3]):  # Show only first 3 samples in text
                context["text_summary"].append(f"- Registro {i+1}: " + ", ".join([f"{k}={v}" for k, v in list(row.items())[:5]]) + "...")
        except Exception as e:
            context["text_summary"].append(f"  [Error al añadir muestra: {str(e)}]")
            
        return context
    except Exception as e:
        # If anything fails, return a minimal context
        return {
            "dataset_info": {"rows": len(df), "columns": len(df.columns)},
            "text_summary": [f"Error al preparar contexto: {str(e)}"],
            "error": str(e)
        }

def get_data_for_query(df, query_text):
    """Extract specific data related to the user query"""
    
    # Initialize results dictionary
    result = {}
    
    try:
        # Lowercase query for easier matching
        query_lower = query_text.lower()
        
        # Try to identify columns mentioned in the query
        mentioned_columns = []
        for col in df.columns:
            col_lower = col.lower()
            # Check if column name is mentioned in the query
            if col_lower in query_lower:
                mentioned_columns.append(col)
        
        # If specific columns are mentioned, include their data
        if mentioned_columns:
            result["mentioned_columns"] = mentioned_columns
            
            # For each mentioned column, get relevant statistics
            for col in mentioned_columns:
                col_data = {}
                
                # For numeric columns
                if pd.api.types.is_numeric_dtype(df[col]):
                    # Get descriptive statistics
                    col_data["type"] = "numeric"
                    stats = df[col].describe().to_dict()
                    # Clean up stats to ensure they're serializable
                    for stat, value in stats.items():
                        if np.isnan(value) or pd.isna(value):
                            stats[stat] = None
                        elif np.isinf(value):
                            stats[stat] = "Infinity" if value > 0 else "-Infinity"
                    col_data["stats"] = stats
                    
                    # Check for NaN values
                    nan_count = df[col].isna().sum()
                    col_data["nan_count"] = int(nan_count)
                    col_data["nan_percentage"] = float(nan_count / len(df) * 100)
                    
                # For categorical columns
                elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
                    col_data["type"] = "categorical"
                    # Get value counts (top 10)
                    try:
                        value_counts = df[col].value_counts().head(10).to_dict()
                        # Convert any non-serializable keys to strings
                        clean_counts = {}
                        for k, v in value_counts.items():
                            if pd.isna(k):
                                clean_key = "NaN"
                            else:
                                clean_key = str(k)
                            clean_counts[clean_key] = int(v)
                        col_data["value_counts"] = clean_counts
                        col_data["unique_count"] = df[col].nunique()
                    except:
                        col_data["error"] = "Error getting value counts"
                
                # For datetime columns
                elif pd.api.types.is_datetime64_dtype(df[col]):
                    col_data["type"] = "datetime"
                    # Get min and max dates
                    min_date = df[col].min()
                    max_date = df[col].max()
                    col_data["min_date"] = str(min_date)
                    col_data["max_date"] = str(max_date)
                    col_data["range_days"] = (max_date - min_date).days
                
                result[col] = col_data
        
        # Check if query is asking for correlations
        correlation_keywords = ["correlación", "correlacion", "relación", "relacion", "correlation"]
        if any(keyword in query_lower for keyword in correlation_keywords):
            # Calculate correlation matrix for numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 1:
                try:
                    corr_matrix = df[numeric_cols].corr().abs()
                    # Convert to dictionary with clean values
                    corr_dict = {}
                    for col1 in corr_matrix.columns:
                        corr_dict[col1] = {}
                        for col2 in corr_matrix.index:
                            val = corr_matrix.loc[col2, col1]
                            if np.isnan(val) or pd.isna(val):
                                corr_dict[col1][col2] = None
                            else:
                                corr_dict[col1][col2] = float(val)
                    result["correlation_matrix"] = corr_dict
                except:
                    result["correlation_error"] = "Error calculating correlation matrix"
        
        # Check if query is asking for groupby analysis
        groupby_keywords = ["agrupar", "grupo", "groupby", "group by"]
        if any(keyword in query_lower for keyword in groupby_keywords):
            # Try to identify potential categorical columns for groupby
            cat_cols = df.select_dtypes(include=['object', 'category']).columns
            if cat_cols.any():
                # Use the first categorical column as group by column
                group_col = cat_cols[0]
                # Try to find aggregation columns
                agg_cols = df.select_dtypes(include=['number']).columns[:2]  # Use first 2 numeric columns
                
                if not agg_cols.empty:
                    try:
                        # Perform group by and aggregation
                        agg_dict = {col: ['mean', 'sum', 'count'] for col in agg_cols}
                        grouped = df.groupby(group_col)[agg_cols].agg(agg_dict).head(10)
                        
                        # Convert result to a clean dictionary
                        group_result = {}
                        for col, group_data in grouped.items():
                            col_name, agg_func = col
                            if group_data.name not in group_result:
                                group_result[str(group_data.name)] = {}
                            
                            # Clean the values
                            clean_values = {}
                            for idx, val in group_data.items():
                                if pd.isna(val) or np.isnan(val):
                                    clean_val = None
                                elif np.isinf(val):
                                    clean_val = "Infinity" if val > 0 else "-Infinity"
                                else:
                                    clean_val = float(val)
                                clean_values[str(idx)] = clean_val
                            
                            group_result[str(group_data.name)][agg_func] = clean_values
                        
                        result["groupby_analysis"] = {
                            "group_column": group_col,
                            "agg_columns": list(agg_cols),
                            "results": group_result
                        }
                    except:
                        result["groupby_error"] = "Error performing groupby analysis"
        
        return result
    except Exception as e:
        return {"error": str(e)}

def generate_basic_analysis(user_query, df):
    """Generate basic analysis without using AI"""
    response_text = f"## Respuesta a: '{user_query}'\n\n"
    response_text += f"El conjunto de datos tiene {len(df)} registros y {len(df.columns)} columnas.\n\n"
    
    # Add column information
    response_text += "### Columnas disponibles:\n\n"
    for col in df.columns[:10]:  # Show first 10 columns
        response_text += f"- {col}\n"
    
    if len(df.columns) > 10:
        response_text += f"- ... y {len(df.columns) - 10} columnas más\n\n"
    
    # Add basic statistics if numeric columns are present
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        response_text += "\n### Resumen estadístico de columnas numéricas:\n\n"
        for col in numeric_cols[:5]:  # Show first 5 numeric columns
            response_text += f"**{col}**:\n"
            response_text += f"- Media: {df[col].mean():.2f}\n"
            response_text += f"- Mediana: {df[col].median():.2f}\n"
            response_text += f"- Mínimo: {df[col].min():.2f}\n"
            response_text += f"- Máximo: {df[col].max():.2f}\n\n"
    
    return {"response": response_text} 