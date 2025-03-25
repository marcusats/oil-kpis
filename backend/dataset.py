import pandas as pd
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global dataframe to store the loaded dataset
_dataframe = None

def get_dataframe():
    """Return the currently loaded dataframe."""
    global _dataframe
    
    # For testing purposes, if no dataframe is loaded, create a sample one
    if _dataframe is None:
        logger.info("No dataset loaded, creating a sample dataset for testing")
        # Create a sample dataframe with petroleum-related data
        data = {
            'Date': pd.date_range(start='2023-01-01', periods=100, freq='D'),
            'Well_ID': ['W' + str(i % 10) for i in range(100)],
            'Production_BBL': [round(500 + 200 * (i % 10) + 50 * (i % 5), 2) for i in range(100)],
            'Pressure_PSI': [round(2000 + 500 * (i % 7) - 200 * (i % 3), 2) for i in range(100)],
            'Temperature_F': [round(150 + 20 * (i % 5) - 10 * (i % 2), 2) for i in range(100)],
            'Water_Cut_PCT': [round(5 + 2 * (i % 10), 2) for i in range(100)],
            'Gas_Oil_Ratio': [round(1000 + 200 * (i % 8), 2) for i in range(100)],
            'API_Gravity': [round(30 + 5 * (i % 4), 2) for i in range(100)],
            'Field': ['Field_' + chr(65 + i % 5) for i in range(100)],
            'Operator': ['Operator_' + chr(65 + i % 3) for i in range(100)]
        }
        _dataframe = pd.DataFrame(data)
    
    return _dataframe

def load_dataset(file_path):
    """Load a dataset from a file path."""
    global _dataframe
    
    logger.info(f"Loading dataset from: {file_path}")
    
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        # Load the dataset based on file extension
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.csv':
            _dataframe = pd.read_csv(file_path)
        elif file_ext in ['.xls', '.xlsx']:
            _dataframe = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        # Log success
        logger.info(f"Successfully loaded dataset with {len(_dataframe)} rows and {len(_dataframe.columns)} columns")
        
        # Return dataset info
        return {
            "success": True,
            "rows": len(_dataframe),
            "columns": len(_dataframe.columns),
            "column_names": _dataframe.columns.tolist(),
            "dtypes": {col: str(_dataframe[col].dtype) for col in _dataframe.columns}
        }
        
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        } 