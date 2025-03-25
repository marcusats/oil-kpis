import logging
from fastapi import APIRouter, HTTPException, UploadFile, File
import pandas as pd
import os
from backend.dataset import get_dataframe, load_dataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

@router.get("/data/dataset-info")
async def get_dataset_info():
    """Get basic information about the loaded dataset."""
    df = get_dataframe()
    if df is None:
        raise HTTPException(status_code=400, detail="No dataset loaded")
    
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": df.columns.tolist(),
        "dtypes": {col: str(df[col].dtype) for col in df.columns}
    }

@router.get("/data/raw-data")
async def get_raw_data(offset: int = 0, limit: int = 100):
    """Get raw data from the loaded dataset with pagination."""
    df = get_dataframe()
    if df is None:
        raise HTTPException(status_code=400, detail="No dataset loaded")
    
    # Calculate total rows and pages
    total_rows = len(df)
    total_pages = (total_rows + limit - 1) // limit if limit > 0 else 1
    
    # Get the current page of data
    end_idx = min(offset + limit, total_rows)
    data = df.iloc[offset:end_idx].to_dict(orient="records")
    
    return {
        "data": data,
        "pagination": {
            "offset": offset,
            "limit": limit,
            "total_rows": total_rows,
            "total_pages": total_pages
        }
    }

@router.post("/data/load-dataset")
async def load_dataset_endpoint(file: UploadFile = File(...)):
    """Load a dataset from an uploaded file."""
    # Create uploads directory if it doesn't exist
    upload_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    
    # Save the uploaded file
    file_path = os.path.join(upload_dir, file.filename)
    
    try:
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Load the dataset
        result = load_dataset(file_path)
        
        if not result.get("success", False):
            raise HTTPException(status_code=400, detail=result.get("error", "Failed to load dataset"))
        
        return result
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error loading dataset: {str(e)}") 