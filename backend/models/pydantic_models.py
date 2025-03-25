from pydantic import BaseModel
from typing import Dict, List, Optional, Any, Union

# API Models
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