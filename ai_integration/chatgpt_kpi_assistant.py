import openai
from pydantic import BaseModel
from typing import List, Optional

class ColumnMappingResponse(BaseModel):
    mappings: List[Optional[str]] # List where each detected column maps to a standard column (or None if no match)

def ai_fallback_column_mapping(unmatched_columns, standard_columns):

    """
    Uses ChatGPT to suggest column mappings in a single batch request.
    
    :param unmatched_columns: List of column names that couldn't be matched.
    :param standard_columns: List of standard KPI columns.
    :return: List of suggested standard names for each unmatched column.
    """
    if not unmatched_columns:
        return []  # No unmatched columns to process

    prompt = f"""
    You are an expert in oil & gas data processing. Match the following detected column names 
    to their corresponding standard column names based on well stimulation chemical operations.

    Detected Columns: {unmatched_columns}
    Standard Columns: {standard_columns}

    Return a JSON list where each detected column is mapped to its best matching standard column.
    If no suitable match is found, return null for that column.
    
    Example response:
    {{
        "mappings": ["Qo Before (bpd)", "Brine Volume (m³)", null]
    }}
    """

    try:
        client = openai.OpenAI()  # Correct API initialization
        
        response = client.beta.chat.completions.parse(  # Correct method call
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format=ColumnMappingResponse
        )

        ai_response = response.choices[0].message.parsed
        print(f"Step 1: {ai_response}")
        
        return ai_response.mappings  # Extract and return structured mapping list

    except Exception as e:
        print(f"Error processing AI column mapping: {e}")
        return [None] * len(unmatched_columns)  # Return None for all unmatched columns in case of failure
