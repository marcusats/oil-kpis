import openai
from pydantic import BaseModel
from typing import List, Optional
import openai
import json
from pydantic import BaseModel, ValidationError 
import os

class ColumnMappingResponse(BaseModel):
    mappings: List[Optional[str]] # List where each detected column maps to a standard column (or None if no match)

class AIKPI(BaseModel):
    name: str
    formula: str
    description: str
    service_line: str

### Extract Column Groups from Mapping File
def extract_column_groups(mapping_path):
    """
    Extracts all column names and column groups dynamically from the provided mapping JSON file.
    Returns:
        - ALL_COLUMNS: A list of all available column names.
        - COLUMN_GROUPS: A dictionary of grouped column names.
    """
    try:
        with open(mapping_path, "r", encoding="utf-8") as f:
            COLUMN_MAPPING = json.load(f)

        # ✅ Extract all column names
        ALL_COLUMNS = list(COLUMN_MAPPING.keys())

        # ✅ Extract column groups dynamically
        COLUMN_GROUPS = {}
        for column_name, properties in COLUMN_MAPPING.items():
            group = properties.get("group")
            if group:
                if group not in COLUMN_GROUPS:
                    COLUMN_GROUPS[group] = []
                COLUMN_GROUPS[group].append(column_name)

        return ALL_COLUMNS, COLUMN_GROUPS  # ✅ Return both

    except (UnicodeDecodeError, FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading column mapping: {e}")
        return [], {}  # ✅ Return empty values to prevent crashes

def ai_fallback_column_mapping(unmatched_columns, standard_columns):
    """
    Uses ChatGPT to suggest column mappings in a single batch request.
    """
    # Get API key from environment
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OpenAI API key not found in environment variables")
        return []
    
    if not unmatched_columns or not standard_columns:
        return []

    prompt = f"""
    Map each of these custom data columns to the most suitable standard column name, or return null if no match:

    Custom columns: {unmatched_columns}
    Standard columns: {standard_columns}

    For each custom column, return either the matching standard column name or null.
    The order should match the custom columns order exactly.
    """

    try:
        # Use the standard OpenAI client
        client = openai.OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a helpful assistant that maps custom data columns to standard column names based on semantic similarity."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=800
        )
        
        mapping_text = response.choices[0].message.content
        
        # Parse the response - handle different response formats
        mappings = []
        
        # Try to parse as a list or other structured format
        try:
            # Check if it's a JSON array format
            if mapping_text.strip().startswith("[") and mapping_text.strip().endswith("]"):
                mappings = json.loads(mapping_text)
            # Check for line-by-line format
            else:
                for line in mapping_text.strip().split("\n"):
                    if "->" in line or ":" in line:
                        delimiter = "->" if "->" in line else ":"
                        parts = line.split(delimiter)
                        if len(parts) == 2:
                            standard_col = parts[1].strip()
                            # Convert "null" or "None" text to None
                            if standard_col.lower() in ["null", "none", "n/a"]:
                                mappings.append(None)
                            else:
                                mappings.append(standard_col)
                        else:
                            mappings.append(None)
                    else:
                        # If line contains a standard column exactly, use it
                        match = next((col for col in standard_columns if col in line), None)
                        mappings.append(match)
        except:
            # If parsing fails, return empty list
            print("Failed to parse AI response for column mapping")
            return []
            
        # Ensure we have the same number of mappings as unmatched columns
        if len(mappings) != len(unmatched_columns):
            print(f"Warning: AI returned {len(mappings)} mappings for {len(unmatched_columns)} columns")
            # Pad or truncate the list as needed
            if len(mappings) < len(unmatched_columns):
                mappings.extend([None] * (len(unmatched_columns) - len(mappings)))
            else:
                mappings = mappings[:len(unmatched_columns)]
                
        return mappings
    except Exception as e:
        print(f"Error in AI column mapping: {str(e)}")
        return [None] * len(unmatched_columns)  # Return None for all columns on error


def ai_suggest_column_category(column_name, sample_data):
    """
    Uses ChatGPT to suggest a category for a column based on its name and sample data.
    Returns: One of 'categorical', 'numerical', 'date', or 'unknown'
    """
    # Get API key from environment
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OpenAI API key not found in environment variables")
        return "unknown"
    
    if not column_name:
        return "unknown"

    # Prepare sample data for the prompt
    sample_str = str(sample_data)[:200]  # Limit sample data size
    
    prompt = f"""
    Analyze this column and determine its type:
    
    Column name: {column_name}
    Sample data: {sample_str}
    
    Return ONLY ONE of these categories:
    - categorical: For enumerated types, text categories, IDs, etc.
    - numerical: For integers, floats, percentages, etc.
    - date: For dates, timestamps, etc.
    - unknown: If you cannot determine the type
    
    Reply with just the category name, nothing else.
    """

    try:
        # Use the standard OpenAI client
        client = openai.OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a data analyst that categorizes data columns."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=20  # Short response
        )
        
        # Extract category from response
        category = response.choices[0].message.content.strip().lower()
        
        # Validate category
        valid_categories = ["categorical", "numerical", "date", "unknown"]
        if category not in valid_categories:
            return "unknown"
            
        return category
        
    except Exception as e:
        print(f"Error in AI column categorization: {str(e)}")
        return "unknown"  # Return unknown on error
    
class AIKPI(BaseModel):
    name: str
    formula: str
    description: str
    service_line: str

def generate_ai_kpi(user_prompt, service_line, mapping_path):
    """
    Uses ChatGPT to generate a new KPI formula for the selected service line.
    Ensures AI-generated KPIs are in the correct format using Pydantic validation.
    """
    # ✅ Load all column names and groups dynamically
    ALL_COLUMNS, COLUMN_GROUPS = extract_column_groups(mapping_path)

    # ✅ Convert data to JSON string for AI reference
    all_columns_str = json.dumps(ALL_COLUMNS, indent=4)
    column_groups_str = json.dumps(COLUMN_GROUPS, indent=4)

    system_prompt = f"""
    You are an expert in oil & gas KPI calculations.
    The user wants to create a new KPI for the {service_line} dataset.

    Here are all available column names:
    {all_columns_str}

    Here are predefined column groups (Use these when applicable):
    {column_groups_str}

    - Ensure that any formula you generate only references valid columns.
    - Prioritize using column groups when relevant.
    - If a column is not in a group, reference it directly.
    - Your output can contain multiple lines of Python code.
    - Always assign the final KPI result to a variable named `result`.
    - Dataframe should always be referred to as "df"

    Return the output as a dictionary with these exact keys:
    {{
        "name": "KPI Name",
        "formula": "MULTI-LINE CODE HERE",
        "description": "Brief description of the KPI",
        "service_line": "{service_line}"
    }}
    """
    
    # Get API key from environment
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OpenAI API key not found in environment variables")
        return None

    try:
        # Use the standard OpenAI client
        client = openai.OpenAI(api_key=api_key)
        
        # Use chat completions with JSON response
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=800,
            response_format={"type": "json_object"}
        )
        
        # Extract the JSON content
        try:
            response_content = response.choices[0].message.content
            kpi_data = json.loads(response_content)
            
            # Validate with Pydantic
            validated_kpi = AIKPI(**kpi_data)
            return validated_kpi.model_dump()  # Convert to dictionary
            
        except json.JSONDecodeError:
            print(f"Error: AI response is not valid JSON: {response_content}")
            return None
        except ValidationError as e:
            print(f"Error: AI-generated KPI is not in the correct format: {e}")
            return None

    except Exception as e:
        print(f"Error generating KPI: {e}")
        return None