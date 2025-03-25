import openai
from pydantic import BaseModel
from typing import List, Optional
import openai
import json
from pydantic import BaseModel, ValidationError 

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
    
    :param unmatched_columns: List of column names that couldn't be matched.
    :param standard_columns: List of standard KPI columns.
    :return: List of suggested standard names for each unmatched column.
    """
    if not unmatched_columns:
        print('No unmatched columns')
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
        client = openai.OpenAI(api_key=OPENAI_API_KEY)  # Correct API initialization
        
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


def ai_suggest_column_category(column_name, sample_data):
    """
    Uses ChatGPT to suggest a category for an unknown column.
    :param column_name: The name of the column.
    :param sample_data: A sample value from the column.
    :return: Suggested metadata for the column.
    """
    prompt = f"""
    You are an expert in oil & gas data processing. Categorize the following column:
    Column Name: {column_name}
    Sample Value: {sample_data}
    Return a JSON object with:
    - "category": (e.g., "Chemical Volume", "Pressure", "Flow Rate", etc.)
    - "type": ("numerical", "categorical", "date")
    - "description": A short explanation of what this column represents.
    - "group": If applicable, provide a group name (e.g., "Chemical Usage", "Well Pressure").
    """
    
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )

        ai_response = response.choices[0].message.content.strip()
        suggested_metadata = json.loads(ai_response)
        return suggested_metadata

    except Exception as e:
        print(f"Error categorizing column '{column_name}': {e}")
        return {"category": "Uncategorized", "type": "unknown", "description": "No AI suggestion available.", "group": "None"}
    
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

    Return the output in valid JSON format as shown below:

    {{
        "name": "KPI Name",
        "formula": "MULTI-LINE CODE HERE",
        "description": "Brief description of the KPI",
        "service_line": "{service_line}"
    }}
    """

    try:
        client = openai.OpenAI()
        response = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format=AIKPI
        )

        validated_kpi = response.choices[0].message.parsed
        print(validated_kpi)
        return validated_kpi.model_dump()  # Convert AI response to dictionary

    except ValidationError as e:
        print(f"Error: AI-generated KPI is not in the correct format: {e}")
        return None

    except Exception as e:
        print(f"Error generating KPI: {e}")
        return None