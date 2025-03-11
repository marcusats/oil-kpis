import pandas as pd
import json
from rapidfuzz import process, fuzz
from ai_integration.chatgpt_kpi_assistant import ai_fallback_column_mapping

class DataHandler:
    def __init__(self, source_type, source, mapping_path):
        """
        :param source_type: Type of data source (excel, csv, db, api, json).
        :param source: File path, database connection, or API endpoint.
        :param mapping_path: Path to JSON column mapping file.

        """
        self.source_type = source_type
        self.source = source
        self.mapping_path = mapping_path
        self.column_mappings = self._load_column_mapping()
        self.data = None
        

    def load_data(self):
        """Load data based on the selected source type and apply column mapping."""
        try:
            if self.source_type == "excel":
                self.data = pd.read_excel(self.source)
            elif self.source_type == "csv":
                self.data = pd.read_csv(self.source)
            elif self.source_type == "db":
                self.data = self._load_from_db()
            elif self.source_type == "api":
                self.data = self._load_from_api()
            elif self.source_type == "json":
                self.data = self._load_from_json()
            else:
                return f"Unsupported source type: {self.source_type}"

            # Apply column mapping after loading data
            self._apply_column_mapping()

            return self.data
        except Exception as e:
            return f"Error loading data: {str(e)}"

    def _load_from_db(self):
        """Load data from an SQLite database (can be extended for other DBs)."""
        conn = sqlite3.connect(self.source)  # Example for SQLite
        query = "SELECT * FROM kpi_data"  # Modify as needed
        return pd.read_sql(query, conn)

    def _load_from_api(self):
        """Fetch data from an API endpoint."""
        response = requests.get(self.source)
        if response.status_code == 200:
            return pd.DataFrame(response.json())  # Assuming API returns JSON
        return f"API request failed with status: {response.status_code}"

    def _load_from_json(self):
        """Load data from a JSON file."""
        with open(self.source, "r") as f:
            return pd.DataFrame(json.load(f))
    
    def _load_column_mapping(self):
        """Loads column mapping definitions from a JSON file."""
        try:
            with open(self.mapping_path, "r", encoding="utf-8") as file:
                return json.load(file)
        except Exception as e:
            print(f"Error loading column mapping: {e}")
            return {}

    def _find_best_match(self, detected_column):
            """Finds the best match for a detected column using RapidFuzz."""
            best_match = None
            highest_score = 0
            for standard_name, details in self.column_mappings.items():
                possible_matches = [standard_name] + details["alternative"]
                match, score,index = process.extractOne(detected_column, possible_matches, scorer=fuzz.ratio)

                if score > highest_score:
                    highest_score = score
                    best_match = standard_name
            # Return highest score
            if highest_score > 75:
                return best_match, highest_score
            else: 
                return None, None
        
    def _apply_column_mapping(self):
        """Maps detected columns to standarized names using RapidFuzz and AI fallback"""
        detected_columns = list(self.data.columns)
        mapped_columns = {}
        unmatched_columns = []
        
        # Step 1: Try RapidFuzz first
        for detected_col in detected_columns:
            best_match, highest_score  = self._find_best_match(detected_col)
            print(f"Input: {detected_col} -> Standard: {best_match} with a score of {highest_score}")

            if best_match:
                mapped_columns[detected_col] = best_match

            else:
                unmatched_columns.append(detected_col)

        # Step 2: Use ChatGPT for all unmatched columns in one request
        if unmatched_columns:
            ai_suggestions = ai_fallback_column_mapping(unmatched_columns, list(self.column_mappings.keys()))
            # ai_suggestions = [None for i in unmatched_columns] # Place holder while AI tool is addressed
            

            # Update mappings with AI suggestions
            for i, col in enumerate(unmatched_columns):
                mapped_columns[col] = ai_suggestions[i] if ai_suggestions[i] else col  # Keep original if AI fails

        # Step 3: Rename columns in the DataFrame
        self.data.rename(columns=mapped_columns, inplace=True)

        # Rename columns in the DataFrame
        self.data.rename(columns=mapped_columns, inplace=True)
