import pandas as pd
from filters import FilterHandler
from base_kpi import BaseKPI

class KPIEngine:
    def __init__(self, df, service_line,mapping_path):
        """
        :param df: Filtered DataFrame from FilterHandler.
        :param service_line: The service line (e.g., "Stimulation", "Artificial Lift").
        """
        self.df = df
        self.mapping_path = mapping_path
        self.service_line = service_line
        self.kpi_registry = BaseKPI(service_line=self.service_line,mapping_path=self.mapping_path)
        
    def calculate_kpi(self, kpi_name):
        """
        Processes and calculates a KPI.
        :param kpi_name: Name of the KPI to compute.
        :return: Computed KPI result.
        """
        if kpi_name not in self.kpi_registry.kpis:
            raise ValueError(f"KPI '{kpi_name}' is not registered for {self.kpi_registry.service_line}.")

        # Retrieve KPI function (fixing unpacking issue)
        kpi_function = self.kpi_registry.kpis[kpi_name]

        # Compute KPI on the pre-filtered dataset
        return kpi_function(self.df)

    def _extract_column_groups(self):
        """
        Extracts all column names and column groups dynamically from the provided mapping JSON file.
        Returns:
            - ALL_COLUMNS: A list of all available column names.
            - COLUMN_GROUPS: A dictionary of grouped column names.
        """
        try:
            with open(self.mapping_path, "r", encoding="utf-8") as f:
                COLUMN_MAPPING = json.load(f)

            # Extract all column names
            ALL_COLUMNS = list(COLUMN_MAPPING.keys())

            # Extract column groups dynamically
            COLUMN_GROUPS = {}
            for column_name, properties in COLUMN_MAPPING.items():
                group = properties.get("group")
                if group:
                    if group not in COLUMN_GROUPS:
                        COLUMN_GROUPS[group] = []
                    COLUMN_GROUPS[group].append(column_name)

            return ALL_COLUMNS, COLUMN_GROUPS  # Return both

        except (UnicodeDecodeError, FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading column mapping: {e}")
            return [], {}  # Return empty values to prevent crashes

    def create_dynamic_kpi(self, formula):
        """
        Converts an AI-generated formula into a callable function.
        Automatically replaces COLUMN_GROUPS references with actual column names.
        """
        # Load all column names and groups dynamically
        ALL_COLUMNS, COLUMN_GROUPS = self._extract_column_groups()

        # Replace COLUMN_GROUPS references with actual column names
        for group, columns in COLUMN_GROUPS.items():
            formula = formula.replace(f"COLUMN_GROUPS['{group}']", str(columns))

        def safe_exec(df):
            safe_globals = {"pd": pd}  # Allow only Pandas
            safe_locals = {"df": df}  # Allow access to DataFrame
            
            try:
                exec(formula, safe_globals, safe_locals)  # Executes multiple lines of Python code
                return safe_locals.get("result", None)  # Ensure "result" is returned
            except Exception as e:
                print(f"Error evaluating AI-generated KPI: {e}")
                return None

        return safe_exec  # Returns a callable function