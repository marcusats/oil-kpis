import json
import pandas as pd
import numpy as np
import importlib

class BaseKPI:
    def __init__(self, service_line,mapping_path, kpi_storage_file="custom_kpi.json"):
        """
        Generic KPI registry that loads predefined and AI-generated KPIs.
        """
        self.service_line = service_line
        self.mapping_path = mapping_path
        self.kpi_storage_file = kpi_storage_file
        self.kpis = {}

        # Load both predefined and AI-generated KPIs
        self.load_kpis()

    def load_kpis(self):
        """
        Dynamically loads predefined and AI-generated KPIs.
        """
        try:
            # Import the service-line module and get all functions
            module_name = f"{self.service_line.lower()}_kpi_definitions"
            kpi_module = importlib.import_module(module_name)

            # Get all function-based KPIs
            self.kpis.update(kpi_module.get_kpis(self.mapping_path))  # Loads functions instead of a dictionary
        except ModuleNotFoundError:
            print(f"Warning: No predefined KPIs found for {self.service_line}.")

        # Load AI-generated KPIs
        try:
            with open(self.kpi_storage_file, "r", encoding="utf-8") as f:
                ai_kpis = json.load(f)
                for kpi in ai_kpis:
                    if kpi["service_line"].lower() == self.service_line.lower():
                        self.kpis[kpi["name"]] = self.create_dynamic_kpi(kpi["formula"])
        except FileNotFoundError:
            print("No AI-generated KPIs found.")

    def create_dynamic_kpi(self, formula):
        """
        Safely converts an AI-generated formula into a callable function.
        Allows only mathematical operations on the DataFrame.
        """
        def safe_eval(df):
            safe_globals = {"pd": pd, "np": np}  # Allow only Pandas & NumPy
            safe_locals = {"df": df}  # Only allow the DataFrame inside eval()
            
            try:
                return eval(formula, safe_globals, safe_locals)  # Secure eval execution
            except Exception as e:
                print(f"Error evaluating AI-generated KPI: {e}")
                return None

        return safe_eval
