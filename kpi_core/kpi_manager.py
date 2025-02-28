# Not verified
# kpi_core/kpi_manager.py

import json
import importlib
from kpi_core.base_kpi import KPI
from kpi_modules.upstream.well_stimulation.stimulation_chemicals import get_kpis

class KPIManager:
    def __init__(self):
        self.kpis = {}

    def load_kpi_definitions(self, filepath):
        """Load KPI definitions from JSON"""
        with open(filepath, "r") as f:
            kpi_data = json.load(f)
            category = kpi_data["category"]
            subcategory = kpi_data["subcategory"]
            
            for kpi in kpi_data["kpis"]:
                self.kpis[f"{category} - {subcategory} - {kpi['name']}"] = KPI(
                    name=kpi["name"],
                    formula=eval(f"lambda {', '.join(kpi['formula'].split(' ')[::2])}: {kpi['formula']}"),  
                    category=f"{category} - {subcategory}",
                    unit=kpi["unit"],
                    description=kpi["description"],
                    ai_suggestions=kpi.get("ai_suggestions", {})
                )

    def load_all_kpis(self):
        """Load both JSON-defined and Python-implemented KPIs"""
        self.load_kpi_definitions("kpi_definitions/stimulation_chemicals.json")
        for kpi in get_kpis():
            self.kpis[kpi.name] = kpi  # Load KPIs from stimulation_chemicals.py

    def calculate_kpi(self, kpi_name, **kwargs):
        """Calculate KPI by name"""
        if kpi_name in self.kpis:
            return self.kpis[kpi_name].calculate(**kwargs)
        return f"KPI '{kpi_name}' not found."

    def list_kpis(self):
        """List all available KPIs"""
        return list(self.kpis.keys())

# Example Usage
if __name__ == "__main__":
    manager = KPIManager()
    manager.load_all_kpis()

    print("Available KPIs:", manager.list_kpis())

    # Example: Calculate Chemical Efficiency
    result = manager.calculate_kpi("Chemical Efficiency", chemical_used=100, production_increase=500)
    print("Chemical Efficiency:", result)

    # Example: Calculate Cost per Barrel
    result = manager.calculate_kpi("Cost Per Barrel", cost=10000, barrels_recovered=500)
    print("Cost Per Barrel:", result)
