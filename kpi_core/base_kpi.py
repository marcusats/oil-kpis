# kpi_core/base_kpi.py

class KPI:
    def __init__(self, name, formula, category, unit, description="", ai_suggestions=None):
        """
        Base class for KPI calculations.
        :param name: KPI Name
        :param formula: Function to calculate KPI
        :param category: Category of KPI (e.g., 'Upstream', 'Midstream')
        :param unit: Measurement unit
        :param description: Explanation of the KPI
        :param ai_suggestions: AI-recommended benchmarks (optional)
        """
        self.name = name
        self.formula = formula
        self.category = category
        self.unit = unit
        self.description = description
        self.ai_suggestions = ai_suggestions or {}

    def calculate(self, **kwargs):
        """Calculate KPI based on provided arguments."""
        try:
            return self.formula(**kwargs)
        except Exception as e:
            return f"Error calculating {self.name}: {str(e)}"

    def __repr__(self):
        return f"{self.name} ({self.unit}): {self.description}"


