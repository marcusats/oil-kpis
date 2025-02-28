# kpi_modules/upstream/well_stimulation/stimulation_chemicals.py
from kpi_core.base_kpi import KPI

# Define KPI calculation functions
def chemical_efficiency(chemical_used, production_increase):
    """Efficiency = Production Increase / Chemical Used"""
    return production_increase / chemical_used if chemical_used else 0

def cost_per_barrel(cost, barrels_recovered):
    """Cost per Barrel = Total Cost / Barrels Recovered"""
    return cost / barrels_recovered if barrels_recovered else 0

# Register KPIs for Stimulation Chemicals
STIMULATION_CHEMICALS_KPIS = [
    KPI("Chemical Efficiency", chemical_efficiency, "bbl/kg", "Effectiveness of stimulation chemicals."),
    KPI("Cost Per Barrel", cost_per_barrel, "$/bbl", "Cost-effectiveness of chemical stimulation."),
]

def get_kpis():
    """Return all KPIs related to Stimulation Chemicals"""
    return STIMULATION_CHEMICALS_KPIS
