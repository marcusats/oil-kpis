import json

def get_industry_context(kpi_name):
    """Return industry-specific context for KPI interpretation"""
    if kpi_name == "ResumenEstadístico":
        return """
        En el sector petrolero, las estadísticas descriptivas son cruciales para comprender la distribución de variables como:
        - Producción diaria de barriles
        - Costos operativos por pozo
        - Eficiencia de extracción
        - Métricas ambientales y de seguridad
        
        Valores atípicos pueden indicar problemas operativos o oportunidades de optimización.
        """
    elif kpi_name == "MatrizCorrelación":
        return """
        Las correlaciones en el sector petrolero pueden revelar:
        - Factores que afectan la producción de petróleo
        - Relaciones entre variables operativas y financieras
        - Dependencias entre parámetros técnicos
        - Oportunidades para optimizar procesos y reducir costos
        
        Correlaciones fuertes pueden indicar causalidad o dependencia que debería ser investigada.
        """
    elif kpi_name == "DistribuciónNumérica":
        return """
        Las distribuciones numéricas en datos petroleros ayudan a:
        - Identificar rangos normales de operación
        - Detectar ineficiencias o anomalías
        - Establecer valores de referencia para KPIs
        - Planificar capacidad y recursos
        
        La forma de la distribución puede revelar desequilibrios o sesgos en los procesos operativos.
        """
    elif kpi_name == "TendenciaTemporal":
        return """
        El análisis temporal en el sector petrolero es fundamental para:
        - Identificar patrones estacionales en producción o consumo
        - Detectar tendencias a largo plazo en eficiencia o rendimiento
        - Evaluar el impacto de eventos externos o cambios operativos
        - Proyectar resultados futuros y planificar inversiones
        
        Cambios bruscos en tendencias pueden indicar problemas o mejoras significativas.
        """
    elif kpi_name == "ConteoPorCategoría":
        return """
        El análisis categórico en el sector petrolero permite:
        - Segmentar activos por tipo, región o rendimiento
        - Identificar distribuciones de problemas o incidentes
        - Comparar rendimiento entre diferentes categorías operativas
        - Asignar recursos más eficientemente basándose en categorías de mayor impacto
        
        Categorías con frecuencias inusuales pueden requerir atención especial.
        """
    else:
        return """
        El análisis de KPIs en el sector petrolero es crítico para optimizar operaciones, 
        reducir costos, mejorar la seguridad y cumplir objetivos ambientales.
        """

def get_kpi_interpretation_prompt(kpi_name, kpi_result, context):
    """Generate a prompt for KPI interpretation"""
    industry_context = get_industry_context(kpi_name)
    
    # Extract dataset info and text summary from the new context structure
    dataset_info = "No hay información disponible sobre el conjunto de datos."
    
    if isinstance(context, dict):
        # Extract basic dataset information
        rows = context.get("dataset_info", {}).get("rows", "desconocido")
        columns = context.get("dataset_info", {}).get("columns", "desconocido")
        dataset_info = f"El conjunto de datos tiene {rows} filas y {columns} columnas.\n\n"
        
        # Add text summary if available
        if "text_summary" in context and isinstance(context["text_summary"], list):
            dataset_info += "\n".join(context["text_summary"])
    
    prompt = f"""
    Eres un analista de datos experto especializado en la interpretación de KPIs en el sector petrolero, con experiencia específica en pozos petroleros, estimulaciones, y operaciones de campo.
    
    Se te ha proporcionado el siguiente KPI: "{kpi_name}"
    
    Los resultados del KPI son: {json.dumps(kpi_result, indent=2, ensure_ascii=False)}
    
    Información sobre el conjunto de datos:
    {dataset_info}
    
    {industry_context}
    
    IMPORTANTE: Debes proporcionar un análisis ESPECÍFICO basado en los datos reales mostrados, NO una interpretación genérica. Refiérete a valores concretos, tendencias específicas y hallazgos particulares de ESTOS datos.
    
    Por favor, proporciona:
    
    1. Una interpretación clara y detallada de lo que muestran estos resultados específicos (2-3 párrafos)
       - CITA VALORES NUMÉRICOS ESPECÍFICOS de los resultados mostrados
       - Identifica las tendencias más importantes y patrones relevantes para este conjunto de datos
       - Explica el significado de estos valores específicos en el contexto del sector petrolero
       - Menciona cualquier anomalía, valor atípico o patrón inusual que detectes
    
    2. Los 3-5 insights o hallazgos más importantes de ESTOS DATOS ESPECÍFICOS (en formato de lista)
       - Cada insight debe citar valores/resultados específicos del análisis (usa números concretos)
       - Explica por qué cada hallazgo es relevante para las operaciones petroleras
       - Califica la importancia o impacto potencial de cada hallazgo
       - Relaciona los hallazgos con posibles causas operativas o técnicas en el sector petrolero
    
    3. Acciones recomendadas basadas en ESTOS DATOS ESPECÍFICOS (en formato de lista)
       - Propone 3-5 acciones concretas que respondan directamente a los hallazgos identificados
       - Para cada acción, explica cómo abordaría los problemas o aprovecharía las oportunidades identificadas
       - Sugiere métricas específicas para monitorear el impacto de cada acción recomendada
       - Indica la prioridad relativa de cada acción (alta/media/baja)
    
    Utiliza un lenguaje profesional pero accesible, evitando jerga técnica excesiva.
    RECUERDA: Debes referirte explícitamente a los datos mostrados, citando valores concretos y evitando interpretaciones genéricas o plantillas.
    Enmarca tu análisis en el contexto específico de estos datos y la industria petrolera.
    """
    
    return prompt

def get_chat_query_prompt(user_query, data_context, query_specific_data=None):
    """Generate a prompt for chat query"""
    prompt = f"""
    Eres un analista de datos experto especializado en el sector petrolero, con profundo conocimiento en operaciones de campo, estimulaciones de pozos, y análisis de producción petrolera.
    
    Tienes acceso a un conjunto de datos con la siguiente información:
    
    {data_context}
    
    Por favor, responde a la siguiente pregunta del usuario sobre los datos:
    "{user_query}"
    
    """
    
    # Add specific data relevant to the query
    if query_specific_data and len(query_specific_data) > 0:
        prompt += f"""
        Aquí hay información específica relacionada con la consulta:
        
        {json.dumps(query_specific_data, indent=2, ensure_ascii=False)}
        """
    
    prompt += """
    INSTRUCCIONES IMPORTANTES:
    
    1. Proporciona un análisis basado ESPECÍFICAMENTE en los datos provistos, no generalidades
    2. Cita valores específicos, estadísticas y tendencias reales de los datos
    3. Relaciona tu respuesta con el contexto de la industria petrolera (operaciones, estimulaciones, producción)
    4. Estructura tu respuesta para máxima claridad:
       - Primero, un resumen directo que responda a la pregunta (1-2 párrafos)
       - Luego, los puntos clave o hallazgos respaldados por datos específicos
       - Finalmente, una breve conclusión con implicaciones o recomendaciones
    5. Usa formato Markdown para mejorar la legibilidad:
       - Utiliza tablas para comparar datos cuando sea relevante
       - Usa listas para enumerar hallazgos o recomendaciones
       - Emplea encabezados para organizar secciones
    6. Menciona explícitamente cualquier limitación en los datos proporcionados
    7. Siempre que sea posible, ofrece perspectivas sobre las implicaciones operativas o económicas
    
    Responde en español, usando lenguaje técnico pero accesible. Evita respuestas vagas o generales - enfócate en LOS DATOS ESPECÍFICOS.
    """
    
    return prompt

def get_fallback_interpretation(kpi_name):
    """Get fallback interpretation for when AI fails"""
    if kpi_name == "MatrizCorrelación":
        return f"""
        # Interpretación de Matriz de Correlación
        
        La matriz de correlación muestra la relación entre diferentes variables numéricas en el dataset. En el sector petrolero, estas relaciones pueden indicar factores que influyen en la producción, eficiencia o costos operativos.
        
        ## Insights principales
        
        * Las correlaciones con valores cercanos a 1 indican una fuerte relación positiva entre variables, lo que puede señalar factores que se mueven juntos
        * Las correlaciones con valores cercanos a -1 indican una fuerte relación negativa, sugiriendo factores que tienen un comportamiento inverso
        * Las correlaciones cercanas a 0 indican poca o ninguna relación, lo que ayuda a descartar dependencias aparentes
        * El análisis de estas relaciones puede ayudar a identificar variables clave que afectan los resultados operativos
        
        ## Recomendaciones
        
        * Explore en profundidad las variables con correlaciones fuertes (>0.7) para identificar posibles relaciones causales
        * Considere utilizar las variables altamente correlacionadas para modelos predictivos de producción o eficiencia
        * Investigue las correlaciones negativas fuertes para identificar posibles compensaciones en procesos operativos
        * Documente estas relaciones como parte de los indicadores de rendimiento para monitoreo continuo
        * Realice análisis adicionales para confirmar si las correlaciones fuertes representan relaciones causales o son coincidenciales
        """
    else:
        return f"""
        # Interpretación de {kpi_name}
        
        Lo sentimos, no se pudo generar una interpretación automática de este KPI en este momento.
        
        ## Recomendaciones
        
        * Revise los datos manualmente para identificar patrones y relaciones relevantes
        * Consulte con un analista de datos especializado en el sector petrolero para una interpretación detallada
        * Verifique que los datos de entrada sean correctos, completos y representativos de sus operaciones
        * Considere visualizar estos datos con herramientas especializadas para facilitar su interpretación
        """ 