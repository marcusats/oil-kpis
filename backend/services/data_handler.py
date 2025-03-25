import pandas as pd
import json
import os

class DataHandler:
    def __init__(self, source_type, source, mapping_path):
        self.source_type = source_type
        self.source = source
        self.mapping_path = mapping_path
        
        # Cargar mapeo de columnas desde archivo JSON
        try:
            with open(mapping_path, "r") as f:
                self.mapping = json.load(f)
        except Exception as e:
            self.mapping = {}
            print(f"Error al cargar archivo de mapeo: {str(e)}")
    
    def load_data(self):
        """Cargar datos desde fuente especificada"""
        try:
            if self.source_type.lower() == "excel":
                # Verificar si el archivo existe
                if not os.path.exists(self.source):
                    return f"Archivo no encontrado: {self.source}"
                
                # Intentar cargar todas las hojas
                try:
                    # Primero intentar cargar la primera hoja
                    df = pd.read_excel(self.source)
                    print(f"Columnas cargadas: {df.columns.tolist()}")
                    return df
                except Exception as excel_error:
                    # Si falla, intentar con parámetros específicos
                    print(f"Error al cargar Excel normalmente: {str(excel_error)}")
                    try:
                        # Intentar con engine='openpyxl'
                        df = pd.read_excel(self.source, engine='openpyxl')
                        print(f"Cargado con openpyxl. Columnas: {df.columns.tolist()}")
                        return df
                    except Exception as openpyxl_error:
                        print(f"Error con openpyxl: {str(openpyxl_error)}")
                        try:
                            # Última opción: especificar sheet_name='Sheet1'
                            df = pd.read_excel(self.source, sheet_name=0)
                            print(f"Cargado con sheet_name=0. Columnas: {df.columns.tolist()}")
                            return df
                        except Exception as sheet_error:
                            return f"Error al cargar Excel: {str(sheet_error)}"
            elif self.source_type.lower() == "csv":
                if not os.path.exists(self.source):
                    return f"Archivo no encontrado: {self.source}"
                return pd.read_csv(self.source)
            elif self.source_type.lower() == "json":
                if not os.path.exists(self.source):
                    return f"Archivo no encontrado: {self.source}"
                with open(self.source, "r") as f:
                    data = json.load(f)
                return pd.DataFrame(data)
            elif self.source_type.lower() == "db":
                # Implementar conexión a base de datos (ejemplo con SQLAlchemy)
                # from sqlalchemy import create_engine
                # engine = create_engine(self.source)
                # return pd.read_sql(self.query, engine)
                return f"Conexiones a bases de datos aún no implementadas"
            elif self.source_type.lower() == "api":
                # Implementar obtención de datos de API
                # import requests
                # response = requests.get(self.source)
                # data = response.json()
                # return pd.DataFrame(data)
                return f"Fuentes de datos API aún no implementadas"
            else:
                return f"Tipo de fuente no soportado: {self.source_type}"
        except Exception as e:
            return f"Error al cargar datos: {str(e)}" 