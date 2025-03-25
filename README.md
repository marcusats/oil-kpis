# Oil KPIs Analysis Platform

## Project Overview

The Oil KPIs Analysis Platform is a comprehensive data analytics solution designed for the petroleum industry. This full-stack application enables petroleum engineers and analysts to upload, process, visualize, and interpret key performance indicators (KPIs) from operational data. The platform combines robust data processing capabilities with AI-powered interpretations to deliver actionable insights.

## System Architecture

The application follows a client-server architecture with clear separation of concerns:

```
┌────────────────┐       ┌─────────────────┐       ┌──────────────┐
│   Frontend     │       │    Backend      │       │   External   │
│  (Streamlit)   │<─────>│   (FastAPI)     │<─────>│   Services   │
└────────────────┘       └─────────────────┘       └──────────────┘
                                 │
                                 │
                         ┌───────┴───────┐
                         │   Database    │
                         │   Storage     │
                         └───────────────┘
```

- **Frontend**: Built with Streamlit, providing an interactive web interface for data visualization and user interaction
- **Backend**: Powered by FastAPI, handling data processing, KPI calculations, and AI integrations
- **External Services**: Integration with OpenAI for AI-powered data interpretations
- **Storage**: Local file system for uploaded datasets and temporary storage

## Backend Implementation

### Core Components

1. **FastAPI Application**: The main application server that handles HTTP requests, middleware, and routing
2. **Data Processing Module**: Responsible for loading, transforming, and managing datasets
3. **KPI Calculation Engine**: Implements various KPI calculations for petroleum data analysis
4. **AI Integration**: Connects with OpenAI's API to provide intelligent interpretations of KPIs

### Key API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/upload-file` | POST | Handles file uploads and saves them to a temporary location |
| `/load-data` | POST | Loads a dataset from a specified source |
| `/raw-data` | GET | Retrieves raw data from the loaded dataset |
| `/available-kpis` | GET | Lists all available KPIs with descriptions |
| `/calculate-kpi/{kpi_name}` | GET | Calculates a specific KPI with optional column selection |
| `/interpret-kpi` | POST | Generates AI interpretation of KPI results |
| `/chat-query` | POST | Provides AI responses to natural language queries about the data |
| `/available-filters` | GET | Returns columns that can be used for filtering data |
| `/unique-values/{column}` | GET | Gets unique values for a specific column (for filter options) |

### KPI Calculation System

The platform offers several predefined KPIs for petroleum data analysis:

1. **ResumenEstadístico (Statistical Summary)**: Basic statistical measures for numerical columns, including mean, median, outlier detection, and distribution characteristics.

2. **MatrizCorrelación (Correlation Matrix)**: Identifies relationships between numerical variables, highlighting strong correlations that may indicate important operational connections.

3. **DistribuciónNumérica (Numerical Distribution)**: Visualizes the distribution of numerical variables through histograms and statistical metrics.

4. **TendenciaTemporal (Time Trend Analysis)**: Analyzes how variables change over time, identifying trends, seasonality, and potential forecasting opportunities.

5. **ConteoPorCategoría (Category Count)**: Examines categorical variables by counting frequencies and calculating percentages.

6. **Production Analysis**: Specialized analysis focused on production metrics in petroleum operations.

Each KPI calculation includes comprehensive error handling, JSON serialization safeguards, and formatted outputs designed to integrate seamlessly with the frontend visualization components.

### AI Integration

The backend integrates with OpenAI's API to provide two key AI capabilities:

1. **KPI Interpretation**: Analyzes calculated KPIs and provides expert interpretations relevant to petroleum operations.

2. **Natural Language Querying**: Allows users to ask questions about their data in plain language and receive AI-generated responses with insights.

The AI component is implemented with failover mechanisms to handle API errors and provide fallback responses when needed. It utilizes OpenAI's gpt-3.5-turbo model with petroleum industry-specific prompting to generate relevant insights.

## Frontend Implementation

### User Interface Components

The frontend is implemented with Streamlit and organized into several key sections:

1. **Data Management**: Tools for uploading, selecting, and filtering datasets
2. **KPI Dashboard**: Interactive visualization of calculated KPIs with customization options
3. **AI Interpretation Panel**: Displays AI-generated interpretations of selected KPIs
4. **Query Interface**: Allows natural language querying of the dataset with AI responses
5. **Data Export**: Options to download processed data and visualization results

### Data Visualization System

The application uses a combination of Plotly and built-in Streamlit components to create interactive visualizations:

1. **Statistical Charts**: Bar charts, histograms, and box plots for numerical distributions
2. **Correlation Visualizations**: Heatmaps and network diagrams for correlation matrices
3. **Time Series Charts**: Line charts with trend lines for temporal analysis
4. **Categorical Analysis**: Bar charts and pie charts for categorical distributions
5. **Custom Indicators**: Specialized visualizations for petroleum-specific metrics

The visualization system dynamically adapts to different KPI types and data structures, with robust error handling for missing or incomplete data.

### User Workflow

1. **Data Upload and Selection**: Users begin by uploading a dataset (CSV, Excel) or selecting from existing files
2. **Data Exploration**: Basic statistics and sample data are displayed for initial understanding
3. **KPI Selection**: Users select from available KPIs and customize column selections
4. **Visualization**: Interactive charts display the calculated KPI results
5. **AI Interpretation**: Users can request AI interpretation of any KPI result
6. **Natural Language Querying**: Users can ask questions about the data in plain language
7. **Export and Sharing**: Results can be downloaded or shared as reports

## Technical Implementation Details

### Backend Technologies

- **FastAPI**: High-performance API framework for the backend server
- **Pandas & NumPy**: Data processing and numerical computations
- **SciPy**: Statistical calculations and analysis
- **OpenAI API**: AI-powered interpretations and natural language processing
- **Python-Multipart**: File upload handling
- **Pydantic**: Data validation and settings management

### Frontend Technologies

- **Streamlit**: Interactive web application framework
- **Plotly**: Advanced data visualization library
- **Matplotlib**: Additional visualization capabilities
- **Pandas**: Data manipulation and processing

### Data Processing Pipeline

1. **Data Loading**: Files are uploaded and processed into pandas DataFrames
2. **Data Validation**: Basic checks ensure data quality and compatibility
3. **Type Inference**: Column types are determined for appropriate processing
4. **KPI Calculation**: Selected KPIs are calculated on the processed data
5. **Visualization Preparation**: Data is transformed into formats suitable for visualization
6. **AI Processing**: KPI results are sent to the AI service for interpretation
7. **Response Formatting**: Results are formatted for frontend display

### Error Handling and Robustness

The application implements comprehensive error handling throughout:

1. **Input Validation**: All user inputs are validated before processing
2. **Exception Handling**: Structured try-except blocks capture and report errors
3. **Fallback Mechanisms**: Default responses when calculations or AI services fail
4. **JSON Serialization Safety**: Special handling for NaN, null, and infinite values
5. **Logging**: Detailed logging for debugging and monitoring

## Deployment and Usage

### Requirements

- Python 3.8+ environment
- Required Python packages (specified in requirements.txt)
- OpenAI API key for AI interpretation features
- Sufficient storage for dataset processing

### Configuration

The application uses environment variables for configuration:
- `OPENAI_API_KEY`: Required for AI interpretation features
- Additional configuration can be set in a `.env` file

### Running the Application

1. Install dependencies: `pip install -r requirements.txt`
2. Configure environment variables
3. Run the backend: `python backend/app.py`
4. Run the frontend: `streamlit run frontend/app.py`
5. Access the application in a web browser

## Conclusion

The Oil KPIs Analysis Platform demonstrates advanced engineering in data processing, statistical analysis, and AI integration. It provides petroleum engineers with powerful tools to gain insights from operational data, combining traditional statistical methods with cutting-edge AI interpretation capabilities. The system architecture ensures scalability, maintainability, and extensibility for future enhancements. 