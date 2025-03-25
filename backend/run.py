import os
import sys

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the app and run it
from app import app

if __name__ == "__main__":
    import uvicorn
    
    # Check if we're in debug mode
    debug = os.environ.get("DEBUG", "False").lower() in ("true", "1", "t")
    
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=debug) 