"""
Quick server startup script
"""

import uvicorn
import os

if __name__ == "__main__":
    # Change to backend directory
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(backend_dir)
    
    print("Starting Financial Transaction Categorisation API Server...")
    print("API will be available at: http://localhost:8000")
    print("API documentation at: http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop the server\n")
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

