from cryptography.fernet import Fernet
import sys
import os
# Force Pytest to see the src directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import tempfile
import pytest

# Generate a valid dummy Fernet key BEFORE the FastAPI app initializes
DUMMY_FERNET_KEY = Fernet.generate_key().decode()

@pytest.fixture(scope="session", autouse=True)
def setup_test_env():
    """Sets up a safe, isolated environment for all tests."""
    os.environ["AIRFLOW__CORE__FERNET_KEY"] = DUMMY_FERNET_KEY
    os.environ["MODEL_SERVE_MODE"] = "local"
    
    # Create a temporary database for testing so we don't corrupt production logs
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        temp_db = f.name
        
    os.environ["DB_PATH"] = temp_db
    
    yield  # Tests run here
    
    # Delete the temporary database after tests finish
    if os.path.exists(temp_db):
        os.unlink(temp_db)

@pytest.fixture(scope="session")
def api_client(setup_test_env):
    """Create FastAPI test client with startup/shutdown lifecycle."""
    from fastapi.testclient import TestClient
    from main import app
    
    # Using 'with' forces FastAPI to run the @app.on_event("startup") code
    with TestClient(app) as client:
        yield client