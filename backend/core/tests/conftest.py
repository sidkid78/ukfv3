"""
Pytest configuration and fixtures for simulation tests.
"""
import pytest
from backend.core.simulation_engine import SimulationEngine
from backend.core.layers.base import BaseLayer
from backend.core.layers import Layer1, Layer2, Layer3, Layer4, Layer5, Layer6, Layer7, Layer8, Layer9, Layer10

@pytest.fixture
def simulation_engine():
    """Fixture providing a configured simulation engine."""
    engine = SimulationEngine()
    # Register all layers
    engine.register_layer(Layer1())
    engine.register_layer(Layer2())
    engine.register_layer(Layer3())
    engine.register_layer(Layer4())
    engine.register_layer(Layer5())
    engine.register_layer(Layer6())
    engine.register_layer(Layer7())
    engine.register_layer(Layer8())
    engine.register_layer(Layer9())
    engine.register_layer(Layer10())
    return engine

@pytest.fixture
def base_context():
    """Fixture providing a basic simulation context."""
    return {
        "query": "test query",
        "session_id": "test_session"
    } 