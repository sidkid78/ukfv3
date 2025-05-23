"""
Tests for the simulation engine core functionality.
"""
from backend.core.simulation_engine import SimulationEngine
import pytest

class TestSimulationEngine:
    """Test suite for SimulationEngine class."""
    
    def test_initialization(self):
        """Test that the engine initializes with empty layers."""
        engine = SimulationEngine()
        assert len(engine.layers) == 0
        assert isinstance(engine.current_context, dict)
        assert len(engine.audit_logs) == 0
    
    @pytest.mark.asyncio
    async def test_layer_registration(self, simulation_engine):
        """Test that layers are properly registered."""
        assert len(simulation_engine.layers) == 10
        assert simulation_engine.layers[0].layer_id == 1
        assert simulation_engine.layers[-1].layer_id == 10
    
    @pytest.mark.asyncio
    async def test_full_execution(self, simulation_engine, base_context):
        """Test executing all layers with a basic context."""
        result = await simulation_engine.execute(base_context)
        
        # Verify final context contains expected keys
        assert "layer1_processed" in result
        assert "graph_initialized" in result
        assert "reasoning_completed" in result
        assert "advanced_reasoning_completed" in result
        assert "validation_completed" in result
        assert "decision_completed" in result
        assert "execution_completed" in result
        assert "adaptation_completed" in result
        assert "verification_completed" in result
        assert "finalized" in result
        
        # Verify audit logs were created
        assert len(simulation_engine.audit_logs) == 10
        for log in simulation_engine.audit_logs:
            assert "layer_id" in log
            assert "timestamp" in log
            assert "execution_time" in log 