"""
Tests for layer implementations and base layer functionality.
"""
from backend.core.layers import BaseLayer, Layer1, Layer2, Layer3
import pytest

class TestBaseLayer:
    """Test suite for BaseLayer functionality."""
    
    def test_abstract_method(self):
        """Test that BaseLayer can't be instantiated directly."""
        with pytest.raises(TypeError):
            BaseLayer(layer_id=1)
    
    def test_concrete_layer(self):
        """Test that concrete layers implement required methods."""
        layer = Layer1()
        assert hasattr(layer, 'execute')
        assert callable(layer.execute)

class TestLayer1:
    """Test suite for Layer1 (Input Processing)."""
    
    @pytest.mark.asyncio
    async def test_execute_valid_input(self):
        """Test Layer1 with valid input context."""
        layer = Layer1()
        context = {"query": "test", "session_id": "123"}
        result = await layer.execute(context)
        
        assert "layer1_processed" in result
        assert "normalized" in result
        assert result["normalized"]["query"] == "test"
    
    @pytest.mark.asyncio
    async def test_execute_missing_field(self):
        """Test Layer1 with missing required field."""
        layer = Layer1()
        context = {"query": "test"}  # Missing session_id
        
        with pytest.raises(ValueError):
            await layer.execute(context)

class TestLayer2:
    """Test suite for Layer2 (Knowledge Graph)."""
    
    @pytest.mark.asyncio
    async def test_execute_valid_input(self):
        """Test Layer2 with valid input from Layer1."""
        layer = Layer2()
        context = {
            "layer1_processed": True,
            "normalized": {"query": "test"}
        }
        result = await layer.execute(context)
        
        assert "graph_initialized" in result
        assert "entities" in result and "facts" in result
        
    @pytest.mark.asyncio
    async def test_execute_missing_prerequisite(self):
        """Test Layer2 without Layer1 processing."""
        layer = Layer2()
        context = {"query": "test"}
        
        with pytest.raises(ValueError):
            await layer.execute(context) 