"""
Base Layer - Abstract class defining the layer interface
"""
from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseLayer(ABC):
    """
    Abstract base class for all simulation layers
    
    Attributes:
        layer_id: int - The layer's position in the stack (1-10)
        requires_escalation: bool - Whether this layer can trigger escalation
    """
    
    def __init__(self, layer_id: int):
        self.layer_id = layer_id
        self.requires_escalation = False
    
    @abstractmethod
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the layer's core logic
        
        Args:
            context: The current simulation context
            
        Returns:
            Updated context dictionary
        """
        pass 