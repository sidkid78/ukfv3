"""
Layer 1 - Initial Input Processing

Responsible for:
- Validating and normalizing simulation input
- Setting up initial context structure
- Basic input sanitization
"""
from typing import Dict, Any
from .base import BaseLayer

class Layer1(BaseLayer):
    """First simulation layer - input processing"""
    
    def __init__(self):
        super().__init__(layer_id=1)
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process initial simulation input"""
        # Validate required fields
        required = ['query', 'session_id']
        for field in required:
            if field not in context:
                raise ValueError(f"Missing required field: {field}")
        
        # Normalize input structure
        processed = {
            'raw_input': context,
            'normalized': {
                'query': str(context['query']).strip(),
                'session_id': str(context['session_id']),
                'metadata': context.get('metadata', {})
            },
            'layer1_processed': True
        }
        
        # Merge with existing context
        return {**context, **processed} 