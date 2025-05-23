"""
Simulation Engine - Core Layer Orchestration

Handles the execution of layers 1-10 with escalation logic,
confidence tracking, and audit logging.
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from backend.core.layers.base import BaseLayer
import logging
import hashlib
import json
from datetime import datetime
from backend.core.layers import (
    Layer1, Layer2, Layer3, Layer4, Layer5,
    Layer6, Layer7, Layer8, Layer9, Layer10
)
from pathlib import Path
LOG_DIR = Path("logs")


logger = logging.getLogger(__name__)


@dataclass
class LayerResult:
    layer_id: int
    output: Any
    confidence: float
    requires_escalation: bool = False
    audit_log: Optional[Dict] = None

class SimulationEngine:
    """Orchestrates the execution of simulation layers 1-10"""
    
    def __init__(self):
        self.layers: List[BaseLayer] = [
            Layer1(), Layer2(), Layer3(), Layer4(), Layer5(),
            Layer6(), Layer7(), Layer8(), Layer9(), Layer10()
        ]
        self.current_context: Dict = {}
        self.audit_logs: List[Dict] = []
    
    async def run_simulation(self, initial_context: Dict) -> Dict:
        """Execute the full simulation pipeline"""
        self.current_context = initial_context
        
        for layer in self.layers:
            try:
                layer_result = await self._execute_layer(layer)
                self._log_layer_result(layer_result)
                
                if layer_result.requires_escalation:
                    await self._handle_escalation(layer_result)
                
            except Exception as e:
                logger.error(f"Layer {layer.layer_id} failed: {str(e)}")
                self._log_error(layer.layer_id, str(e))
                raise
        
        return {
            "final_output": self.current_context,
            "audit_logs": self.audit_logs,
            "conversation_logs": self.conversation_logs
        }
    
    async def _execute_layer(self, layer: BaseLayer) -> LayerResult:
        """Execute a single layer and return its result"""
        logger.info(f"Executing Layer {layer.layer_id}")
        
        output = await layer.execute(self.current_context)
        self.current_context = output
        
        conversation_log = output.get('conversation_log', [])
        
        return LayerResult(
            layer_id=layer.layer_id,
            output=output,
            confidence=output.get('confidence', 0.0),
            requires_escalation=layer.requires_escalation,
            audit_log=self._create_audit_log(layer),
            conversation_log=conversation_log
        )
    
    def _create_audit_log(self, layer: BaseLayer) -> Dict:
        """Create an audit log entry for the layer execution"""
        return {
            "timestamp": datetime.now().isoformat(),
            "layer_id": layer.layer_id,
            "context_snapshot": self._sanitize_context(self.current_context),
            "hash": hashlib.sha256(
                json.dumps(self.current_context, sort_keys=True).encode()
            ).hexdigest()
        }
    
    def _log_layer_result(self, result: LayerResult):
        """Log the layer execution result"""
        self.audit_logs.append({
            **result.audit_log,
            "result": "success",
            "confidence": result.confidence
        })
    
    def _log_error(self, layer_id: int, error: str):
        """Log a layer execution error"""
        self.audit_logs.append({
            "timestamp": datetime.now().isoformat(),
            "layer_id": layer_id,
            "result": "error",
            "error": error
        })
    
    async def _handle_escalation(self, result: LayerResult):
        """Handle layer escalation requirements"""
        logger.warning(
            f"Layer {result.layer_id} requires escalation (confidence: {result.confidence:.1%})"
        )
        # Placeholder for actual escalation logic
        
    def _sanitize_context(self, context: Dict) -> Dict:
        """Create a sanitized version of the context for logging"""
        return {
            k: v for k, v in context.items() 
            if not k.endswith('_completed') and not isinstance(v, (list, dict))
        }
    
    def _save_conversation_logs(self):
        """Save conversation logs to file"""
        log_file = LOG_DIR / f"conversation_{self.simulation_id}.json"
        with open(log_file, 'w') as f:
            json.dump({
                "simulation_id": self.simulation_id,
                "timestamp": datetime.now().isoformat(),
                "logs": self.conversation_logs
            }, f, indent=2)
        logger.info(f"Saved conversation logs to {log_file}")