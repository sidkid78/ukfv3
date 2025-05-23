"""
Simulation Engine - Core Layer Orchestration

Handles the execution of layers 1-10 with escalation logic,
confidence tracking, and audit logging.
"""
from typing import List, Dict, Any, Optional, AsyncGenerator
from dataclasses import dataclass
from backend.core.layers.base import BaseLayer
import logging
import hashlib
import json
from datetime import datetime
import uuid
from backend.core.layers import (
    Layer1, Layer2, Layer3, Layer4, Layer5,
    Layer6, Layer7, Layer8, Layer9, Layer10
)
from pathlib import Path
from backend.services.knowledge_expert_service import KnowledgeExpertService
import time
import psutil
LOG_DIR = Path("logs")


logger = logging.getLogger(__name__)


@dataclass
class LayerResult:
    layer_id: int
    output: Any
    confidence: float
    requires_escalation: bool = False
    audit_log: Optional[Dict] = None
    performance_metrics: Optional[Dict] = None # Made optional for now

class SimulationEngine:
    """Orchestrates the execution of simulation layers 1-10"""
    
    def __init__(self, simulation_id: str = None):
        self.layers: List[BaseLayer] = []
        self.current_context: Dict = {}
        self.audit_logs: List[Dict] = []
        self.conversation_logs: List[Dict] = []
        self.simulation_id = simulation_id or str(uuid.uuid4())
    
    async def _execute_layer(self, layer: BaseLayer) -> LayerResult:
        """Execute a single layer and return its result"""
        logger.info(f"Executing Layer {layer.layer_id}")
        
        start_time = time.perf_counter()
        # For simplicity in environments where psutil might not be fully configured or for basic tests,
        # let's make detailed performance metrics optional or conditional.
        cpu_start = 0
        mem_start = 0
        process = psutil.Process()
        if process:
            cpu_start = process.cpu_percent()
            mem_start = process.memory_info().rss
        
        try:
            layer_output = await layer.execute(self.current_context) # Assuming layer.execute is async
            
            audit_log = self._create_audit_log(layer)
            self.current_context = {**self.current_context, **layer_output}

            perf_metrics = None
            if process:
                perf_metrics = {
                    "execution_time_sec": time.perf_counter() - start_time,
                    "cpu_usage": process.cpu_percent() - cpu_start,
                    "memory_delta_bytes": process.memory_info().rss - mem_start
                }
            
            return LayerResult(
                layer_id=layer.layer_id,
                output=layer_output,
                confidence=layer_output.get('confidence', 0.0),
                requires_escalation=layer_output.get('requires_escalation', False), # Get from layer_output
                audit_log=audit_log,
                performance_metrics=perf_metrics
            )
        except Exception as e:
            self._log_error(layer.layer_id, str(e))
            # Ensure requires_escalation is set to False or a sensible default on error
            # and confidence is 0, or handle as per specific error policy.
            return LayerResult(
                layer_id=layer.layer_id,
                output={"error": str(e)},
                confidence=0.0,
                requires_escalation=False, # Default on error, or could be True based on policy
                audit_log=self._create_audit_log(layer, error=True),
                performance_metrics=None
            )
    
    def _create_audit_log(self, layer: BaseLayer, error: bool = False) -> Dict:
        """Create an audit log entry for the layer execution"""
        return {
            "timestamp": datetime.now().isoformat(),
            "layer_id": layer.layer_id,
            "execution_time": datetime.now().isoformat(), # Consider if this should be duration
            "context_snapshot": self._sanitize_context(self.current_context) if not error else {"error_context": True},
            "hash": hashlib.sha256(
                json.dumps(self.current_context, sort_keys=True).encode()
            ).hexdigest() if not error else "error_hash"
        }
    
    def _log_layer_result(self, result: LayerResult):
        """Log the layer execution result"""
        log_entry = {
            **(result.audit_log or {}), # Ensure audit_log is not None
            "result": "success" if result.confidence > 0 else "error", # Simplified success/error
            "confidence": result.confidence
        }
        if result.output and "error" in result.output:
            log_entry["error_details"] = result.output["error"]
        self.audit_logs.append(log_entry)

    
    def _log_error(self, layer_id: int, error: str): # This might be redundant if _execute_layer handles it
        """Log a layer execution error"""
        # This is now handled by _log_layer_result based on LayerResult content
        pass # logger.error(f"Error in layer {layer_id}: {error}") already logs it.

    
    async def _handle_escalation(self, result: LayerResult):
        """Handle layer escalation with concrete actions"""
        escalation_actions = {
            2: lambda: self._escalate_to_knowledge_expert(result),
            # Add other layer-specific handlers
        }
        
        handler = escalation_actions.get(result.layer_id)
        if handler:
            await handler()
        else:
            logger.warning(f"No specific escalation handler for layer {result.layer_id}, or requires_escalation was false in layer output.")

    async def _escalate_to_knowledge_expert(self, result):
        """Example escalation implementation"""
        # Ensure KnowledgeExpertService.consult is async if it involves I/O
        expert_response = await KnowledgeExpertService.consult(
            context=self.current_context,
            confidence=result.confidence
        )
        self.current_context.update(expert_response)
    
    def _sanitize_context(self, context: Dict) -> Dict:
        """Create a sanitized version of the context for logging"""
        # Simplified sanitization
        return {k: str(v)[:200] for k, v in context.items()}


    def _save_conversation_logs(self):
        """Save conversation logs to file"""
        LOG_DIR.mkdir(exist_ok=True)
        log_file = LOG_DIR / f"conversation_{self.simulation_id}.json"
        # Ensure conversation_logs is populated if this is to be used.
        # For now, focusing on audit_logs for layer progress.
        # If self.conversation_logs is the target, ensure it's filled.
        # This example will save audit_logs as a stand-in for conversation_logs for demo.
        data_to_save = self.audit_logs if not self.conversation_logs else self.conversation_logs
        with open(log_file, 'w') as f:
            json.dump({
                "simulation_id": self.simulation_id,
                "timestamp": datetime.now().isoformat(),
                "logs": data_to_save 
            }, f, indent=2)
        logger.info(f"Saved logs to {log_file}")

    def register_layer(self, layer: BaseLayer):
        """Register a layer implementation"""
        if not isinstance(layer, BaseLayer):
            raise TypeError(f"Expected BaseLayer instance, got {type(layer)}")
        self.layers.append(layer)

    async def execute_simulation_iteratively(self, initial_context: Dict) -> AsyncGenerator[LayerResult, None]:
        """Execute all layers in sequence, yielding results after each layer."""
        self.current_context = self._sanitize_context(initial_context)
        if not self.layers:
            logger.warning("No layers registered in the simulation engine.")
            return

        for layer in self.layers:
            layer_result = await self._execute_layer(layer)
            self._log_layer_result(layer_result)
            
            yield layer_result # Yield the result for real-time processing
            
            if layer_result.requires_escalation:
                await self._handle_escalation(layer_result)
            
            if "error" in layer_result.output: # Stop processing if a layer errored significantly
                logger.error(f"Stopping simulation due to error in layer {layer.layer_id}.")
                break
        
        self._save_conversation_logs() # Save logs at the end