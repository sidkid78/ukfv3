from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Any
import asyncio
import json

# Core simulation imports
from backend.core.simulation_engine import SimulationEngine, LayerResult
from backend.core.layers import (
    Layer1, Layer2, Layer3, Layer4, Layer5,
    Layer6, Layer7, Layer8, Layer9, Layer10
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# WebSocket manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: Dict):
        for connection in self.active_connections:
            await connection.send_json(message)

manager = ConnectionManager()



# WebSocket endpoint
@app.websocket("/ws/simulation")
async def websocket_simulation(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text() # Keep alive, handle incoming if needed
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket Error: {e}") # Log other errors
        manager.disconnect(websocket)

@app.get('/')
def read_root():
    return {"msg": "Simulation backend is up"}

# Modified simulate endpoint to use WebSocket
@app.post("/simulation/run")
async def run_simulation_endpoint(request: Request):
    initial_context = await request.json()
    query = initial_context.get("query", "No query provided") # Extract query for context

    engine = SimulationEngine() # Consider passing a simulation_id if needed

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

    final_status = "simulation_complete"
    final_output_summary = {}

    try:
        async for layer_result in engine.execute_simulation_iteratively(initial_context):
            # Broadcast progress after each layer
            await manager.broadcast({
                "type": "layer_progress",
                "layer_id": layer_result.layer_id,
                "status": "completed" if "error" not in layer_result.output else "error",
                "confidence": layer_result.confidence,
                "output": layer_result.output, # Send relevant parts of the output
                "performance": layer_result.performance_metrics
            })
            if "error" in layer_result.output:
                final_status = "simulation_error"
                final_output_summary = layer_result.output
                break # Stop on error
        
        # After the loop, the engine.current_context has the final state
        # or the state at the point of error.
        if final_status != "simulation_error":
            final_output_summary = engine.current_context

    except Exception as e:
        print(f"Error during simulation execution: {e}")
        await manager.broadcast({
            "type": "simulation_error",
            "message": str(e)
        })
        return {"status": "error", "message": str(e)}

    await manager.broadcast({
        "type": final_status,
        "final_output_summary": final_output_summary,
        "audit_log_summary": engine.audit_logs[-1] if engine.audit_logs else None # Example summary
    })
    
    return {"status": "processing_complete", "final_status_broadcasted": final_status}

if __name__ == "__main__":
    import uvicorn
    # Ensure backend.services.knowledge_expert_service is available if used by layers
    # Example: from backend.services import knowledge_expert_service 
    uvicorn.run(app, host="0.0.0.0", port=8001)
