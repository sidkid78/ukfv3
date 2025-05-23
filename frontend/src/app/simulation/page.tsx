"use client";

import { useState, useEffect, FormEvent } from "react";
// import { runSimulation } from "@/lib/api"; // No longer used directly for starting
import { 
    SimulationInput, 
    SimulationState, 
    WebSocketMessage, 
    LayerResult, 
    SimulationContext // Added SimulationContext for explicit typing
} from "@/types/simulation";

// Define the WebSocket URL, defaulting to localhost:8000 for Dockerized backend
const WEBSOCKET_URL = process.env.NEXT_PUBLIC_WEBSOCKET_URL || "ws://localhost:8000/ws/simulation";

const initialSimulationContext: SimulationContext = {
    simulation_id: "",
    initial_query: "",
    current_layer_id: 0,
    overall_confidence: 0,
    escalation_count: 0,
    active_agents: [],
    knowledge_graph_summary: null,
    audit_trail_summary: null,
};

export default function SimulationPage() {
    const [query, setQuery] = useState<string>("");
    const [simulationState, setSimulationState] = useState<SimulationState | null>(null);
    const [isLoading, setIsLoading] = useState<boolean>(false);
    const [error, setError] = useState<string | null>(null);
    const [socket, setSocket] = useState<WebSocket | null>(null);

    useEffect(() => {
        const ws = new WebSocket(WEBSOCKET_URL);
        setSocket(ws);

        ws.onopen = () => console.log("WebSocket connection established");

        ws.onmessage = (event) => {
            try {
                const message = JSON.parse(event.data as string) as WebSocketMessage;
                console.log("WebSocket message received:", message);

                setSimulationState(prevState => {
                    const baseState = prevState ?? {
                        simulation_id: null,
                        status: "idle",
                        input: null,
                        current_context: null,
                        layer_results: [],
                        final_output: null,
                    } as SimulationState; // Type assertion for initial baseState

                    switch (message.type) {
                        case "layer_update":
                            const layerResult = message.payload as LayerResult;
                            const updatedContextLayerUpdate: SimulationContext = {
                                // Ensure all fields of SimulationContext are present
                                ...(baseState.current_context ?? initialSimulationContext),
                                simulation_id: message.simulation_id || baseState.simulation_id || "temp-id", // Ensure string for context
                                current_layer_id: layerResult.layer_id,
                                overall_confidence: layerResult.confidence, 
                                initial_query: baseState.input?.query ?? "",
                            };
                            return {
                                ...baseState,
                                simulation_id: message.simulation_id ?? baseState.simulation_id ?? null,
                                status: "running",
                                layer_results: [...baseState.layer_results, layerResult],
                                current_context: updatedContextLayerUpdate,
                            };
                        case "simulation_complete":
                            const completePayload = message.payload as Partial<SimulationState>; // Payload might be partial
                            const updatedContextComplete: SimulationContext = {
                                ...(baseState.current_context ?? initialSimulationContext),
                                ...(completePayload.current_context ?? {}),
                                simulation_id: completePayload.simulation_id || baseState.simulation_id || "temp-id", // Ensure string for context
                                initial_query: completePayload.input?.query ?? baseState.input?.query ?? "",
                            };
                            return {
                                ...baseState,
                                ...completePayload,
                                current_context: updatedContextComplete,
                                status: "completed",
                            };
                        case "status_update":
                            const statusPayload = message.payload as Partial<SimulationState>; 
                             return {
                                ...baseState,
                                ...statusPayload,
                                simulation_id: statusPayload.simulation_id ?? baseState.simulation_id ?? null,
                                // Ensure current_context is maintained or updated properly if part of payload
                                current_context: statusPayload.current_context ? 
                                    {...(baseState.current_context ?? initialSimulationContext), ...statusPayload.current_context} 
                                    : (baseState.current_context ?? initialSimulationContext),
                             } as SimulationState; // Assert as SimulationState after merging
                        case "error":
                            const errorMessage = (message.payload as { message: string }).message;
                            setError(errorMessage);
                            return {
                                ...baseState,
                                status: "error",
                                error_message: errorMessage,
                            } as SimulationState;
                        default:
                            return baseState;
                    }
                });
            } catch (e) {
                console.error("Error processing WebSocket message:", e);
                setError("Failed to process update from server.");
            }
        };

        ws.onerror = (err) => {
            console.error("WebSocket error:", err);
            setError("WebSocket connection error.");
            setIsLoading(false);
        };

        ws.onclose = () => console.log("WebSocket connection closed");

        return () => ws.close();
    }, []);

    const handleSubmit = async (e: FormEvent) => {
        e.preventDefault();
        if (!query.trim()) return setError("Query cannot be empty.");
        if (!socket || socket.readyState !== WebSocket.OPEN) {
            return setError("WebSocket is not connected. Please wait or refresh.");
        }

        setIsLoading(true);
        setError(null);
        const inputForSubmit: SimulationInput = { query };
        
        // Reset state for new simulation, but keep input query if needed for display
        setSimulationState({
            simulation_id: null,
            status: "running", 
            input: inputForSubmit, // Set the input used for this run
            current_context: initialSimulationContext, // Reset context
            layer_results: [],
            final_output: null,
        });

        try {
            socket.send(JSON.stringify({ type: "start_simulation", payload: inputForSubmit }));
        } catch (err: unknown) { // Changed to unknown
            console.error("Error initiating simulation:", err);
            const message = err instanceof Error ? err.message : "Failed to start simulation.";
            setError(message);
            setSimulationState(prevState => prevState ? {...prevState, status: "error", error_message: message } : null);
            setIsLoading(false);
        }
    };
    
    const renderOutput = (output: unknown) => {
        if (typeof output === 'string') return output;
        if (output === null || output === undefined) return "No output";
        try {
            return JSON.stringify(output, null, 2);
        } catch {
            return "Could not stringify output";
        }
    }

    return (
        <div style={{ fontFamily: 'Arial, sans-serif', padding: '20px', maxWidth: '800px', margin: '0 auto' }}>
            <h1>Simulation Platform</h1>
            <form onSubmit={handleSubmit} style={{ marginBottom: '20px' }}>
                <input 
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)} 
                    placeholder="Enter your query"
                    disabled={isLoading}
                    style={{ padding: '10px', marginRight: '10px', width: '70%', border: '1px solid #ccc', borderRadius: '4px' }}
                />
                <button 
                    type="submit" 
                    disabled={isLoading || !socket || socket.readyState !== WebSocket.OPEN}
                    style={{ padding: '10px 15px', border: 'none', background: '#007bff', color: 'white', borderRadius: '4px', cursor: 'pointer' }}
                >
                    {isLoading ? "Simulating..." : "Run Simulation"}
                </button>
            </form>

            {error && <p style={{ color: 'red' }}>Error: {error}</p>}

            {simulationState && (
                <div style={{ marginTop: '20px', padding: '15px', border: '1px solid #eee', borderRadius: '4px' }}>
                    <h2>Simulation Status: {simulationState.status}</h2>
                    {simulationState.simulation_id && <p>ID: {simulationState.simulation_id}</p>}
                    {simulationState.input && <p>Query: {simulationState.input.query}</p>}

                    {simulationState.layer_results.length > 0 && (
                        <div>
                            <h3>Layer Progress:</h3>
                            <ul style={{ listStyle: 'none', paddingLeft: 0}}>
                                {simulationState.layer_results.map((lr, index) => (
                                    <li key={index} style={{ marginBottom: '10px', padding: '10px', border: '1px solid #ddd', borderRadius: '4px' }}>
                                        <strong>Layer {lr.layer_id}:</strong> Confidence {lr.confidence.toFixed(2)}
                                        {lr.output.final_answer !== undefined && lr.output.final_answer !== null && <div>Final Answer: <pre>{renderOutput(lr.output.final_answer)}</pre></div>}
                                        {lr.output.agent_responses && lr.output.agent_responses.length > 0 && (
                                            <div>
                                                <h4>Agent Responses:</h4>
                                                {lr.output.agent_responses.map((agentRes, agentIdx) => (
                                                    <div key={agentIdx} style={{ marginLeft: '15px', fontSize: '0.9em' }}>
                                                        <p><strong>{agentRes.persona} (Agent {agentRes.agent_id}):</strong> Confidence {agentRes.confidence.toFixed(2)}</p>
                                                        <p>Reasoning: {agentRes.reasoning}</p>
                                                    </div>
                                                ))}
                                            </div>
                                        )}
                                    </li>
                                ))}
                            </ul>
                        </div>
                    )}
                    {simulationState.status === "completed" && simulationState.final_output !== undefined && simulationState.final_output !== null && (
                        <div>
                            <h3>Final Output:</h3>
                            <pre>{renderOutput(simulationState.final_output)}</pre>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}