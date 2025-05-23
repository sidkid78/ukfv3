import axios from "axios"
import { 
    SimulationInput, 
    SimulationState, 
    // AgentResponse, // Commented out as getActiveAgents is not yet implemented
    // Assuming you might have types for Plugin, AuditLog, etc.
    // import { Plugin, AuditLog } from "../types/simulation"; 
} from "../types/simulation";

const API_BASE_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8001";

const apiClient = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        "Content-Type": "application/json",
    },
});

/**
 * Runs a simulation.
 * @param input The simulation input data.
 * @returns The initial state or result of the simulation.
 */
export const runSimulation = async (input: SimulationInput): Promise<SimulationState> => {
    try {
        const response = await apiClient.post<SimulationState>("/simulation/run", input);
        return response.data;
    } catch (error) {
        console.error("Error running simulation:", error);
        // It's good practice to throw a more specific error or handle it
        if (axios.isAxiosError(error) && error.response) {
            throw new Error(`Simulation API error: ${error.response.status} ${error.response.data?.detail || error.message}`);
        }
        throw error; // Re-throw original error if not an Axios error with response
    }
};

/**
 * Fetches the current list of active agents in a simulation.
 * @param simulationId The ID of the simulation.
 */
// export const getActiveAgents = async (simulationId: string): Promise<AgentResponse[]> => {
//     try {
//         const response = await apiClient.get<AgentResponse[]>(`/agents/${simulationId}/active`);
//         return response.data;
//     } catch (error) {
//         console.error("Error fetching active agents:", error);
//         if (axios.isAxiosError(error) && error.response) {
//             throw new Error(`Agent API error: ${error.response.status} ${error.response.data?.detail || error.message}`);
//         }
//         throw error;
//     }
// };

// Placeholder for fetching simulation history or specific run details
// export const getSimulationHistory = async (): Promise<SimulationState[]> => { ... };

// Placeholder for fetching plugin/KA list
// export const getPlugins = async (): Promise<Plugin[]> => { ... };

// Placeholder for fetching audit logs for a simulation
// export const getAuditLogs = async (simulationId: string): Promise<AuditLog[]> => { ... };

// Add other API functions as needed, e.g., for stepping, pausing, resuming simulations,
// managing agents, KAs, etc., based on your backend API capabilities.