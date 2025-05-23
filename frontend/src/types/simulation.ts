export interface SimulationInput {
  query: string;
  session_id?: string;
  context?: Record<string, unknown>; // Broader context from client
  axes?: number[];
  // Add other relevant input fields from your SimulationQuery Pydantic model
}

export enum ResearchMode {
  PARALLEL = "parallel",
  SEQUENTIAL = "sequential",
  HIERARCHICAL = "hierarchical",
  CONSENSUS = "consensus",
}

export enum ConflictResolution {
  MAJORITY_VOTE = "majority_vote",
  CONFIDENCE_WEIGHTED = "confidence_weighted",
  EXPERT_OVERRIDE = "expert_override",
  FORK_BRANCH = "fork_branch",
}

export interface ResearchTask {
  query: string;
  context: Record<string, unknown>;
  priority?: number;
  mode?: ResearchMode;
  required_personas?: string[];
  confidence_threshold?: number;
  max_iterations?: number;
}

export interface AgentResponse {
  agent_id: string;
  persona: string;
  answer: unknown;
  confidence: number;
  reasoning: string;
  evidence: string[];
  timestamp: number; // Assuming float is converted to number (milliseconds)
  memory_patches?: Record<string, unknown>[];
  // Add any other fields from backend AgentResponse Pydantic model
}

export interface LayerOutput {
  // Define based on what each layer specifically outputs
  // For EnhancedLayer3, this would be substantial
  layer_specific_output: Record<string, unknown> | null;
  final_answer?: unknown; // If this layer provides the final answer
  agent_responses?: AgentResponse[];
  research_summary?: string;
  consensus_details?: {
    conflict_resolution_mode: ConflictResolution;
    final_confidence: number;
    reasoning: string;
  };
  forks_created?: Record<string, unknown>[]; // Details about memory forks
  // ... other potential outputs from EnhancedLayer3
}

export interface LayerResult {
  layer_id: number;
  output: LayerOutput; // This will now be more structured
  confidence: number;
  requires_escalation: boolean;
  audit_log_summary?: string; // Or a more structured audit object
  // Potentially include performance metrics here if added to backend
  execution_time?: number;
  cpu_usage?: number;
  memory_usage?: number;
}

export interface SimulationContext {
  // Mirrors the backend's simulation context state
  simulation_id: string;
  initial_query: string;
  current_layer_id: number;
  overall_confidence: number;
  escalation_count: number;
  active_agents?: AgentResponse[]; // Summary of currently active agents
  knowledge_graph_summary?: Record<string, unknown> | string | null; // Snapshot or summary
  audit_trail_summary?: Record<string, unknown>[] | null; // Summary of audit logs
  // ... any other relevant context fields from backend
}

export interface SimulationState {
  // Represents the full state of a simulation run on the frontend
  simulation_id: string | null;
  status: "idle" | "running" | "paused" | "completed" | "error";
  error_message?: string;
  input: SimulationInput | null;
  current_context: SimulationContext | null;
  layer_results: LayerResult[]; // History of results from each layer
  final_output: unknown | null; // The ultimate answer/result of the simulation
  raw_backend_response?: Record<string, unknown> | null; // For debugging
}

export interface WebSocketMessage {
  type: "layer_update" | "simulation_complete" | "error" | "status_update" | "agent_update";
  payload: LayerResult | Partial<SimulationState> | AgentResponse | { message: string } | Record<string, unknown>;
  simulation_id?: string;
} 