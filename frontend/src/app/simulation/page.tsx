"use client";

import { useState } from "react";
import { runSimulation } from "@/lib/api";

interface SimulationResult {
  layer_id: number;
  output: unknown;
  confidence: number;
  requires_escalation?: boolean;
  audit_log?: Record<string, unknown>;
  performance_metrics?: Record<string, unknown>;
}


export default function SimulationPage() {
    const [input, setInput] = useState("");
    const [result, setResult] = useState<SimulationResult | null>(null);

    const handleSubmit = async () => {
        const response = await runSimulation({ query: input });
        setResult(response);
    };

    return (
        <div>
            <input 
            value={input} 
            onChange={(e) => setInput(e.target.value)} 
            placeholder="Enter your query"
            />
            <button onClick={handleSubmit}>Run Simulation</button>
            {result && <pre>{JSON.stringify(result, null, 2)}</pre>}
        </div>
    );
}