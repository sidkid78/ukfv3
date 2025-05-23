import axios from "axios"

const api = axios.create({
    baseURL: "http://localhost:8001",
});

export const runSimulation = async (input: { query: string }) => {
    try {
        const response = await api.post("/simulation/run", input, {
            headers: {
                "Content-Type": "application/json",
            },
        });
        return response.data;
    } catch (error) {
        console.error("Error running simulation:", error);
        throw error;
    }
};