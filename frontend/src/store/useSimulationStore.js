// Zustand store for simulation state management
import create from 'zustand';

const useSimulationStore = create((set) => ({
  // Example state
  running: false,
  step: 0,
  startSimulation: () => set({ running: true }),
  stopSimulation: () => set({ running: false }),
  nextStep: () => set((state) => ({ step: state.step + 1 })),
}));

export default useSimulationStore;
