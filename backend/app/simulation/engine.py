class Simulation:
    def __init__(self, name: str = "DefaultSimulation"):
        self.name = name
        self.state = {}

    def run(self, steps: int = 1):
        """Run the simulation for a number of steps."""
        for i in range(steps):
            self.state[f"step_{i}"] = f"State at step {i}"
        return self.state
