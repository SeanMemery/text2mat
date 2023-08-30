from omni.isaac.kit import SimulationApp


class App:
    def __init__(self, options={"headless": True}):
        self.simulation_app = SimulationApp(options)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.simulation_app.close()
