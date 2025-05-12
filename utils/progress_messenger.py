from flask_sse import sse

class ProgressMessenger:
    def __init__(self):
        self.module_name = "Undefined"

    def set_module(self, module_acronym):
        self.module_name = module_acronym
        sse.publish(
            { "message": f"[{self.module_name}] Initiating..." },
            type=self.module_name
        )

    def send(self, message: str):
        sse.publish(
            { "message": f"[{self.module_name}] {message}" },
            type=self.module_name
        )