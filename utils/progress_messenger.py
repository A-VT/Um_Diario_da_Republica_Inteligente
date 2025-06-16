from flask import current_app
from flask_sse import sse

class ProgressMessenger:
    def __init__(self, channel):
        self._app = current_app._get_current_object()
        self.module_name = "Undefined"
        self.channel_id = channel

    def set_module(self, module_acronym: str):
        self.module_name = module_acronym
        ctx = self._app.app_context()
        ctx.push()
        try:
            sse.publish(
                { "message": f"[{self.module_name}] Initiating..." },
                type=self.module_name,
                channel=self.channel_id
            )
        finally:
            ctx.pop()

    def send(self, message: str):
        ctx = self._app.app_context()
        ctx.push()
        try:
            sse.publish(
                { "message": f"[{self.module_name}] {message}" },
                type=self.module_name,
                channel=self.channel_id
            )
        finally:
            ctx.pop()