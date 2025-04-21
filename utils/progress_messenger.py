from flask_sse import sse

class ProgressMessenger:
    def __init__(self, module_name="GLOBAL"):
        self.module_name = module_name
        self.current_progress = 0
        self.total_tasks = 1

    def define_max_tasks(self, max_tasks:int):
        self.total_tasks = max_tasks

    def log(self, message, increment=1):
        self.current_progress += increment

        print(f"[{self.module_name}] {message}")
        self._send_sse_message({"type": "log", "message": message})

    def update_progress(self, increment=1):
        percent = int((self.current_progress / self.total_tasks) * 100)
        self._send_sse_message({"type": "progress", "progress": percent})

    def _send_sse_message(self, data):
        try:
            sse.publish(data, type=self.module_name.lower())
        except Exception as e:
            print(f"[SSE ERROR] Could not send SSE message: {e}")