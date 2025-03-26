import os
import json

class JSONFileHandler:
    def __init__(self, output_path):
        self.output_path = output_path
    
    def save_results(self, results):
        try:
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            with open(self.output_path, "w", encoding="utf-8") as file:
                json.dump(results, file, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving results: {e}")

    def read_results(self):
        try:
            with open(self.output_path, "r", encoding="utf-8") as file:
                return json.load(file)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error reading results file: {e}")
            return []

    def delete_results(self):
        try:
            if os.path.exists(self.output_path):
                os.remove(self.output_path)
        except Exception as e:
            print(f"Error deleting results: {e}")
