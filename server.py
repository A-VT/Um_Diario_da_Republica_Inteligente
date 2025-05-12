from flask import Flask, request, jsonify, render_template, Response, current_app
from flask_sse import sse

from Interface import comm_req
from IR import module as ir_module
from GR import module as gr_module

from utils.retriever.model_type import ModelType
from utils.json_file_handler import JSONFileHandler
from utils.progress_messenger import ProgressMessenger

import threading
import time

app = Flask(__name__, template_folder='Interface/templates', static_folder='Interface/static')
app.config["REDIS_URL"] = "redis://localhost:6379"

app.register_blueprint(sse, url_prefix='/stream')

current_messenger = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/send', methods=['POST'])
def send():
    data = request.get_json()
    request_data = comm_req.QueryRequest(**data)

    def background_task():
        try:
            # Use the global app context
            with app.app_context():
                progress = ProgressMessenger()

                dir_answers = "IR_analysis/admin_pub"

                progress.set_module("IR")
                ir_sys = ir_module.IRSystem(progress, directory_answers=dir_answers)
                results = ir_sys.search(
                    user_query=request_data.text,
                    user_models=request_data.models,
                    user_nres=request_data.n_docs,
                    user_autokeywords=request_data.auto_select_keywords
                )

                list_ids = ir_sys.get_result_ids(results)
                file_handler = JSONFileHandler(f"{dir_answers}/IR_final_results.json")
                file_handler.delete_results()
                file_handler.save_results(results=results)

                progress.set_module("GR")
                gr_sys = gr_module.GRSystem(progress, list_doc_ids=list_ids, directory_answers="GR/results")
                summary = gr_sys.get_summaries(user_query=request_data.text)

                file_handler = JSONFileHandler(f"{dir_answers}/GR_final_results.json")
                file_handler.delete_results()
                file_handler.save_results(results=summary)

                # Final response message
                sse.publish(
                    { "message": summary },
                    type="FINAL"
                )

        except Exception as e:
            print("Error:", e)
            with app.app_context():
                sse.publish(
                    { "message": f"[ERROR] {str(e)}" },
                    type="FINAL"
                )

    threading.Thread(target=background_task).start()
    return jsonify({"status": "Processing started"})

if __name__ == '__main__':
    app.run(debug=True, threaded=True)



