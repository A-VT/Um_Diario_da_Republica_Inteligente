import os
import time
import threading
import traceback

from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from flask_sse import sse

from Interface import comm_req
from IR import module as ir_module
from GR import module as gr_module
from utils.json_file_handler import JSONFileHandler
from utils.progress_messenger import ProgressMessenger
from utils.llm_communicator import LLMCommunicator, LLMModelType

# Config ─────────────────────────────────────────────────────────
load_dotenv()

app = Flask(
    __name__,
    template_folder='Interface/templates',
    static_folder='Interface/static'
)
app.config["REDIS_URL"] = os.getenv("REDIS_URL", "redis://localhost:6379")
app.register_blueprint(sse, url_prefix='/stream')



# Routes ─────────────────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/send', methods=['POST'])
def send():
    data = request.get_json()
    request_data = comm_req.QueryRequest(**data)

    def background_task():
        with app.app_context():
            try:
                dir_answers = os.getenv("DIR_RESULTS")
                if not dir_answers:
                    raise RuntimeError("`DIR_RESULTS` not set in environment")

                start_time = time.time()

                # Setup IR phase
                progress = ProgressMessenger()
                llm = LLMCommunicator(LLMModelType.QWEN2_5__14b, 6000, progress)

                progress.set_module("IR")
                ir_sys = ir_module.IRSystem(dir_answers, progress, llm)
                ir_results = ir_sys.search(
                    user_query=request_data.text,
                    user_models=request_data.models,
                    user_nres=request_data.n_docs,
                    user_autokeywords=request_data.auto_select_keywords
                )
                doc_ids = ir_sys.get_result_ids(ir_results)

                # Persist IR
                ir_handler = JSONFileHandler(f"{dir_answers}/IR_final_results.json")
                ir_handler.delete_results()
                ir_handler.save_results(results=ir_results)

                # Setup GR phase
                progress.set_module("GR")
                gr_sys = gr_module.GRSystem(
                    dir_answers,
                    progress,
                    llm,
                    list_doc_ids=doc_ids
                )
                gr_output = gr_sys.get_summaries(
                    user_query=request_data.text,
                    previous_responses=request_data.previous_responses
                )

                summary_text = gr_output["summary"]
                doc_ids_list  = gr_output["document_ids"]

                # Persist GR
                gr_handler = JSONFileHandler(f"{dir_answers}/GR_final_results.json")
                gr_handler.delete_results()
                gr_handler.save_results(results=gr_output)

                # Send via SSE
                sse.publish(
                    {
                        "summary": summary_text,
                        "document_ids": doc_ids_list
                    },
                    type="FINAL"
                )

                print(f"[TIMER] Total duration: {time.time() - start_time:.2f}s")

            except Exception as e:
                print("Error in background task:\n", traceback.format_exc())
                sse.publish(
                    {
                        "summary": f"[_ERROR_] {str(e)}",
                        "document_ids": []
                    },
                    type="FINAL"
                )

    thread = threading.Thread(target=background_task, daemon=True)
    thread.start()
    return jsonify({"status": "Processing started"}), 202


if __name__ == '__main__':
    app.run(debug=True, threaded=True)
