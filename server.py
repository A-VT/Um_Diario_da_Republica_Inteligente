from flask import Flask, request, jsonify, render_template, Response
from flask_sse import sse

from Interface import comm_req
from IR import module as ir_module
from GR import module as gr_module

from utils.retriever.model_type import ModelType
from utils.json_file_handler import JSONFileHandler
from utils.progress_messenger import ProgressMessenger

import threading
import time

app = Flask(__name__, template_folder='Interface/templates')
app.config["REDIS_URL"] = "redis://localhost"

app.register_blueprint(sse, url_prefix='/stream')

current_messenger = None


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/send', methods=['POST'])
def send():
    global current_messenger

    data = request.get_json()
    request_data = comm_req.QueryRequest(**data)

    input_note = f"You sent: {request_data.text}. Models: {', '.join(request_data.models)}. Docs: {request_data.n_docs}. Auto Select: {request_data.auto_select_keywords}"

    # Initialize the ProgressMessenger for IR
    #current_messenger = ProgressMessenger(module_name="IR")

    # IR Module
    IR_Module = ir_module.IRSystem() #mess=current_messenger
    results = IR_Module.search(
        user_query=request_data.text,
        user_models=request_data.models,
        user_nres=request_data.n_docs,
        user_autokeywords=request_data.auto_select_keywords
    )
    
    list_ids = IR_Module.get_result_ids(results)


    file_handler = JSONFileHandler("IR_analysis/parl_europeu/final_results.json")
    file_handler.delete_results()
    file_handler.save_results(results=results)

    """ 
    # GR Module (commented out for now)
    GR_Module = gr_module.GRSystem(list_doc_ids=list_ids)
    summary_response = GR_Module.get_summaries(user_query=request_data.text)

    file_handler = JSONFileHandler("GR/results/temp_results.json")
    file_handler.delete_results()
    file_handler.save_results(results=summary_response)

    return jsonify(comm_req.QueryResponse(answer=summary_response).model_dump())
    """

    return jsonify(comm_req.QueryResponse(answer=input_note).model_dump())

if __name__ == '__main__':
    app.run(debug=True, threaded=True)



