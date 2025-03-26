from flask import Flask, request, jsonify, render_template
from utils.retriever.model_type import ModelType
from Interface import comm_req
from IR import module as IR_module
from utils.json_file_handler import JSONFileHandler

app = Flask(__name__, template_folder='Interface/templates')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/send', methods=['POST'])
def send():

    data = request.get_json()
    request_data = comm_req.QueryRequest(**data)

    input_note = f"You sent: {request_data.text}. Models: {', '.join(request_data.models)}. Docs: {request_data.n_docs}. Auto Select: {request_data.auto_select_keywords}"

    IR_Module = IR_module.IRSystem()
    results = IR_Module.search(user_query = request_data.text, user_models = request_data.models, user_nres = request_data.n_docs, user_autokeywords = request_data.auto_select_keywords)

    file_handler= JSONFileHandler("IR/results/temp_results.json")
    file_handler.delete_results()
    file_handler.save_results(results=results)
    
    return jsonify(comm_req.QueryResponse(answer=input_note).model_dump())

if __name__ == '__main__':
    app.run(debug=True)
