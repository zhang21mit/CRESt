import sys

sys.path.append('/')
from CallingGPT.src.CallingGPT.session.session import GPT_Session_Handler
from Auto_SEM.image_analysis import image_analysis_func_list
from Auto_SEM.image_analysis.image_analysis_prompt import system_prompt
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging

app = Flask(__name__)
CORS(app, resources={r"/chatgpt/messages": {"origins": "*"}})

# log settings
app.logger.setLevel(logging.WARNING)
log = logging.getLogger('werkzeug')
log.setLevel(logging.WARNING)


def get_gpt_session():
    return GPT_Session_Handler.get_instance(
        modules=[image_analysis_func_list],
        model="gpt-4",
        system_prompt=system_prompt,
        temperature=0,
    )


@app.route("/")
def status_check():
    return "the server is running ok!", 200


@app.route('/chatgpt/messages', methods=['POST'])
def update_prompt():
    session = get_gpt_session()
    prompt = request.get_json()['text']
    completion = session.ask(prompt, fc_chain=True)
    msg = {
        'answer': completion,
        'messageId': session.resp_log[-1]['id'],
    }
    return jsonify(msg), 200


@app.route('/flush')
def flush():
    session = get_gpt_session()
    session.ask("hello, what's your name?")
    return "Crest's memory successfully flushed!", 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
