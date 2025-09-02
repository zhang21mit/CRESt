import logging

import openai
from flask import Flask, request

from CallingGPT.src.CallingGPT.session.session import GPT_Session_Handler
from Auto_SEM.autonomous_sem import sem_func_list
from Auto_SEM.autonomous_sem.sem_prompt import SYSTEM_PROMPT_SEM_AGENT, INIT_PROMPT_SEM_AGENT
from utils.sensitives import OPENAI_API_KEY

app = Flask(__name__)

# log.settings
app.logger.setLevel(logging.WARNING)
log = logging.getLogger('werkzeug')
log.setLevel(logging.WARNING)

prompt_ending = ""


class SEM_GPT_Actuator(GPT_Session_Handler):
    """
    this class is used to store a separate sem actuator session, which can iterate with the information provided by the web chat session with vision
    """
    pass


@app.route('/')
def status_check():
    return 'the server is running ok!', 200


@app.route('/sem_imaging', methods=['POST'])
def sem_imaging():

    # initialize sem agent session
    openai.api_key = OPENAI_API_KEY
    sem_agent = SEM_GPT_Actuator.get_instance(
        modules=[sem_func_list],
        model="gpt-4-1106-preview",
        system_prompt=SYSTEM_PROMPT_SEM_AGENT + request.get_json()['final_objective'],
        temperature=0,
    )
    final_res = sem_agent.ask(INIT_PROMPT_SEM_AGENT)
    return final_res, 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8040, debug=True)
