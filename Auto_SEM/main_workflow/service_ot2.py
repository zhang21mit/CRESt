import logging
from threading import Thread

import pandas as pd
from flask import Flask, request

# sys.path.append('/home/li/PycharmProjects/catalyst')
from liquid_handling_robot.ot2_env import OT2_Env_Linux
from utils.utils import get_latest_file_name, hyperlapse_video, email

app = Flask(__name__)

# log.settings
app.logger.setLevel(logging.WARNING)
log = logging.getLogger('werkzeug')
log.setLevel(logging.WARNING)

cur_tasks = None


def run_ot2(exp_name, email_notify, email_address, user_name, **kwargs):
    global cur_tasks
    ot2_env = OT2_Env_Linux(exp_name)
    ot2_env.run(cur_tasks, run=True)
    if email_notify:
        email_success(email_address, user_name, cur_tasks)


@app.route('/')
def status_check():
    return 'the server is running ok!', 200


@app.route('/run', methods=['POST'])
def run():
    global cur_tasks
    kwargs = request.get_json()
    if cur_tasks is None:
        return 'Please specify tasks first!', 400
    thread = Thread(target=run_ot2, kwargs=kwargs)
    thread.start()
    return 'opentrons started!', 200


@app.route('/update_tasks', methods=['POST'])
def update_tasks():
    global cur_tasks
    tasks = request.get_json()
    cur_tasks = pd.DataFrame(columns=tasks['elements'], data=tasks['task_list'])
    print(cur_tasks.to_string(index=False))
    return 'tasks added/updated!', 200


def email_success(email_address, user_name, task_df):
    content = f'Hi {user_name.capitalize()},\n\n' \
              f'The sample preparation job of project acidic_mor is finished!\n\n' \
              f'{task_df.shape[0]} samples are successfully prepared:\n\n' \
              f'{task_df.to_string(index=False)}\n\n' \
              f'Please let me know if you want the recording post-processed and sent to you.\n\n' \
              f'Your research assistant,\n' \
              f'Crest'
    email(email_address, '[success] opentrons tasks finished!', [content])


@app.route('/email_video', methods=['POST'])
def email_video():
    email_address = request.get_json().get('email_address')
    content = request.get_json().get('content')
    raw_path = '/media/li/HDD/ot2_recordings/raw'
    hyperlapse_path = '/media/li/HDD/ot2_recordings/hyperlapse'
    file_name = get_latest_file_name(raw_path)
    hyperlapse_video_path = hyperlapse_video(file_name, raw_path, hyperlapse_path)
    email(email_address, '[Success] sample preparation recording', [content, hyperlapse_video_path])
    return 'email with video sent', 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
