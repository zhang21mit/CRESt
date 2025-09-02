import logging
import time
from datetime import datetime
from flask import Flask, request
from Auto_SEM.common.gpt4v_api import GPT4V_API_Session
from Auto_SEM.image_analysis.image_analysis_prompt import SYSTEM_PROMPT
from software_control.software_control import MetaView
from utils.utils import adb_pull_files_after_timestamp, get_project_path

app = Flask(__name__)

# log.settings
app.logger.setLevel(logging.WARNING)
log = logging.getLogger('werkzeug')
log.setLevel(logging.WARNING)

ADB_PATH = "C:/Program Files/platform-tools/adb.exe"
ANDROID_PHONE_DIR = 'sdcard/Download/Meta View'
LOCAL_DIR = f'{get_project_path()}/crest/image_analysis/images'
ANDROID_PHONE_ID = 'R3CR806WN8F'


@app.route('/')
def status_check():
    return 'the server is running ok!', 200


def import_image():
    mv = MetaView()
    mv.import_image()
    mv.check_import_complete()


@app.route('/image_analysis', methods=['POST'])
def image_analysis():
    cur_time = datetime.now()
    import_image()
    time.sleep(10)
    image_path_list = adb_pull_files_after_timestamp(ADB_PATH, ANDROID_PHONE_DIR, ANDROID_PHONE_ID, LOCAL_DIR, cur_time)
    prompt = request.get_json()['prompt'] or "describe the image precisely"
    s = GPT4V_API_Session()
    print(image_path_list)
    res = s.image_analysis(
        image_path_list=image_path_list,
        system_prompt=SYSTEM_PROMPT,
        prompt=prompt,
        image_window_pop_up=True,
    )
    return res, 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8020, debug=True)
