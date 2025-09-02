from typing import Callable

import pandas as pd
import requests
from ax.service.utils.report_utils import exp_to_df
from mp_api.client import MPRester

from active_learning.al import AL
from db_control.database import Database
from utils.sensitives import url_dict, mp_api
from utils.utils import email, post_to

user_profiles = {
    'chu': {
        'name': 'chu',
        'projects': {
            'acidic_oer_IrRuNi': 'oxygen evolution reaction in acidic condition with IrRuNi catalyst',
            'acidic_mor_PtRuSc': 'methanol oxidation reaction in acidic condition',
        },
        'email': 'Auto_SEM.mit.demo@gmail.com',
    },

    'zhen': {
        'name': 'zhen',
        'projects': {
            # 'DFFC_PdPtCu': 'Direct formate fuel cell with PdPtCu catalyst',
            # 'DFFC_8D_4': 'Direct formate fuel cell with an octonary catalyst',
            # 'acidic_oer_IrRuNi': 'oxygen evolution reaction in acidic condition with IrRuNi catalyst',
            'ECSA_FeNiCo': 'electrochemically active surface area with FeNiCo catalyst',
        },
        'email': 'zhang21@mit.edu',
    },

    'Tom': {
        'name': 'Tom',
        'projects': {
            # 'DFFC_PdPtCu': 'Direct formate fuel cell with PdPtCu catalyst',
            # 'DFFC_8D_4': 'Direct formate fuel cell with an octonary catalyst',
            # 'acidic_oer_IrRuNi': 'oxygen evolution reaction in acidic condition with IrRuNi catalyst',
            'ECSA_FeNiCo': 'electrochemically active surface area with FeNiCo catalyst',
        },
        'email': 'zhang21@mit.edu',
    },

    'ju': {
        'name': 'ju',
        'projects': {
            'acidic_oer_IrRuNi': 'oxygen evolution reaction in acidic condition with IrRuNi catalyst',
            'acidic_mor_PtRuSc': 'methanol oxidation reaction in acidic condition',
        },
        'email': 'liju@mit.edu',
    }
}

cur_al = None


def call_again(ret_msg):
    return f"function successfully called with return value: {ret_msg}, please call this function again."


def proceed(ret_msg):
    return f"function successfully called with return value: {ret_msg}, please go to the next step."


def function_guide(ret_msg, function_name: Callable):
    return f"function successfully called with return value: {ret_msg}, please call {function_name.__name__} next."


def sanity_check(args, criteria=None):
    for arg_name, arg_val in args.items():
        if criteria is None:
            if arg_val is None:
                raise ValueError(
                    call_again(f'please ask the human operator to provide the missing information for {arg_name}'))
        else:
            if arg_val == criteria:
                raise ValueError(
                    call_again(f'please ask the human operator to provide the missing information for {arg_name}'))


def al_init(exp_name: str):
    try:
        sanity_check(locals().copy())
    except ValueError as e:
        return str(e)

    global cur_al
    if cur_al is None or cur_al.exp_name != exp_name:
        cur_al = AL(exp_name, load_from_db=True, connect_ot2=True)
    return cur_al


def flush_memory():
    """
    Flush the memory of Crest, when the human operator mentions the word 'flush'

    Args:

    Returns:
        A string indicating the status of flushing the memory
    """

    resp = requests.get(f"{url_dict['crest_backend']}/flush")
    return f"{resp.text}"


def load_user_profile(user_name: str):
    """
    Load the user profile when the user ask for current projects or email address

    Args:
        user_name(str): the name of the human operator, currently only support 'chu', 'zhen', 'ju', if not requested yet, use 'None'

    Returns:
        user_profile(dict): a dictionary containing the user profile
    """

    if user_name not in ['chu', 'zhen', 'ju', 'Tom']:
        return call_again('user name not specified/found, ask the name of the human operator again')

    return user_profiles[user_name]


def add_manual_trial(exp_name: str, elements: list, recipes: list[list[int or float]]):
    """
    Create a new trial if the human user gives the recipes to try in next batch of experiment

    Args:
        exp_name: the name of the current project, e.g. 'acidic_mor_PtRuSc'
        elements(array): A list of the chemical elements involved in the recipe. For example, ['Fe', 'Co', 'Ni']
        recipes(array): A list of the recipes. Each recipe is a list of the ratio of each element in the recipe. For example, [[1, 1, 1], [1, 1, 2]]

    Returns:
        A string indicating the status of adding the manual trial
    """

    try:
        sanity_check(locals().copy())
    except ValueError as e:
        return str(e)

    al = al_init(exp_name)
    tasks_df = pd.DataFrame(columns=elements, data=recipes)
    al.add_manual_trial(tasks_df)

    print(exp_to_df(al.exp).to_string(index=False))

    return proceed('manual trial added!')


def add_bo_trial(exp_name: str, num: int = None):
    """
    Create a new trial if the human user indicates the recipes in next batch of experiment will be selected by Active learning (Bayesian optimization)

    Args:
        exp_name: the name of the current project, e.g. 'acidic_mor_PtRuSc'
        num: the number of recipes to be selected by Active learning (Bayesian optimization)

    Returns:
        A string indicating the status of adding the BO trial
    """
    try:
        sanity_check(locals().copy())
    except ValueError as e:
        return str(e)

    al = al_init(exp_name)
    print(f'\n'
          f'[success] database connected!\n'
          f'[success] active learning history loaded!\n\n'
          f'running bayesian optimization...\n')
    al.generate_botorch_trial(num)

    print(exp_to_df(al.exp).to_string(index=False))

    trail_df = al.convert_trial_to_df(
        al.exp.trials[max(k for k in al.exp.trials.keys())],
        add_pred=True,
        model=al.get_bo_model()
    )

    trail_df_text = trail_df.to_csv(sep='\t', index=False)

    return proceed(f"BO trial added! The recipes selected are \n{trail_df_text}")


def run_opentrons(
        exp_name: str,
        user_name: str,
        elements: list,
        recipes: list[list[int or float]],
        recipe_confirm: bool = None,
        ot2_deck_check: bool = None,
        email_notify: bool = None,
        start: bool = None,
):
    """
    Run Opentrons protocol to prepare the precursor solution with the given recipe and transfer it to the carbon substrate. Make sure to check each boolean argument individually with the human operator

    Args:
        exp_name: the name of the current project, e.g. 'acidic_mor_PtRuSc'
        user_name: the name of the human operator
        elements(array): A list of the chemical elements involved in the recipe. For example, ['Fe', 'Co', 'Ni']
        recipes(array): A list of the recipes. Each recipe is a list of the ratio of each element in the recipe. For example, [[1, 1, 1], [1, 1, 2]]
        recipe_confirm: A bool indicating whether the human operator says that the recipes of the next batch of experiment are correct
        ot2_deck_check: A bool indicating whether the human operator says that he/she has checked (i) cap of the centrifuge tube is open (ii) carbon substrates are loaded on the stage
        email_notify: A bool indicating whether the human operator says that he/she wishes to receive an email notification when the experiment is finished
        start: A bool indicating whether the human operator explicitly says "start sample preparation" to start running the Opentrons


    Returns:
        a string indicating the status of the Opentrons protocol
    """

    # ask the human operator to provide the missing information if any arg is None
    args = locals().copy()
    try:
        sanity_check(args)
    except ValueError as e:
        return str(e)

    # double check if any flag is False
    checks = ['recipe_confirm', 'ot2_deck_check', 'start']
    checks_dict = {k: args[k] for k in checks}
    try:
        sanity_check(checks_dict, criteria=False)
    except ValueError as e:
        return str(e)

    # update the tasks on ot2 server
    tasks_dict = {
        'elements': elements,
        'task_list': recipes,
    }
    resp = post_to('ot2', 'update_tasks', tasks_dict)

    print(resp.text)

    # start running the Opentrons
    data = {
        'exp_name': exp_name,
        'email_notify': email_notify,
        'email_address': user_profiles[user_name]['email'],
        'user_name': user_name,
    }
    resp = post_to('ot2', 'run', data)

    print(resp.text)

    if email_notify:
        task_df = pd.DataFrame(columns=elements, data=recipes)
        content = f'Hi {user_name.capitalize()},\n\n' \
                  f'The sample preparation job of project acidic_mor_PtRuSc has started. ' \
                  f'{task_df.shape[0]} samples are being prepared:\n\n' \
                  f'{task_df.to_string(index=False)}\n\n' \
                  f'I will send you another email when the sample preparation is complete.\n\n' \
                  f'Your research assistant,\n' \
                  f'Crest'
        email_text(user_name, f'[info] sample preparation started', content)
        return f'opentrons task started! Email sent! Please wait until the experiment is finished'
    else:
        return f'opentrons task started! Please wait until the experiment is finished'


def email_text(user_name: str, subject: str, content: str):
    """
    Email the human operator with the given content

    Args:
        user_name: the name of the human operator
        subject: the subject of the email depending on the context
        content(str): the content of the email depending on the context

    Returns:
        A string indicating the status of the email sending
    """

    try:
        sanity_check(locals().copy())
    except ValueError as e:
        return str(e)

    email(user_profiles[user_name]['email'], subject, [content])
    return proceed('email successfully sent')


def email_video(user_name: str, content: str):
    """
    Email the human operator with the recording of the experiment

    Args:
        user_name: the name of the human operator
        content(str): the content of the email, including a quick wish to human operator if he/she specified the reason of being absent in lab. e.g. 'Hi Chu, \n\nPlease see the hyperlapse video in the attachment and enjoy your lunch.\n\nYour research assistant,\n Crest'

    Returns:
        A string indicating the status of the email sending
    """

    try:
        sanity_check(locals().copy())
    except ValueError as e:
        return str(e)

    data = {
        'email_address': user_profiles[user_name]['email'],
        'content': content,
    }
    resp = post_to('ot2', 'email_video', data)

    return resp.text


def database_query(exp_name: str, sql: str):
    """
    Execute SQL command to update the database

    Args:
        exp_name(str): the name of the project, e.g. acidic_mor_PtRuSc
        sql(str): the PostgreSQL command to be executed. The only accessible object is a view called 'full_table' under the schema of 'active_learning'. Columns of the view include 'trial_index', 'arm_index', 'arm_name', 'sample_id', 'sample_batch_id', 'test_id', 'abandoned', 'abandon_reason', 'outlier', 'overpotential', 'max_power', 'max_i'. Other columns are elements, but since they vary across projects, please always call SELECT * to retrieve all the information. e.g. to find the 'sample_id' with the highest 'overpotential' value: "SELECT * FROM active_learning.full_table ORDER BY overpotential DESC LIMIT 1;"

    Returns:
        A string indicating the status of the database transaction
    """

    try:
        sanity_check(locals().copy())
    except ValueError as e:
        return str(e)

    if exp_name not in ['acidic_oer_IrRuNi', 'DFFC_PdPtCu', 'acidic_mor_PtRuSc']:
        return proceed(f'{exp_name} project not supported yet in database query, please switch to another project')
    db = Database(exp_name=exp_name, to_log=False, user='li')
    try:
        ret_val = pd.read_sql(sql, db.dbConnection).to_dict()
    except Exception as e:
        return proceed(f'{exp_name} database transaction failed: {e}')
    return proceed(str(ret_val))


def phase_query(elements: str):
    """
    Query the stable phases in the chemical system of given elements

    Args:
        elements: a string of elements present in the recipe connected by dash, e.g. 'Pt-Ru-Ir'

    Returns:
        stable phases in the chemical system of given elements
    """

    try:
        sanity_check(locals().copy())
    except ValueError as e:
        return str(e)

    with MPRester(api_key=mp_api) as mpr:
        try:
            ret_val = mpr.materials.thermo.get_phase_diagram_from_chemsys(elements, thermo_type="GGA_GGA+U")
        except Exception:
            return proceed(f'phase diagram was not found in the Materials Project database')
    return proceed(f'according to Materials Project database: {str(ret_val)}')


def pump_control(exp_name: str, action: str):
    """
    Control the pump to either start or stop

    Args:
        exp_name: the name of the project, e.g. acidic_mor_PtRuSc
        action: a string indicating the action to be taken, either 'on' or 'off'

    Returns:
        A string indicating the status of the pump
    """

    try:
        sanity_check(locals().copy())
    except ValueError as e:
        return str(e)

    if action not in ['on', 'off']:
        return call_again(
            f'pump action {action} not supported, please ask the human operator to choose between on and off')
    data = {
        'exp_name': exp_name,
        'action': action,
    }
    resp = post_to('rt', 'pump_control', data)

    return resp.text


def gas_control(exp_name: str, action: str):
    """
    Control the gas to either start or stop

    Args:
        exp_name: the name of the project, e.g. acidic_mor_PtRuSc
        action: a string indicating the action to be taken, either 'on' or 'off'

    Returns:
        A string indicating the status of the gas
    """

    try:
        sanity_check(locals().copy())
    except ValueError as e:
        return str(e)

    if action not in ['on', 'off']:
        return call_again(
            f'gas action {action} not supported, please ask the human operator to choose between on and off')
    data = {
        'exp_name': exp_name,
        'action': action,
    }
    resp = post_to('rt', 'gas_control', data)

    return resp.text


def laser_cut(exp_name: str):
    """
    Cut the laser to either start or stop

    Args:
        exp_name: the name of the project, e.g. acidic_mor_PtRuSc

    Returns:
        A string indicating the status of the laser
    """

    try:
        sanity_check(locals().copy())
    except ValueError as e:
        return str(e)

    data = {
        'exp_name': exp_name,
    }
    resp = post_to('rt', 'laser_cut', data)

    return resp.text


def run_robot_test(
        exp_name: str,
        user_name: str,
        sample_id_starting: int = None,
        num_of_samples: int = None,
        sample_rack_id_starting: int = None,
        data_analysis: bool = None,
):
    """
    Run the robot test, which is a serial electrochemistry testing on the as-prepared samples with a robotic platform.

    Args:
        exp_name: the name of the project, e.g. acidic_mor_PtRuSc
        user_name: the name of the human operator
        sample_id_starting: the starting sample id of the robot test
        num_of_samples: the number of samples to be tested
        sample_rack_id_starting: the starting sample rack id of the robot test
        data_analysis: a boolean indicating whether to perform data analysis

    Returns:
        A string indicating the status of the robot test
    """

    try:
        sanity_check(locals().copy())
    except Exception as e:
        return str(e)

    if exp_name not in ['acidic_oer_IrRuNi', 'acidic_mor_PtRuSc']:
        return proceed(f'{exp_name} project not supported yet in robot test, please switch to another project')

    data = {
        'exp_name': exp_name,
        'operator_name': user_name,
        'sample_id_starting': sample_id_starting,
        'sample_id_ending': sample_id_starting + num_of_samples - 1,
        'sample_rack_id_starting': sample_rack_id_starting,
        'data_analysis': data_analysis,
    }
    resp = post_to('rt', 'run_test', data)

    return resp.text


__functions__ = [
    flush_memory,
    load_user_profile,
    add_manual_trial,
    add_bo_trial,
    email_text,
    email_video,
    run_opentrons,
    database_query,
    phase_query,
    pump_control,
    gas_control,
    laser_cut,
    run_robot_test,
]
