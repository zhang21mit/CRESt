from time import sleep

from Auto_SEM.autonomous_sem.phenom import SEM_Handler
from Auto_SEM.autonomous_sem.sem_prompt import SYSTEM_PROMPT_VISUAL_AGENT
from Auto_SEM.common.gpt4v_api import GPT4V_API_Session
from utils.sensitives import url_dict


def move_to(
        x: float,
        y: float,
):
    """
    SEM Action. Move to a position specified by absolute coordinates, the origin is the center of the stage, the x-axis is the horizontal direction with positive values to the right, and the y-axis is the vertical direction with positive values to the top

    Args:
        x: Stage position in absolute coordinates (in meters)
        y: Stage position in absolute coordinates (in meters)

    Returns:
        None
    """
    sem = SEM_Handler.get_instance()
    sem.phenom.MoveTo(x, y)
    sleep(2)
    sem.auto_adjustment()
    res = f'Move to position ({x}, {y}) successfully!'
    return res


def move_by(
        x: float,
        y: float,
):
    """
    SEM Action. Move to a position specified by relative coordinates

    Args:
        x: Move to a position specified relative to the current position (in meters), a positive x value will move to the right direction
        y: Move to a position specified relative to the current position (in meters), a positive y value will move to the up direction

    Returns:
        None
    """
    sem = SEM_Handler.get_instance()
    sem.phenom.MoveBy(x, y)
    sleep(2)
    sem.auto_adjustment()
    res = f'Move by ({x}, {y}) meters successfully!'
    return res


def set_HFW(
        HFW: float,
):
    """
    SEM Action. Set the field of view (horizontal field width, "HFW") of the currently active imaging device(i.e., NavCam or SEM) in meters, effectively setting the zoom level hfw -- Horizontal Field Width in meters(smaller number is higher magnification factor)

    Args:
        HFW: Horizontal field width (in meters)

    Returns:
        None
    """
    sem = SEM_Handler.get_instance()
    sem.phenom.SetHFW(HFW)
    sleep(2)
    sem.auto_adjustment()
    res = f'Set HFW to {HFW} meters successfully!'
    return res


def image_analysis(
        prompt: str
):
    """

    Analyze the image after operating the SEM

    Args:
        prompt: The message to convey to the visual agent, including final objective, current state and current task info. e.g. Final objective: xxx. Current state: Searching state. Current task: xxx.

    Returns:
        res: A detailed description of the image
    """

    sem = SEM_Handler.get_instance()
    image_path = sem.image_to_jpg()

    s = GPT4V_API_Session()
    res = s.image_analysis_with_som(
        image_path_list=[image_path],
        som_url=url_dict['som'],
        system_prompt=SYSTEM_PROMPT_VISUAL_AGENT,
        prompt=prompt,
        image_window_pop_up=True,
        image_window_pop_up_index=-1,
    )
    return res


def summarize(imaging_history: str):
    """
    Summarize the imaging history if continue flag is False

    Args:
        imaging_history: Imaging history in string format, about a short introduction of how we achieve the final objective and the detailed description of the final image

    Returns:
        summary: Summary of the imaging history
    """
    return imaging_history


__functions__ = [
    # move_to,
    move_by,
    set_HFW,
    image_analysis,
    summarize,
]
