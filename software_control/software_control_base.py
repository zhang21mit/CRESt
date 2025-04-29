from time import sleep
from typing import List, Dict
import pyautogui

from utils.utils import log_and_print, turn_capslock_off, get_logger


class SoftwareControlBase:
    def __init__(self,
                 icons_dir: str,
                 title: List[str] = None,
                 menu_icons: List[str] = None,
                 icons_dict: Dict[str, List[str]] = None,
                 exp_name: str = None,
                 force_print: bool = False,
                 **kwargs
                 ):
        self.name = self.__class__.__name__
        self.to_log = True if exp_name else False
        self.icons_dir = icons_dir
        self.menu_icons = menu_icons
        self.title = title
        self.icons_dict = icons_dict
        for k, v in kwargs.items():
            setattr(self, k, v)

        # logger settings
        self.logger = get_logger(exp_name=exp_name, module_name=self.__class__.__name__) if self.to_log else None
        self.force_print = force_print

        # pyautogui settings
        self.confidence = 0.9
        self.confidence_high = 0.99
        self.confidence_low = 0.6
        pyautogui.FAILSAFE = False

    def log_and_print(self, level, msg):
        log_and_print(self.to_log, self.logger, level, msg, self.force_print)

    def pass_screen_grab_error(func):
        def error_passer(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except OSError as error:
                print(f'{error}')
                return None

        return error_passer

    def get_icon_dir(self, icon_name: List[str]) -> List[str]:
        return [f'{self.icons_dir}/{i}' for i in icon_name]

    @pass_screen_grab_error
    def move_to_icon(self, file_name_list: List[str], confidence: float = None):
        file_path_list = self.get_icon_dir(file_name_list)
        confidence = confidence or self.confidence
        # return the first matched icon pos, if no icon detected on screen, will return None
        for file_path in file_path_list:
            pos = pyautogui.locateCenterOnScreen(file_path, confidence=confidence)
            if pos:
                return pos

    def click_button(self, file_name_list: List[str]):
        t = 0
        while t < 100:
            try:
                x, y = self.move_to_icon(file_name_list)
                pyautogui.click(x, y)
                return
            except TypeError:
                self.log_and_print('warning', f'cannot find {file_name_list} on screen')
                t += 1
                sleep(3)
        self.log_and_print('error', f'5 min timed out for finding {file_name_list}')
        raise SystemError(f'5 min timed out for finding {file_name_list}')

    def click_button_and_check_change(self, file_name_list, changed_file_name_list=None, exist=False):
        """
        this function is used to make sure the button is indeed clicked, by checking whether an icon
        exists (or not) on the screen. The default logic is that after we click a button, it will
        disappear.
        :param file_name_list: button to click
        :param changed_file_name_list: icon to check
        :param exist: icon expected to show (True) or not show (False) on the screen after click the button
        :return: None
        """
        changed_file_name_list = changed_file_name_list or file_name_list
        while True:
            self.click_button(file_name_list)
            sleep(3)
            if exist:
                if self.move_to_icon(changed_file_name_list):
                    break
            else:
                if not self.move_to_icon(changed_file_name_list):
                    break

    @staticmethod
    def write(text, **kwargs):
        turn_capslock_off()
        pyautogui.write(text, **kwargs)

    def icon_exists_on_screen(self, file_name_list: List[str], confidence: float = None) -> bool:
        return True if self.move_to_icon(file_name_list, confidence) else False


class ActiveSoftwareControl(SoftwareControlBase):
    """
    This class is used for software that will be actively actuated, e.g. start, stop
    """

    def __init__(self, title: List[str], menu_icons: List[str], **kwargs):
        super().__init__(title=title, menu_icons=menu_icons, **kwargs)

    def in_software(self, confidence: float = None) -> bool:
        # if confidence lower than 0.9, it will recognize the title in grey font (not in software) as in software
        confidence = confidence or self.confidence_high
        if self.icon_exists_on_screen(self.title, confidence):
            return True
        else:
            self.log_and_print('info', 'not in software yet')
            return False

    def open_software(self):
        i = 0
        while not self.in_software() and i < 3:
            self.click_button(self.menu_icons)
            i += 1
            self.log_and_print('info', f'attempting to open {self.name}, trial {i} in 3 times...')
            sleep(3)
        if i == 3:
            self.log_and_print('warning', f'failed to open {self.name}')
            raise SystemError(f'failed to open {self.name}')

    def start(self, **kwargs):
        assert 'start' in self.icons_dict.keys(), 'start function needs to be overwritten in child class'
        self.open_software()
        self.click_button(self.icons_dict['start'])
        self.log_and_print('info', f'{self.name} started')

    def stop(self):
        assert 'stop' in self.icons_dict.keys(), 'stop function needs to be overwritten in child class'
        self.open_software()
        self.click_button(self.icons_dict['stop'])
        self.log_and_print('info', f'{self.name} stopped')


class PassiveSoftwareControl(SoftwareControlBase):
    """
    This class is used for software that will be passively actuated,
    e.g. an event handler like bypass a prompt window whenever it pops up
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

