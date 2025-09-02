from time import sleep

from software_control.software_control_base import ActiveSoftwareControl, PassiveSoftwareControl
from utils.utils import get_project_path, get_counterpart
from utils.utils import log_and_print


class Kamoer(ActiveSoftwareControl):

    def __init__(
            self,
            icons_dir=None,
            title=None,
            menu_icons=None,
            icons_dict=None,
            **kwargs
    ):
        icons_dir = icons_dir if icons_dir else f'{get_project_path()}/software_control/icons/Kamoer'
        title = title if title else ['kamoer_title.png']
        menu_icons = menu_icons if menu_icons else ['kamoer_icon.png']
        icons_dict = icons_dict if icons_dict else {
            'start': ['kamoer_start.png'],
            'stop': ['kamoer_stop.png'],
            'search': ['kamoer_search.png'],
            'flow_in': ['kamoer_flow_in.png'],
            'flow_out': ['kamoer_flow_out.png'],
            'not_found': ['kamoer_device_not_found.png'],
            'continuous': ['kamoer_continuous.png'],
            'return': ['kamoer_return.png'],
        }
        super().__init__(
            icons_dir=icons_dir,
            title=title,
            menu_icons=menu_icons,
            icons_dict=icons_dict,
            **kwargs
        )

    def refresh(self):
        i = 0
        while self.icon_exists_on_screen(self.icons_dict['not_found']) and i < 5:
            self.log_and_print('warning', 'kamoer devices not found, search again...')
            self.click_button(self.icons_dict['search'])
            i += 1
            sleep(1)

    def pump_action(self, direction, action):
        assert direction in ['in', 'out']
        assert action in ['start', 'stop']

        # if in continuous page already, exit first
        if self.icon_exists_on_screen(self.icons_dict['continuous']):
            self.click_button(self.icons_dict['return'])
            sleep(1)

        # go into target pump setting page
        icon = self.icons_dict[f'flow_{direction}']
        while not self.icon_exists_on_screen(icon):
            self.refresh()
        self.click_button(icon)
        sleep(1)

        # click action button
        action_icon = self.icons_dict[action]
        counteraction_icon = self.icons_dict[get_counterpart(action)]
        if not self.icon_exists_on_screen(counteraction_icon):
            self.click_button_and_check_change(action_icon, counteraction_icon, exist=True)
            sleep(1)
        else:
            self.log_and_print('warning', f'pump flow_{direction} already {action}!')
        self.click_button(self.icons_dict['return'])
        sleep(1)

    def start(self):
        self.open_software()
        self.pump_action('out', 'start')
        self.pump_action('in', 'start')

    def stop(self):
        self.open_software()
        self.pump_action('in', 'stop')
        self.pump_action('out', 'stop')


class Smartlife(ActiveSoftwareControl):

    def __init__(
            self,
            icons_dir=None,
            title=None,
            menu_icons=None,
            icons_dict=None,
            **kwargs
    ):
        icons_dir = icons_dir if icons_dir else f'{get_project_path()}/software_control/icons/Smartlife'
        title = title if title else ['smartlife_title.png']
        menu_icons = menu_icons if menu_icons else ['smartlife_icon.png']
        icons_dict = icons_dict if icons_dict else {
            'start': ['smartlife_off.png'],
            'stop': ['smartlife_on.png'],
        }
        super().__init__(
            icons_dir=icons_dir,
            title=title,
            menu_icons=menu_icons,
            icons_dict=icons_dict,
            **kwargs
        )


class LaserPecker(ActiveSoftwareControl):

    def __init__(
            self,
            icons_dir=None,
            title=None,
            menu_icons=None,
            icons_dict=None,
            **kwargs
    ):
        icons_dir = icons_dir if icons_dir else f'{get_project_path()}/software_control/icons/LaserPecker'
        title = title if title else ['laserpecker_title.png']
        menu_icons = menu_icons if menu_icons else ['laserpecker_icon.png']
        icons_dict = icons_dict if icons_dict else {
            'start': ['laserpecker_start.png'],
            'confirm': ['laserpecker_confirm.png'],
        }
        super().__init__(
            icons_dir=icons_dir,
            title=title,
            menu_icons=menu_icons,
            icons_dict=icons_dict,
            **kwargs
        )

    def start(self):
        self.open_software()
        self.click_button(self.icons_dict['start'])
        sleep(1)
        self.click_button(self.icons_dict['confirm'])


# class ECLab_OBS(Software):
#     def __init__(self, exp_name, *args, **kwargs):
#         super().__init__(exp_name, *args, **kwargs)
#         self.channel_id = None
#
#     def check_in_software(self):
#         if self.move_to_icon(f'{self.icons_dir}/ec_lab_title_win11.png', confidence=self.confidence):
#             return True
#         else:
#             return False
#
#     def open_software(self):
#         i = 0
#         while not self.check_in_software() and i < 3:
#             self.click_button(f'{self.icons_dir}/ec_lab_icon_win11.png')
#             i += 1
#             self.logger.info(f'attempting to open software, trial {i} in 3 times...')
#             sleep(1)
#
#     def check_in_camera(self):
#         if self.move_to_icon(f'{self.icons_dir}/OBS_title.png', confidence=self.confidence):
#             return True
#         else:
#             return False
#
#     def open_camera(self):
#         i = 0
#         while not self.check_in_camera() and i < 3:
#             self.click_button([f'{self.icons_dir}/OBS_icon.png', f'{self.icons_dir}/OBS_icon_on_recording.png'])
#             i += 1
#             logging.info(f'attempting to open camera app, trial {i} in 3 times...')
#             sleep(0.5)
#
#     def start_recording(self):
#         self.open_camera()
#         # bypass todesk window if exist
#         self.bypass_todesk_window()
#         self.click_button(f'{self.icons_dir}/OBS_start_record.png')
#         self.recording = True
#
#     def stop_recording(self):
#         if self.recording:
#             self.open_camera()
#             # bypass todesk window if exist
#             self.bypass_todesk_window()
#             self.click_button(f'{self.icons_dir}/OBS_stop_record.png')
#             self.recording = False
#         else:
#             print('recording not started yet')
#
#     def check_in_channel(self):
#         if self.move_to_icon(f'{self.icons_dir}/channel_{self.channel_id}_in_title.png',
#                              confidence=self.confidence_high):
#             return True
#         else:
#             return False
#
#     def open_channel(self):
#         i = 0
#         while not self.check_in_channel() and i < 3:
#             self.click_button([
#                 f'{self.icons_dir}/channel_{self.channel_id}_out.png',
#                 f'{self.icons_dir}/channel_{self.channel_id}_out_relax.png',
#                 f'{self.icons_dir}/channel_{self.channel_id}_out_red.png',
#                 f'{self.icons_dir}/channel_{self.channel_id}_out_ox.png',
#             ])
#             i += 1
#             logging.info(f'attempting to open channel {self.channel_id}, trial {i} in 3 times...')
#             sleep(0.5)
#
#     def start_test(self, test_name):
#         if self.running:
#             self.logger.warning('test is already running!')
#             raise SystemError('test is already running! Restart needed!')
#         else:
#             # start the test
#             self.open_software()
#             self.open_channel()
#             self.click_button(f'{self.icons_dir}/start.png')
#             sleep(0.5)
#             self.write(test_name, interval=0.1)
#             sleep(1)
#             self.click_button_and_check_change(f'{self.icons_dir}/save_win11.png')
#             self.running = True
#
#     def check_test_finished(self):
#         while self.running:
#             # bypass teamviewer window if exist
#             self.bypass_teamviewer_prompt_window()
#             # bypass warning window if exist
#             self.bypass_software_warning_window()
#             # bypass todesk window if exist
#             self.bypass_todesk_window()
#             # open channel 1
#             self.open_channel()
#             # if not in software, go back to software if no human movement
#             x, y = pyautogui.position()
#             sleep(10)
#
#             # check if human is operating PC, any cursor movement during 10s
#             if (x, y) == pyautogui.position():
#                 # return to software if not in software
#                 if not self.check_in_software():
#                     self.open_software()
#                 start_button_location = self.move_to_icon(f'{self.icons_dir}/start.png',
#                                                           confidence=self.confidence)
#                 if start_button_location is not None:
#                     self.running = False
#             else:
#                 self.logger.info(f'human operating, fail to detect...')


# class Software:
#     def __init__(self, exp_name, to_log=True, *args, **kwargs):
#         self.configs = config_loader(exp_name, 'robot_test_configs')
#
#         # dir settings
#         self.data_dir = self.configs['data_dir']
#         self.recording_dir = self.configs['recording_dir']
#         self.icons_dir = get_dir('icons_dir')
#
#         # logger settings
#         self.to_log = to_log
#         if self.to_log:
#             self.logger = logging.getLogger(__name__)
#             self.logger.setLevel(logging.INFO)
#             formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#             # create log file if not exist
#             log_dir = get_dir('log_dir', exp_name=exp_name)
#             if not os.path.exists(log_dir):
#                 os.makedirs(log_dir)
#             file_handler = logging.FileHandler(get_log_path(exp_name))
#             file_handler.setLevel(logging.INFO)
#             file_handler.setFormatter(formatter)
#             self.logger.addHandler(file_handler)
#         else:
#             self.logger = None
#
#         # pyautogui settings
#         self.confidence = 0.9
#         self.confidence_high = 0.99
#         self.confidence_low = 0.6
#         pyautogui.FAILSAFE = False
#
#         # pre-define state variables
#         self.running = False
#         self.recording = False
#
#     def pass_screen_grab_error(func):
#         def error_passer(self, *args, **kwargs):
#             try:
#                 return func(self, *args, **kwargs)
#             except OSError as error:
#                 print(f'{error}')
#                 return None
#
#         return error_passer
#
#     @pass_screen_grab_error
#     def move_to_icon(self, file_path, confidence=None):
#         confidence = confidence or self.confidence
#         # if no icon detected on screen, will return None
#         return pyautogui.locateCenterOnScreen(file_path, confidence=confidence)
#
#     def click_button(self, file_path):
#         sleep(0.5)
#         t = 0
#         while t < 100:
#             try:
#                 if isinstance(file_path, str):
#                     x, y = self.move_to_icon(file_path, confidence=self.confidence)
#                     pyautogui.click(x, y)
#                 elif isinstance(file_path, list):
#                     for icon in file_path:
#                         if self.move_to_icon(icon, confidence=self.confidence):
#                             x, y = self.move_to_icon(icon, confidence=self.confidence)
#                             pyautogui.click(x, y)
#                 return None
#             except TypeError:
#                 log_and_print(self.to_log, self.logger, 'warning', f'cannot find {file_path} on screen')
#                 t += 1
#                 sleep(3)
#         log_and_print(self.to_log, self.logger, 'error', f'5 min timed out for finding {file_path}')
#         raise SystemError(f'5 min timed out for finding {file_path}')
#
#     def click_button_and_check_change(self, file_path, changed_file_path=None, exist=False):
#         """
#         this function is used to make sure the button is indeed clicked, by checking whether an icon
#         exists (or not) on the screen. The default logic is that after we click a button, it will
#         disappear.
#         :param file_path: button to click
#         :param changed_file_path: icon to check
#         :param exist: icon expected to show (True) or not show (False) on the screen after click the button
#         :return: None
#         """
#         changed_file_path = changed_file_path or file_path
#         while True:
#             self.click_button(file_path)
#             sleep(2)
#             if exist:
#                 if self.move_to_icon(changed_file_path):
#                     break
#             else:
#                 if not self.move_to_icon(changed_file_path):
#                     break
#
#     @staticmethod
#     def write(text, **kwargs):
#         turn_capslock_off()
#         pyautogui.write(text, **kwargs)
#
#     def bypass_teamviewer_prompt_window(self):
#         if self.move_to_icon(f'{self.icons_dir}/teamviewer_window.png', confidence=self.confidence) or \
#                 self.move_to_icon(f'{self.icons_dir}/teamviewer_window_grey.png', confidence=self.confidence) or \
#                 self.move_to_icon(f'{self.icons_dir}/teamviewer.png', confidence=self.confidence):
#             self.click_button_and_check_change(f'{self.icons_dir}/teamviewer_ok.png')
#             log_and_print(self.to_log, self.logger, 'warning', f'teamviewer prompt window bypassed!')
#
#     def bypass_software_warning_window(self):
#         if self.move_to_icon(f'{self.icons_dir}/warning.png', confidence=self.confidence):
#             self.click_button_and_check_change(f'{self.icons_dir}/close.png')
#             log_and_print(self.to_log, self.logger, 'warning', f'warning window bypassed!')
#
#     def bypass_todesk_window(self):
#         if self.move_to_icon(f'{self.icons_dir}/todesk_window.png', confidence=self.confidence):
#             self.click_button_and_check_change(
#                 f'{self.icons_dir}/todesk_window.png', f'{self.icons_dir}/todesk_window_folded.png', exist=True
#             )
#             log_and_print(self.to_log, self.logger, 'warning', f'todesk window folded!')
#
#     def bypass_teamviewer_window(self):
#         if self.move_to_icon(f'{self.icons_dir}/teamviewer_arrow.png', confidence=self.confidence):
#             self.click_button_and_check_change(
#                 f'{self.icons_dir}/teamviewer_arrow.png', f'{self.icons_dir}/teamviewer_arrow_folded.png', exist=True
#             )
#             log_and_print(self.to_log, self.logger, 'warning', f'teamviewer side window folded!')
#
#     # check whether software is available on the screen
#     def check_in_software(self):
#         raise NotImplementedError('function needs to be specified in the child class')
#
#     # open software if not in software interface, else do nothing
#     def open_software(self):
#         raise NotImplementedError('function needs to be specified in the child class')
#
#     def start_test(self, *args, **kwargs):
#         raise NotImplementedError('function needs to be specified in the child class')
#
#     def stop_exp(self):
#         if self.running:
#             self.open_software()
#             self.click_button(f'{self.icons_dir}/stop.png')
#             self.running = False
#         else:
#             log_and_print(self.to_log, self.logger, 'error', 'test is not running!')
#
#     def check_test_finished(self):
#         raise NotImplementedError('function needs to be specified in the child class')
#
#     # data analysis
#
#     def export_to_txt(self, file_name):
#         self.open_software()
#         self.click_button(f'{self.icons_dir}/ec_lab_experiment.png')
#         self.click_button(f'{self.icons_dir}/ec_lab_export_as_text.png')
#         sleep(1)
#         self.click_button(f'{self.icons_dir}/ec_lab_add.png')
#         self.write(file_name, interval=0.1)
#         sleep(0.5)
#         pyautogui.press('enter')
#         sleep(0.5)
#         pyautogui.hotkey('alt', 'e')
#         sleep(0.5)
#         pyautogui.hotkey('alt', 'c')
#
#     # camera module
#
#     # check whether camera application is available on the screen
#     def check_in_camera(self):
#         raise NotImplementedError('function needs to be specified in the child class')
#
#     # open camera app if not available, else do nothing
#     def open_camera(self):
#         raise NotImplementedError('function needs to be specified in the child class')
#
#     def start_recording(self):
#         raise NotImplementedError('function needs to be specified in the child class')
#
#     def stop_recording(self):
#         raise NotImplementedError('function needs to be specified in the child class')
#
#     def clear_recording_folder(self):
#         for f in os.listdir(self.recording_dir):
#             os.remove(os.path.join(self.recording_dir, f))


class CameraControl(ActiveSoftwareControl):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class OBS_Linux(CameraControl):

    def __init__(
            self,
            icons_dir=None,
            title=None,
            menu_icons=None,
            icons_dict=None,
            **kwargs
    ):
        icons_dir = icons_dir or f'{get_project_path()}/software_control/icons/OBS_Linux'
        title = title or ['OBS_title.png']
        menu_icons = menu_icons or ['OBS_icon.png', 'OBS_icon_hover.png']
        icons_dict = icons_dict or {
            'start': ['OBS_start_record.png', 'OBS_start_record_hover.png'],
            'stop': ['OBS_stop_record.png', 'OBS_stop_record_hover.png']
        }
        super().__init__(
            icons_dir=icons_dir,
            title=title,
            menu_icons=menu_icons,
            icons_dict=icons_dict,
            **kwargs
        )


class OBS_Win(CameraControl):

    def __init__(
            self,
            icons_dir=None,
            title=None,
            menu_icons=None,
            icons_dict=None,
            **kwargs
    ):
        icons_dir = icons_dir or f'{get_project_path()}/software_control/icons/OBS_Win'
        title = title or ['OBS_title.png']
        menu_icons = menu_icons or ['OBS_icon.png']
        icons_dict = icons_dict or {
            'start': ['OBS_start_record.png'],
            'stop': ['OBS_stop_record.png']
        }
        super().__init__(
            icons_dir=icons_dir,
            title=title,
            menu_icons=menu_icons,
            icons_dict=icons_dict,
            **kwargs
        )


class Camera_Win10(CameraControl):

    def __init__(
            self,
            icons_dir=None,
            title=None,
            menu_icons=None,
            icons_dict=None,
            **kwargs
    ):
        icons_dir = icons_dir or f'{get_project_path()}/software_control/icons/Camera_Win10'
        title = title or ['camera_title.png']
        menu_icons = menu_icons or ['camera_icon.png']
        icons_dict = icons_dict or {
            'start': ['camera_start.png'],
            'stop': ['camera_stop.png']
        }
        super().__init__(
            icons_dir=icons_dir,
            title=title,
            menu_icons=menu_icons,
            icons_dict=icons_dict,
            **kwargs
        )


class ToDesk(PassiveSoftwareControl):

    def __init__(
            self,
            icons_dir=None,
            icons_dict=None,
            **kwargs
    ):
        icons_dir = icons_dir or f'{get_project_path()}/software_control/icons/ToDesk'
        icons_dict = icons_dict or {
            'todesk_window': ['todesk_window.png'],
            'todesk_window_folded': ['todesk_window_folded.png']
        }
        super().__init__(
            icons_dir=icons_dir,
            icons_dict=icons_dict,
            **kwargs
        )

    def bypass_side_window(self):
        if self.move_to_icon(self.icons_dict['todesk_window']):
            self.click_button_and_check_change(
                self.icons_dict['todesk_window'], self.icons_dict['todesk_window_folded'], exist=True
            )
            log_and_print(self.to_log, self.logger, 'warning', f'todesk side window folded!')


class TeamViewer(PassiveSoftwareControl):

    def __init__(
            self,
            icons_dir=None,
            icons_dict=None,
            **kwargs
    ):
        icons_dir = icons_dir or f'{get_project_path()}/software_control/icons/TeamViewer'
        icons_dict = icons_dict or {
            'session_end_prompt_window': [
                'teamviewer_session_end_prompt_window.png',
                'teamviewer_session_end_prompt_window_grey.png'
            ],
            'ok': ['teamviewer_ok.png']
        }
        super().__init__(
            icons_dir=icons_dir,
            icons_dict=icons_dict,
            **kwargs
        )

    def bypass_session_end_prompt_window(self):
        if self.move_to_icon(self.icons_dict['session_end_prompt_window']):
            self.click_button_and_check_change(self.icons_dict['ok'], exist=False)
            log_and_print(self.to_log, self.logger, 'warning', f'teamviewer session end window bypassed!')

    def bypass_side_window(self):
        pass


class EC_Lab(ActiveSoftwareControl):
    def __init__(
            self,
            icons_dir,
            title=None,
            menu_icons=None,
            icons_dict=None,
            channel_id: int = 1,
            **kwargs
    ):
        title = title or ['title.png']
        menu_icons = menu_icons or ['icon.png']
        super().__init__(
            icons_dir=icons_dir,
            title=title,
            menu_icons=menu_icons,
            icons_dict=icons_dict,
            **kwargs
        )
        self.running = False
        self.channel_id = channel_id

    def check_in_channel(self):
        if self.move_to_icon(
                self.icons_dict[f'channel_{self.channel_id}_on_screen'],
                confidence=self.confidence_high,
        ):
            return True
        else:
            self.logger.info(f'channel {self.channel_id} is not open')
            return False

    def open_channel(self):
        i = 0
        while not self.check_in_channel() and i < 3:
            if not self.in_software():
                self.open_software()
            self.click_button(self.icons_dict[f'channel_{self.channel_id}_buttons'])
            i += 1
            self.logger.info(f'attempting to open channel {self.channel_id}, trial {i} in 3 times...')
            sleep(0.5)

    def start(self, test_name):
        if self.running:
            self.logger.error('test is already running!')
            raise SystemError('test is already running! Restart needed!')
        else:
            # start the test
            self.open_software()
            self.open_channel()
            self.click_button(self.icons_dict['start'])
            sleep(0.5)
            self.write(test_name, interval=0.1)
            sleep(1)
            self.click_button_and_check_change(self.icons_dict['save'])
            self.running = True

    def test_is_complete(self):
        if self.move_to_icon(self.icons_dict['start']):
            self.logger.info('test is complete')
            return True
        else:
            return False


class EC_Lab_Win11(EC_Lab):

    def __init__(
            self,
            icons_dir=None,
            title=None,
            menu_icons=None,
            icons_dict=None,
            channel_id: int = 2,
            **kwargs
    ):
        icons_dir = icons_dir or f'{get_project_path()}/software_control/icons/EC_Lab_Win11'
        icons_dict = icons_dict or {
            'start': ['start.png'],
            'stop': ['stop.png'],
            'save': ['save.png'],
            'channel_1_on_screen': ['channel_1_in_status_bar.png'],
            'channel_1_buttons': [
                'channel_1_in.png',
                'channel_1_in_ox.png',
                'channel_1_in_red.png',
                'channel_1_in_relax.png',
                'channel_1_out.png',
                'channel_1_out_ox.png',
                'channel_1_out_red.png',
                'channel_1_out_relax.png',
            ],
            'channel_2_on_screen': ['channel_2_in_status_bar.png'],
            'channel_2_buttons': [
                'channel_2_in.png',
                'channel_2_in_ox.png',
                'channel_2_in_red.png',
                'channel_2_in_relax.png',
                'channel_2_out.png',
                'channel_2_out_ox.png',
                'channel_2_out_red.png',
                'channel_2_out_relax.png',
            ],
        }
        super().__init__(
            icons_dir=icons_dir,
            title=title,
            menu_icons=menu_icons,
            icons_dict=icons_dict,
            channel_id=channel_id,
            **kwargs
        )


class EC_Lab_Win10(EC_Lab):

    def __init__(
        self,
        icons_dir=None,
        title=None,
        menu_icons=None,
        icons_dict=None,
        channel_id: int = 1,
        **kwargs
    ):
        icons_dir = icons_dir or f'{get_project_path()}/software_control/icons/EC_Lab_Win10'
        icons_dict = icons_dict or {
            'start': ['start.png'],
            'stop': ['stop.png'],
            'save': ['save.png'],
            'warning': ['warning.png'],
            'close': ['close.png'],
            'channel_1_on_screen': ['channel_1_in_title.png'],
            'channel_1_buttons': ['channel_1_in.png'],
        }
        super().__init__(
            icons_dir=icons_dir,
            title=title,
            menu_icons=menu_icons,
            icons_dict=icons_dict,
            channel_id=channel_id,
            **kwargs
        )

    def bypass_warning_window(self):
        if self.move_to_icon(self.icons_dict['warning']):
            self.click_button_and_check_change(self.icons_dict['close'])
            log_and_print(self.to_log, self.logger, 'warning', f'warning window bypassed!')


class AndroidControl(ActiveSoftwareControl):

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(
            *args,
            **kwargs
        )


class MetaView(AndroidControl):

    def __init__(
            self,
            icons_dir=None,
            title=None,
            menu_icons=None,
            icons_dict=None,
            **kwargs
    ):
        icons_dir = icons_dir or f'{get_project_path()}/software_control/icons/MetaView'
        title = title or ['title.png']
        menu_icons = menu_icons or ['icon.png']
        icons_dict = icons_dict or {
            'import': ['import.png'],
            'importing': ['importing.png', 'import_starting_soon.png'],
        }
        super().__init__(
            icons_dir=icons_dir,
            title=title,
            menu_icons=menu_icons,
            icons_dict=icons_dict,
            **kwargs
        )

    def import_image(self):
        self.open_software()
        self.click_button(self.icons_dict['import'])
        self.log_and_print('info', f'image successfully started')

    def check_import_complete(self):
        self.open_software()
        while self.move_to_icon(self.icons_dict['importing']):
            self.log_and_print('info', 'importing image...')
            sleep(1)
        self.log_and_print('info', 'image import complete')
