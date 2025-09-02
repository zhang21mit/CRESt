from datetime import datetime
from time import sleep

import PyPhenom as ppi
from PIL import Image

from Auto_SEM.autonomous_sem import sem_prompt
from utils.utils import get_project_path

SOM_PUBLIC_URL = 'https://78e697bce30d6c90e4.gradio.live'


class SEM:

    def __init__(self, simulation: bool = False):
        PhenomID = 'MVE092827-20102-F' if not simulation else 'Simulator'
        username = 'MVE09282720102F' if not simulation else ''
        password = '8RZ9C72BYK74' if not simulation else ''
        self.phenom = ppi.Phenom(
            PhenomID,
            username,
            password
        )

        self.image_save_dir = f'{get_project_path()}/crest/autonomous_sem/images'

        self.final_objective = None

    def auto_adjustment(self):
        self.phenom.SemAutoFocus()
        sleep(5)
        self.phenom.SemAutoContrastBrightness()
        sleep(3)

    def auto_imaging(self):
        acqScanParams = ppi.ScanParams()
        acqScanParams.size = ppi.Size(1920, 1200)
        acqScanParams.detector = ppi.DetectorMode.All
        acqScanParams.nFrames = 16
        acqScanParams.hdr = False
        acqScanParams.scale = 1.0
        acq = self.phenom.SemAcquireImage(acqScanParams)
        acq.metadata.displayWidth = 0.5
        acq.metadata.dataBarLabel = "Label"
        acqWithDatabar = ppi.AddDatabar(acq)
        file_name = f'{datetime.now().strftime("%Y%m%d_%H%M%S")}.tiff'
        file_path = f'{self.image_save_dir}/{file_name}'
        ppi.Save(acqWithDatabar, file_path)
        return file_path

    def get_meta_data(self):
        meta_data = {
            'HFW': round(self.phenom.GetHFW(), 9),
            'stage_position': self.phenom.GetStageModeAndPosition().position,
        }
        return meta_data

    @staticmethod
    def convert_tiff_to_jpg(file_path: str, ):
        with Image.open(file_path) as image:
            jpg_path = file_path.replace('.tiff', '.jpg')
            image.convert("RGB").save(jpg_path, "JPEG")
        return jpg_path

    def prompt_process(self, new_chat, prompt):
        prompt = prompt if prompt else ''
        if new_chat:
            prompt = sem_prompt.SYSTEM_PROMPT_VISUAL_AGENT + prompt

        # add meta data in json format
        meta_data = self.get_meta_data()
        prompt = f"""
        {prompt}
        meta data:
        {str(meta_data)}\n
        final objective:
        {self.final_objective}
        """

        return prompt

    def image_to_jpg(self):
        tiff_path = self.auto_imaging()
        jpg_path = self.convert_tiff_to_jpg(tiff_path)
        return jpg_path


class SEM_Handler:
    """
    Singleton class to handle SEM object
    """
    _sem_ctl = None

    @classmethod
    def get_instance(cls):
        if cls._sem_ctl is None:
            cls._sem_ctl = SEM()
        return cls._sem_ctl


class WebChatSessionHandler:
    """
    Singleton class to handle ChatSession object
    """
    _chat_session_exist = False

    @classmethod
    def chat_session_exist(cls):
        if not cls._chat_session_exist:
            cls._chat_session_exist = True
            return False
        else:
            return True
