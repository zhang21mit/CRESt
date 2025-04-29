import base64
import shutil

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from gradio_client import Client
from openai import OpenAI


class GPT4V_API_Session:
    def __init__(self):
        self.fig = None
        self.ax = None

    def image_analysis(
            self,
            image_path_list: list[str],
            system_prompt: str = None,
            prompt: str = None,
            image_window_pop_up: bool = False,
            image_window_pop_up_index: int = None,
            **kwargs
    ) -> str:

        if image_window_pop_up:
            self.display_image_with_index(image_path_list, image_window_pop_up_index)

        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        *self.image_messages_process(image_path_list),
                        self.text_message_process(prompt)
                    ]
                }
            ],
            max_tokens=kwargs.get('max_tokens', 1000),
            temperature=kwargs.get('temperature', 0),
        )
        return response.choices[0].message.content

    def image_messages_process(self, image_path_list: list[str]):
        image_messages_list = []
        for image_path in image_path_list:
            image_messages_list.append({
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{self.encode_image(image_path)}",
            })
        return image_messages_list

    @staticmethod
    def text_message_process(prompt: str):
        return {
            "type": "text",
            "text": prompt
        }

    @staticmethod
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    @staticmethod
    def som_processing(
            som_url: str,
            file_path: str,
            granularity: float = 1.8,
            segmentation_mode: str = "Automatic",
            alpha: float = 0.1,
            mark_mode: str = "Alphabet",
            annotation_mode=None
    ) -> str:
        """

        Args:
            som_url: host url of the som service
            file_path: image file path to be processed
            granularity: float (numeric value between 1 and 3) in 'Granularity' Slider component, Choose in [1, 1.5), [1.5, 2.5), [2.5, 3] for [seem, semantic-sam (multi-level), sam]
            segmentation_mode: str, either 'Segmentation Mode' or 'Interactive' in 'Segmentation Mode' Radio component
            alpha: float (numeric value between 0 and 1) in 'Mask Alpha' Slider component
            mark_mode: str, either 'Alphabet' or 'Number' in 'Mark Mode' Radio component
            annotation_mode: List[str] in 'Annotation Mode' Checkboxgroup component, ['Mask', 'Box', 'Mark']

        Returns:
            som_path: image file path of the processed image
        """

        annotation_mode = annotation_mode if annotation_mode else ["Mark", "Mask"]
        client = Client(som_url, verbose=False)
        result = client.predict(
            file_path,
            granularity,
            segmentation_mode,
            alpha,
            mark_mode,
            annotation_mode,
            fn_index=2
        )

        # add suffix som to the original file_path
        som_path = file_path.replace('.jpg', '_som.png')

        # Copy the file to the new location
        shutil.copy(result, som_path)

        return som_path

    def image_analysis_with_som(
            self,
            image_path_list: list[str],
            som_url: str,
            image_window_pop_up: bool = False,
            image_window_pop_up_index: int = None,
            **kwargs
    ) -> str:
        image_path_list_with_som = []
        for image_path in image_path_list:
            image_path_list_with_som.append(image_path)
            som_path = self.som_processing(som_url, image_path)
            image_path_list_with_som.append(som_path)
        return self.image_analysis(
            image_path_list=image_path_list_with_som,
            image_window_pop_up=image_window_pop_up,
            image_window_pop_up_index=image_window_pop_up_index,
            **kwargs
        )

    @staticmethod
    def display_image(image_path):
        img = mpimg.imread(image_path)
        img_height, img_width = img.shape[:2]

        fig, ax = plt.subplots()

        # Calculate figure size to match the image aspect ratio
        DPI = fig.get_dpi()
        fig.set_size_inches(img_width / float(DPI), img_height / float(DPI))

        ax.imshow(img, aspect='auto')
        ax.axis('off')  # Hide the axes

        # Remove all margins
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1)

        # Display the image
        plt.show(block=False)  # Non-blocking show so the script continues

    def display_image_with_index(self, image_path_list, index: int = None):
        """

        Args:
            image_path_list: the list of image path
            index: the index of the image to be displayed, if None, display all images in the list

        """
        if index:
            self.display_image(image_path_list[index])
        else:
            for image_path in image_path_list:
                self.display_image(image_path)
