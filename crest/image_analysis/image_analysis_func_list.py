from utils.utils import post_to


def image_analysis(
        prompt: str = None,
):
    """
    Call this function whenever the user mentioned "analyze". Analyze the image taken by the human operator with the help from another GPT agent with vision capability

    Args:
        prompt: a string indicating user's instruction about image analysis, including all the context information user mentioned in that round of conversation, as complete as possible

    Returns:
        A string indicating the result of the image analysis, which can be directly replied to the user. Make sure to describe the anomalies in the image first
    """

    data = {
        'prompt': prompt,
    }
    resp = post_to('glasses', 'image_analysis', data)

    return 'Image analysis successfully completed, Make sure to describe the anomalies in the image first: ' + resp.text


__functions__ = [
    image_analysis,
]
