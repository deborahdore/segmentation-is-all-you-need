import cv2
import numpy as np
import torch


def prepare_input(frame):
    """
    It takes a frame, resizes it to 135x135, pads it with zeros to 240x135, normalizes it, and converts it to a PyTorch
    tensor

    :param frame: the frame to be processed
    :return: A tensor of size (1, 3, 240, 135)
    """

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (135, 135), interpolation=cv2.INTER_AREA)

    frame = np.pad(frame, pad_width=((0, 0), (52, 53), (0, 0)), mode='maximum')

    norm = np.zeros((240, 240))
    frame = cv2.normalize(frame, norm, 0, 255, cv2.NORM_MINMAX)

    frame = torch.from_numpy(frame)
    frame = frame.permute(2, 0, 1)
    frame = frame.unsqueeze(0)
    frame = frame.float()
    return frame


def get_prediction(input_frame, model):
    """
    It takes in a single frame, passes it through the model, and returns the predicted class

    :param input_frame: The input frame to the model
    :param model: The model that we want to use to make predictions
    :return: The output of the model, which is the predicted class.
    """
    output = model(input_frame)
    _, output = torch.max(output, 1)
    return output
