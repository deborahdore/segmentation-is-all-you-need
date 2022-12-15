import json
import time
from os.path import join

import cv2
import torch

from src.detection.config.config_dir import ConfigDirectory
from src.detection.utils.utility import prepare_input, get_prediction

VIDEO = "ferrari.mp4"
XML = "cars.xml"
MODEL = "model.pth"
CLASSES = "class_to_idx.json"


def main():
    """
    We load the model, create a cascade classifier for cars, create a video capture and then loop through the video frame by frame.
    For each frame, we detect cars, draw a rectangle around them, crop the image, prepare the input for the model,
    get the prediction, and then display the frame with the prediction
    """
    config_dir = ConfigDirectory()

    # load model
    model = torch.load(join(config_dir.get_model_dir(), MODEL))

    # create cascade classifiers for cars
    car_classifier = cv2.CascadeClassifier(join(config_dir.get_xml_dir(), XML))

    # create video capture
    video_cap = cv2.VideoCapture(join(config_dir.get_video_dir(), VIDEO))

    # get all classes
    classes = json.load(open(join(config_dir.get_model_dir(), CLASSES)))

    model.eval()
    while video_cap.isOpened():

        ret, frame = video_cap.read()

        cars = car_classifier.detectMultiScale(frame, 1.05, 4, minSize=(150, 100))

        for (x, y, w, h) in cars:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            crop_img = frame[y:y + h, x:x + w]

            # prepare input for the model
            input_frame = prepare_input(crop_img)
            output = get_prediction(input_frame, model)

            cv2.putText(frame, classes[str(output.item())], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (36, 255, 12), 2)
            cv2.imshow('F1 Car Detection', frame)

            time.sleep(0.1)

        if cv2.waitKey(1) == 13:
            break

    video_cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    time.sleep(2)
    main()
