""" 
RT-DETR Module
======================

This module provides object detection capabilities by using RT-DETR.
"""

from collections import deque
import threading
import time
from transformers import RTDetrImageProcessor
from transformers import RTDetrForObjectDetection
import torch

import retico_core
from retico_vision.vision import ImageIU, DetectedObjectsIU


class RTDETR(retico_core.AbstractModule):
    @staticmethod
    def name():
        return "RTDETR Object Detection Module"

    @staticmethod 
    def description():
        return "An object detection module using RT-DETR."

    @staticmethod
    def input_ius():
        return [ImageIU]

    @staticmethod 
    def output_iu():
        return DetectedObjectsIU

    def __init__(self, model='r18', thresh=0.5, **kwargs):
        super().__init__(**kwargs)

        if model == 'r18':
            self.processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_r18vd_coco_o365")
            self.model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r18vd_coco_o365")
        elif model == 'r50':
            self.processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_r50vd")
            self.model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd")
        else:
            raise Exception("Bad arg: model\n Options are: 'r18', 'r50'")

        self.thresh = thresh
        self.queue = deque(maxlen=1)
        self._detector_thread_active = False

        self.model.to(torch.device("cuda"))

    def process_update(self, update_message):
        for iu, ut in update_message:
            if ut != retico_core.UpdateType.ADD:
                continue
            self.queue.append(iu)

    def _detector_thread(self):
        while self._detector_thread_active:
            if len(self.queue) == 0:
                time.sleep(0.5)
                continue

            input_iu = self.queue.popleft()
            image = input_iu.payload

            model_inputs = self.processor(images=image, return_tensors="pt")
            model_inputs = model_inputs.to(torch.device("cuda"))

            with torch.no_grad():
                model_outputs = self.model(**model_inputs)
            target_sizes = torch.tensor([image.size[::-1]])

            results = self.processor.post_process_object_detection(model_outputs, target_sizes=target_sizes, threshold=self.thresh)[0]

            # print out labels for preds
            # for label_id in results['labels']:
            #     label = label_id.item()
            #     print(f"{self.model.config.id2label[label]}")

            valid_boxes = results['boxes'].detach().cpu().numpy()
            if len(valid_boxes) == 0: continue
            # print(valid_boxes)

            output_iu = self.create_iu(input_iu)
            output_iu.set_detected_objects(image, valid_boxes, "bb")
            um = retico_core.UpdateMessage.from_iu(output_iu, retico_core.UpdateType.ADD)
            self.append(um)

    def prepare_run(self):
        self._detector_thread_active = True
        threading.Thread(target=self._detector_thread).start()

    def shutdown(self):
        self._detector_thread_active = False
