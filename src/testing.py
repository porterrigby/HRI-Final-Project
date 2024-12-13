"""
Testing Module for Object Detection Module Comparisons
"""
import glob
import os
import threading
import time
from collections import deque

import cv2
from PIL import Image

import retico_core
from retico_core.text import TextIU
from retico_core.dialogue import GenericDictIU
from retico_vision.vision import ImageIU, DetectedObjectsIU


class ImageFeederModule(retico_core.AbstractModule):
    @staticmethod
    def name():
        return "ImageFeeder Module"

    @staticmethod
    def description():
        return "A module for sending images single-file"

    @staticmethod
    def input_ius():
        return [TextIU]

    @staticmethod
    def output_iu():
        return ImageIU

    def __init__(self, data_path, n_samples, **kwargs):
        super().__init__(**kwargs)

        self._loop_active = False
        self.data_path = data_path
        self.image = None
        self.queue = deque(maxlen=1)
        self.n_samples = n_samples
        self.count = 0


    def process_update(self, update_message):
        for iu, ut in update_message:
            if ut == retico_core.UpdateType.ADD:
                self.queue.append(iu)

    def _loop(self):
        # path = os.path.join(self.data_path, "/*.*")
        path = "test2017_subsample100/*.*"
        print("Pulling data from: ", path)
        for file in sorted(glob.glob(path)):
            if not self._loop_active: break
            while len(self.queue) == 0: 
                time.sleep(0.01)

            # print(f"reading file: {file}")
            img = cv2.imread(file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)

            input_iu = self.queue.popleft()
            output_iu = self.create_iu(input_iu)
            output_iu.set_image(img, 1, 1)

            um = retico_core.UpdateMessage.from_iu(
                output_iu,
                retico_core.UpdateType.ADD
            )

            self.count += 1
            if self.count == self.n_samples: self._loop_active = False

            self.append(um)

    def prepare_run(self):
        self._loop_active = True
        threading.Thread(target=self._loop).start()

    def shutdown(self):
        self._loop_active = False

class Data2CSVModule(retico_core.AbstractModule):

    @staticmethod
    def name():
        return "Data2CSV Module"

    @staticmethod
    def description():
        return "Logs incoming test data into csv file"

    @staticmethod
    def input_ius():
        return [DetectedObjectsIU]

    @staticmethod
    def output_iu():
        return TextIU

    def __init__(self, filename, **kwargs):
        super().__init__(**kwargs)
        self.queue = deque(maxlen=1)
        self.filename = filename
        self.last_time = time.time()
        self.count = 0

    def process_update(self, update_message):
        for iu, ut in update_message:
            if ut == retico_core.UpdateType.ADD:
                self.queue.append(iu)

    def _loop(self):
        with open(self.filename, 'w') as file:
            file.write(f"n_detections,")
            file.write(f"time_diff,")
            file.write(f"time_created\n")

            while self._loop_active:
                if len(self.queue) == 0:
                    time.sleep(0.01)
                    continue

                input_iu = self.queue.popleft()

                time_diff = input_iu.created_at - self.last_time
                self.last_time = input_iu.created_at

                file.write(f"{len(input_iu.detected_objects)},")
                file.write(f"{time_diff},")
                file.write(f"{input_iu.created_at}\n")

                output_iu = self.create_iu(input_iu)
                output_iu.payload = f"SEND # {self.count}"
                self.count += 1
                um = retico_core.UpdateMessage.from_iu(
                    output_iu,
                    retico_core.UpdateType.ADD
                )
                self.append(um)

    def prepare_run(self):
        self._loop_active = True
        threading.Thread(target=self._loop).start()

    def shutdown(self):
        self._loop_active = False
