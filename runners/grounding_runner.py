import os
import sys
import time
import random
import cozmo
from cozmo.util import degrees

os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"

PREFIX = "/home/prigby/SLIM/Retico/"
os.environ["RETICO"] = PREFIX + "retico-core"
os.environ["RETICOV"] = PREFIX + "retico-vision"
os.environ["COZMO"] = PREFIX + "retico-cozmorobot"
os.environ["WASR"] = PREFIX + "retico-whisperasr"
os.environ["ZMQ"] = PREFIX + "retico-zmq"
os.environ["CLIP"] = PREFIX + "retico-clip"
os.environ["WAC"] = PREFIX + "retico-wacnlu"
os.environ["SRC"] = "/home/prigby/School/HRI/FinalProject/src"
sys.path.append(os.environ["WASR"])
sys.path.append(os.environ["RETICO"])
sys.path.append(os.environ["RETICOV"])
sys.path.append(os.environ["COZMO"])
sys.path.append(os.environ["CLIP"])
sys.path.append(os.environ["WAC"])
sys.path.append(os.environ["ZMQ"])
sys.path.append(os.environ["SRC"])

from retico_clip.clip import ClipObjectFeatures
from retico_core.debug import DebugModule
from retico_core.text import SpeechRecognitionIU
from retico_vision.vision import ExtractObjectsModule, WebcamModule
from retico_wacnlu.words_as_classifiers import WordsAsClassifiersModule
from retico_zmq.zmq import ReaderSingleton
from retico_cozmorobot.cozmo_camera import CozmoCameraModule
from CozmoWAC import CozmoWACModule
from rtdetrv2 import RTDETR
from yolov8 import Yolov8
from yolov11 import Yolov11


def init_all(robot: cozmo.robot.Robot):
    # camera = WebcamModule(rate=1)
    camera = CozmoCameraModule(robot=robot, exposure=0.18, gain=0.6)

    thresh = 0.5
    if sys.argv[1] == 'rtdetr':
        objdet = RTDETR(model='r18', thresh=thresh)
    elif sys.argv[1] == 'yolov8':
        objdet = Yolov8(model='n', thresh=thresh)
    elif sys.argv[1] == 'yolov11':
        objdet = Yolov11(model='n', thresh=thresh)
    else:
        raise Exception("invalid detector passed")

    extractor = ExtractObjectsModule(num_obj_to_display=1, save=True)
    feats = ClipObjectFeatures()
    debug = DebugModule(print_payload_only=True)

    ip = "192.168.1.50"
    asr = ReaderSingleton(ip=ip, port="12345")
    asr.add(topic="asr", target_iu_type=SpeechRecognitionIU)

    wac_dir = "wac"
    train_wac = True
    wac = WordsAsClassifiersModule(train_mode=train_wac, wac_dir=wac_dir)

    camera.subscribe(objdet)
    objdet.subscribe(extractor)
    extractor.subscribe(feats)
    asr.subscribe(wac)
    feats.subscribe(wac)
    wac.subscribe(debug)

    modules = [asr, debug, camera, objdet, feats, extractor, wac]

    print("Press ENTER to begin training.")
    input()
    # START TRAINING
    print("Starting training. Press ENTER to stop training.")
    for module in modules:
        module.run()

    robot.set_head_angle(angle=degrees(10), in_parallel=True).wait_for_completed()

    input()

    for module in modules:
        module.stop()
    # END TRAINING

    # re-init WAC module
    train_wac = False
    wac = WordsAsClassifiersModule(train_mode=train_wac, wac_dir=wac_dir)

    camera.subscribe(objdet)
    objdet.subscribe(extractor)
    extractor.subscribe(feats)
    asr.subscribe(wac)
    feats.subscribe(wac)
    wac.subscribe(debug)

    modules = [asr, debug, camera, objdet, feats, extractor, wac]

    print("Press ENTER to start inference.")
    input()
    # START TESTING
    print("Starting inference. Press ENTER to stop inference.")
    cozmo_wac = CozmoWACModule(robot)
    wac.subscribe(cozmo_wac)

    for module in modules:
        module.run()
    time.sleep(1)
    cozmo_wac.run()

    input()

    for module in modules:
        module.stop()
    cozmo_wac.stop()
    # END TESTING


cozmo.run_program(init_all, use_viewer=True, force_viewer_on_top=True)
