import os
import sys

os.environ["TORCH_CUDA_ARCH_LIST"] = '8.6'

prefix = "/home/prigby/SLIM/Retico/"

os.environ["RETICO"] = prefix + "retico-core"
os.environ["RETICOV"] = prefix + "retico-vision"
os.environ["WASR"] = prefix + "retico-whisperasr"
os.environ["SRC"] = "/home/prigby/School/HRI/FinalProject/src"
os.environ["YOLO8"] = prefix + "retico-yolov8"
os.environ["YOLO11"] = "yolov11"

sys.path.append(os.environ["WASR"])
sys.path.append(os.environ["RETICO"])
sys.path.append(os.environ["RETICOV"])
sys.path.append(os.environ["SRC"])
sys.path.append(os.environ["YOLO8"])
sys.path.append(os.environ["YOLO11"])

import retico_core
from retico_core.debug import DebugModule
from yolov8 import Yolov8
from yolov11 import Yolov11
from rtdetrv2 import RTDETR
from testing import ImageFeederModule, Data2CSVModule


# get pipeline hyperparameters from cli arguments
detector = sys.argv[1]
model = sys.argv[2]
thresh = float(sys.argv[3])
samples = int(sys.argv[4])
output_file = sys.argv[5]

if (detector == "yolov8"):
    objdet = Yolov8(model=model, thresh=thresh)
elif (detector == "yolov11"):
    objdet = Yolov11(model=model, thresh=thresh)
elif (detector == "rtdetr"):
    objdet = RTDETR(model=model, thresh=thresh)
else:
    print("Failed to correctly specify a detector. Defaulting to Yolov8 with model size n.")
    objdet = Yolov8(model='n')

csv = Data2CSVModule(output_file)
imgs = ImageFeederModule("test2017_subsample100", samples)
debug = DebugModule(print_payload_only=True)

print("Outputting to:", output_file)

imgs.subscribe(objdet)
objdet.subscribe(csv)
csv.subscribe(imgs)
csv.subscribe(debug)

imgs.run()
objdet.run()
csv.run()
debug.run()

# send start signal to ImageFeederModule
iu = csv.create_iu()
iu.payload = "SEND"
um = retico_core.UpdateMessage.from_iu(iu, retico_core.UpdateType.ADD)
imgs.process_update(um)

input()

imgs.stop()
objdet.stop()
csv.stop()
debug.stop()