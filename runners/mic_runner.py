import os
import sys

os.environ["RETICO"] = "../retico-core"
os.environ["WASR"] = "../retico-whisperasr"
os.environ["ZMQ"] = "../retico-zmq"

sys.path.append(os.environ["WASR"])
sys.path.append(os.environ["RETICO"])
sys.path.append(os.environ["ZMQ"])

from retico_core.audio import MicrophoneModule
from retico_core.debug import DebugModule
from retico_zmq.zmq import WriterSingleton, ZeroMQWriter
from retico_whisperasr.whisperasr import WhisperASRModule


ip = "192.168.1.50"
WriterSingleton(ip=ip, port="12345")
zmqwriter = ZeroMQWriter(topic="asr")

mic = MicrophoneModule()
debug = DebugModule(print_payload_only=True)
asr = WhisperASRModule(language="english")

mic.subscribe(asr)
asr.subscribe(zmqwriter)
asr.subscribe(debug)

mic.run()
asr.run()
zmqwriter.run()
debug.run()

input()

mic.stop()
asr.stop()
zmqwriter.stop()
debug.stop()