import time
import retico_core
from retico_wacnlu.common import GroundedFrameIU
import cozmo
from cozmo.util import degrees


class CozmoWACModule(retico_core.AbstractModule):
    @staticmethod
    def name():
        return "CozmoWACModule"

    @staticmethod
    def description():
        return "Cozmo control module based on retico_wacnlu"

    @staticmethod
    def input_ius():
        return [GroundedFrameIU]

    @staticmethod
    def output_iu():
        return None

    def __init__(self, robot: cozmo.robot.Robot, **kwargs):
        super().__init__(**kwargs)
        self.robot = robot
        self.turn_angle = -30
        self.target = None
        self.processing = False
        self.prepare_run()

    def prepare_run(self):
        self.robot.set_head_angle(angle=degrees(10), in_parallel=True)

    def process_update(self, update_message):
        for iu, ut in update_message:
            if ut == retico_core.UpdateType.ADD:
                self.process_iu(iu)

    def process_iu(self, iu):
        if 'word_to_find' in iu.payload.keys():
            self.target = iu.payload['word_to_find']
            print("Target set to:", self.target)
            return

        if not self.processing:
            self.processing = True

            if 'best_known_word' in iu.payload.keys() and (iu.payload['best_known_word'].lower() in self.target or self.target in iu.payload['best_known_word'].lower()):
                print("Target found!")
                self.emote()
            else:
                self.set_angle()
                self.robot.turn_in_place(angle=degrees(self.turn_angle), in_parallel=True)

            self.processing = False

        self.robot.set_head_angle(angle=degrees(10), in_parallel=True)

    def set_angle(self):
        self.turn_angle = -self.turn_angle

    def emote(self):
        self.robot.set_all_backpack_lights(cozmo.lights.green_light)
        time.sleep(0.1)
        self.robot.set_all_backpack_lights(cozmo.lights.white_light)
        time.sleep(0.1)
        self.robot.set_all_backpack_lights(cozmo.lights.green_light)
        time.sleep(0.1)
        self.robot.set_all_backpack_lights(cozmo.lights.off_light)
        time.sleep(0.1)

        self.robot.play_anim_trigger(cozmo.anim.Triggers.MeetCozmoLookFaceGetOut).wait_for_completed()
        time.sleep(0.5)
