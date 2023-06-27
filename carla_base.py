import sys
import gym
import base_config


try:
    sys.path.append(base_config.egg_file)
    #print("egg file is passing well ")
except IndexError:
    pass
import carla
from logging_f import logger

class carla_base(gym.Env):
    """connect to the carla server"""
    world = None 
    def __init__(self):
        try:
            self.client = carla.Client("localhost", 2000)
            logger.info('carla connecting...')
            self.client.set_timeout(10.0)
            
        except:
            raise RuntimeError('carla connection fail...')
        else:
            logger.info('carla connection success...')

    def start_synchronous_mode(self):
        """carla synchoronous mode"""
        self.world.apply_settings(carla.WorldSettings(no_rendering_mode=base_config.no_render_mode,
                                                      fixed_delta_seconds=base_config.fixed_delta_seconds,
                                                      synchronous_mode=True))
        print("---------------------------starting synchrounous mode-----------------------")

    def close_synchronous_mode(self):
        """close synchoronous mode"""
        self.world.apply_settings(carla.WorldSettings(no_rendering_mode=base_config.no_render_mode,
                                                      fixed_delta_seconds=base_config.fixed_delta_seconds,
                                                      synchronous_mode=False))
        print("---------------------------close synchrounous mode-----------------------")


    def wait_carla_runing(self, time):
        """Wait for Carla to run for a specific time"""
        time_elapse = 0.
        while True:
            try:
                self.world.tick()
                break
            #time_elapse += self.wait_for_response()
            #if time_elapse > time:
            except:
                print("---------------------- wait Carla running ----------------------------- ")
                time.sleep(2)
     

    

    def pause(self):
        """pause the simulator"""
        self.start_synchronous_mode()

    def resume(self):
        """resume the simulator from pause"""
        self.close_synchronous_mode()

#if __name__ == '__main__':
#    a = carla_base()
#   pass