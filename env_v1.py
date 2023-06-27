
#print("the path root in env_v1.py est : ",str(path_root))

import sys
import base_config
try:
    sys.path.append(base_config.egg_file)
except IndexError:
    pass
import carla

from carla_base import carla_base
import world_ops
import sensor_ops
import env_v1_config
from generate_vehicles_pos import generate_vehicles_pos
from utilities.logging import logger
import os, random
import numpy as np
import math
from gym import spaces
import time as sys_time
import cv2

from kp2hm import heat_map
from kp2hm import gaussian_1d
from kp2hm import gaussian_2d

class ObstacleAvoidanceScenario(carla_base):
    def __init__(self):
        try:
            carla_base.__init__(self) # connecting to the server from 'carla.base.py'  
            print("-----------------------------------trying to loading the town---------------------")
            self.world = self.client.load_world('Town04')
            print("----------------------loading a Town--------------------------")
            self.world.apply_settings(carla.WorldSettings(no_rendering_mode=base_config.no_render_mode,fixed_delta_seconds=base_config.fixed_delta_seconds))
            print("----------------------setting--------------------------")
            #self.world.tick()
            print("---------------------- end of try setting--------------------------")
        except:
            raise RuntimeError('carla_base init fails...')
        
        #if the initialisation process is successful 
        else:
            if env_v1_config.fix_vehicle_pos:
                ## using predefine vehicles position
                self.vehicles_pos = np.load(env_v1_config.vehicles_pos_file)['pos']
            
            else:
                ## generate new random vehicles position
                self.vehicles_pos = generate_vehicles_pos(n_vehicles=random.randint(27, 27))

            if base_config.synchronous_mode: ## synchronous_mode is not in env_v1_config but in base_config
                self.start_synchronous_mode()

            # ------ definir la taille de action space whitch is 5  ----------- #
            self.observation_space = spaces.Discrete(len(list(env_v1_config.actions.keys())))         

            self.action_space = spaces.Discrete(len(list(env_v1_config.actions.keys())))
            print("----------------------defining action space---------------------------")
            # ----nbre of episodes , the current step ---#
            self.max_step = 2000
            self.step_counter = 0  

            # end_point
            self.end_point_x = 50. #le coordonnée x de la ligne d'arrive 

            # max velocity : echelle normalisee de la vitesse longitudinale lateral
            self.longitude_velocity_norm = 6. ## m/s, 
            self.lateral_velocity_norm = 1.5 ## m/s, 

            ## Echelle normalise dans le sens longitudinale
            self.__longitude_norm = env_v1_config.farest_vehicle_consider   

            ##nbre of vehicule on a surpassé
            self.n_been_catched_record = 0

            #-----mode of evaluation ----#
            self.test_mode = False

            self.left_obstacle = None
            self.right_obstacle = None
            print("------------------live the __init__-------------------")
            self.seed =None

    def reset(self):
        """reset the world"""
        super().reset(seed=self.seed)
        self.wait_env_running(time=0.5)
        #self.world.tick()
        print("-------------end of wait_env_running--------")
        
        world_ops.destroy_all_actors(self.world) #world_ops : code contient 2 fct qui cree les vehicules et detruit les vehicules 
        self.wait_env_running(time=0.5)
        #self.world.tick()
        self.respawn_vehicles()
        self.wait_env_running(time=0.5)
        #self.world.tick()
        self.reattach_sensors() #attacher les sensors de collision et camera bgr 

        ## Waiting for the vehicles to land on the ground
        self.wait_env_running(time=0.5)
        #self.world.tick()
        # -- reset some var -- #
        self.last_forward_distance = 0.  ## used to save the distance traveled in the last state 
        self.__ego_init_forward_pos = self.ego.get_location().x

        # step
        self.step_counter = 0
        self.n_been_catched_record = 0
        return self.get_env_state()   # return the init state

    def step(self, action_idx):
        """conduct action in env
        Args:
            action: int, an idx
        Return: env state, reward, done, info
        """
        # --- conduct action and holding a while--- #
        if self.test_mode: #if the test mode is True it gonna perform the smoothign of action by EMA
            action = np.array(env_v1_config.actions[action_idx])
            action = 0.5 * action + 0.5 * self.last_action # the exponential MVA to smooth the action
            self.last_action = action
        else:
            action = env_v1_config.actions[action_idx]

        self.ego.apply_control(carla.VehicleControl(throttle=action[0],
                                                    steer=action[1], brake=action[2]))
        
        self.wait_env_running(time=env_v1_config.action_holding_time)
        #self.world.tick()

        # -- next state -- #
        state = self.get_env_state()

        # --- reward --- # forward distance, velocity and center pos
        # forward_distance = state[0]
        # velocity = math.sqrt(state[3]**2 + state[4]**2 + 1e-8)
        # lateral_pos = state[1]
        # reward = self.__get_reward_v1(forward_distance=forward_distance, velocity=velocity,
        #                               lateral_pos=lateral_pos)
        # reward = self.__get_reward_v1()
        reward = self.get_reward_v1(state=state)
        # --reset some var -- #
        # self.last_forward_distance = forward_distance

        self.step_counter += 1

        done = self.is_done()
        return state, reward, done, {}

    def get_reward_v1(self, **states):
        
            return -2. if self.__is_illegal_done() else 1.

    def get_reward_v2(self, **states):
            reward = 0.
            if self.__get_n_been_catched_so_far() > self.n_been_catched_record:
                # print('catch one')
                reward = 1.
                self.n_been_catched_record = self.__get_n_been_catched_so_far()
            return -1. if self.__is_illegal_done() else reward
    
    def get_reward_v3(self, **states):
        obj_sigma = 10
        lane_sigma = 5
        state = states['state']

        lateral_pos = state[0]  ## [-1, 1]
        ego_pos_x_pixel = int(lateral_pos * 100)
        ego_pos_y_pixel = 512 // 2

        left_line_point = (512 // 2 - 100, 0)
        right_line_point = (512 // 2 + 100, 0)
        lane_reward_map = heat_map((512, 512),
                                  [left_line_point, right_line_point],
                                  sigma=lane_sigma, func=gaussian_1d)

        left_obstacle = state[4:7]
        if left_obstacle[0]:
            # lateral = env_v1_config.lateral_pos_limitation[1] - env_v1_config.lateral_pos_limitation[0]
            left_pos_x_pixel = int(left_obstacle[2] * 2 * 100 + 256 + ego_pos_x_pixel)
            left_pos_y_pixel = int(256 - (left_obstacle[1] * self.__longitude_norm / self.__longitude_norm * 256))

        right_obstacle = state[7:10]
        if right_obstacle[0]:
            right_pos_x_pixel = int(right_obstacle[2] * 2 * 100 + 256 + ego_pos_x_pixel)
            right_pos_y_pixel = int(256 - (right_obstacle[1] * self.__longitude_norm / self.__longitude_norm * 256))

        if left_obstacle[0] and right_obstacle[0]:
            obstacle_reward_map = heat_map((512, 512),
                                  [(left_pos_x_pixel, left_pos_y_pixel), (right_pos_x_pixel, right_pos_y_pixel)],
                                  sigma=obj_sigma, func=gaussian_2d)
        elif left_obstacle[0] and not bool(right_obstacle[0]):
            obstacle_reward_map = heat_map((512, 512),
                                  [(left_pos_x_pixel, left_pos_y_pixel)],
                                  sigma=obj_sigma, func=gaussian_2d)
        elif not bool(left_obstacle[0]) and right_obstacle[0]:
            obstacle_reward_map = heat_map((512, 512),
                                  [(right_pos_x_pixel, right_pos_y_pixel)],
                                  sigma=obj_sigma, func=gaussian_2d)
        else:
            obstacle_reward_map = np.zeros(shape=(512, 512), dtype=np.float32)

        reward_obstacle = - obstacle_reward_map[(ego_pos_y_pixel, min(511, 256 + ego_pos_x_pixel))] * 4.
        reward_lane = - lane_reward_map[(ego_pos_y_pixel, min(511, 256 + ego_pos_x_pixel))] * 3.

        reward_exist = 0.
        if not self.__is_illegal_done():
            reward_exist = 1.

        reward = reward_exist + reward_obstacle + reward_lane
        # print('reward:', reward_lane)
        # cv2.imshow('the reward map of obstacle and lane border', np.maximum(obstacle_reward_map, lane_reward_map))
        return reward

    def get_reward_v4(self, **states):
        reward_v3 = self.get_reward_v3(**states)
        lane_sigma = 10
        state = states['state']

        lateral_pos = state[0]  ## [-1, 1]
        # print('laterral_pos', lateral_pos)
        ego_pos_x_pixel = int(lateral_pos * 100)
        ego_pos_y_pixel = 512 // 2

        left_line_point = (512 // 2 - 55, 0)
        right_line_point = (512 // 2 + 55, 0)
        lane_reward_map_pos = heat_map((512, 512),
                                   [left_line_point, right_line_point],
                                   sigma=lane_sigma, func=gaussian_1d)
        ## make the ego drive at the center of one lane
        reward_lane_pos = lane_reward_map_pos[(ego_pos_y_pixel, min(511, 256 + ego_pos_x_pixel))]
        reward = reward_v3 + reward_lane_pos * 0.3
        # print('center lane reward:', reward_lane_pos)
        # cv2.imshow('reward map of center lane', lane_reward_map_pos)
        return reward


    def __gaussian_1d(self, x, mean, std, max, bias):
        def norm(x, mu, sigma):
            """normal gaussian function
            """
            # print(sigma)
            mu = np.array(mu)
            sigma = np.array(sigma)
            x = np.array(x)
            pdf = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
            return pdf

        pdf = norm(x=x, mu=mean, sigma=std) - norm(x=bias, mu=mean, sigma=std)  ## raw gaussian - bias
        pdf = pdf / (norm(x=mean, mu=mean, sigma=std)- norm(x=bias, mu=mean, sigma=std)) * max
        return pdf

    def get_ego_state(self, ego):
        ego_transform = ego.get_transform()
        ego_velocity = ego.get_velocity()
        ego_angular = ego.get_angular_velocity()
        # ego_acc = ego.get_acceleration()
        # ego_control = ego.get_control()

        # print('ego_transform.rotation.yaw', ego_transform.rotation.yaw)
        # print('ego_transform.rotation.pitch', ego_transform.rotation.pitch)
        # print('ego_transform.rotation.roll', ego_transform.rotation.roll)
        # print('ego_angular.x:', ego_angular.x)
        # print('ego_angular.y:', ego_angular.y)
        # print('ego_angular.z:', ego_angular.z)
        # print('x_v', ego_velocity.x)
        # print('y_v', ego_velocity.y)
        lateral = abs(env_v1_config.lateral_pos_limitation[1] - env_v1_config.lateral_pos_limitation[0])
        init_lateral_point = (env_v1_config.lateral_pos_limitation[1] + env_v1_config.lateral_pos_limitation[0]) / 2.
        haft_lateral = lateral / 2.
        # state = [(ego_transform.location.x - self.__ego_init_forward_pos)/self.__longitude_norm,
        #          (ego_transform.location.y - env_v1_config.lateral_pos_limitation[0])/lateral,
        #          ego_transform.rotation.yaw / 30.,
        #          ego_angular.z / 50.,
        #          ego_velocity.x / self.longitude_velocity_norm,
        #          ego_velocity.y / self.lateral_velocity_norm]
        #          # ego_acc.x, ego_acc.y,
        #          # ego_control.throttle, ego_control.steer, ego_control.brake]

        state = [(ego_transform.location.y - init_lateral_point) / haft_lateral,
                 ego_velocity.y / self.lateral_velocity_norm,
                 ego_transform.rotation.yaw / 30.,
                 ego_angular.z / 50.]

        # state[0] = max(state[0], 1e-2) if state[0] >=0 else min(state[0], -1e-2)
        # state[1] = max(state[1], 1e-2) if state[1] >=0 else min(state[1], -1e-2)
        # state[2] = max(state[2], 1e-2) if state[2] >=0 else min(state[2], -1e-2)
        # state[3] = max(state[3], 1e-2) if state[3] >=0 else min(state[3], -1e-2)

        # ego_acc.x, ego_acc.y,
        # ego_control.throttle, ego_control.steer, ego_control.brake]
        # print('state:', state)
        return state

    def get_obstacles_state(self, ego, obstacles):
        """N'enregistrer que la position et la taille d'un obstacle dans les voies de gauche et de droite qui se 
        trouve devant l'ego et qui est le plus proche de l'ego"""        # ----Obtain the vehicles closest to ego on the left and right lane lines (considering all vehicles that do not exceed 3.8m above obstacles) ----- #
        ego_location = ego.get_location()
        self.right_obstacle = None
        self.left_obstacle = None
        for obstacle in obstacles:
            obstacle_location = obstacle.get_location()
            if obstacle_location.x - ego_location.x > -3.8:  ##Si l'ego ne dépasse pas l'obstacle de plus de 3,8 m, il faut tenir compte de son impact
                if not self.left_obstacle:
                    if abs(obstacle_location.y - env_v1_config.lateral_pos_limitation[0]) <= abs(
                            obstacle_location.y - env_v1_config.lateral_pos_limitation[1]):
                        self.left_obstacle = obstacle
                else:
                    left_obstacle_location = self.left_obstacle.get_location()
                    obstacle2ego_dist = (ego_location.x - obstacle_location.x) ** 2 + (
                            ego_location.y - obstacle_location.y) ** 2
                    if abs(obstacle_location.y - env_v1_config.lateral_pos_limitation[0]) <= abs(
                            obstacle_location.y - env_v1_config.lateral_pos_limitation[1]):
                        current_left2ego_dist = (ego_location.x - left_obstacle_location.x) ** 2 + (
                                    ego_location.y - left_obstacle_location.y) ** 2
                        if obstacle2ego_dist <= current_left2ego_dist:
                            self.left_obstacle = obstacle

                if not self.right_obstacle:
                    if abs(obstacle_location.y - env_v1_config.lateral_pos_limitation[0]) > abs(
                            obstacle_location.y - env_v1_config.lateral_pos_limitation[1]):
                        self.right_obstacle = obstacle
                else:
                    right_obstacle_location = self.right_obstacle.get_location()
                    obstacle2ego_dist = (ego_location.x - obstacle_location.x) ** 2 + (
                            ego_location.y - obstacle_location.y) ** 2
                    if abs(obstacle_location.y - env_v1_config.lateral_pos_limitation[0]) > abs(
                            obstacle_location.y - env_v1_config.lateral_pos_limitation[1]):
                        current_right2ego_dist = (ego_location.x - right_obstacle_location.x) ** 2 + (
                                ego_location.y - right_obstacle_location.y) ** 2
                        if obstacle2ego_dist <= current_right2ego_dist:
                            self.right_obstacle = obstacle
        # ----Obtain the vehicles closest to ego on the left and right lane lines (considering all vehicles that do not exceed 3.8m above obstacles) ----- #

        lateral = env_v1_config.lateral_pos_limitation[1] - env_v1_config.lateral_pos_limitation[0]

        # --- 若没有，则默认0 ---- #
        if not self.left_obstacle:
            left_obstacle_location = [0., 0., 0.]  ##
        else:
            left_obstacle_location = self.left_obstacle.get_location()
            if math.sqrt((left_obstacle_location.x - ego_location.x) ** 2 + (
                    left_obstacle_location.y - ego_location.y) ** 2) <= env_v1_config.farest_vehicle_consider:
                left_obstacle_location = [1.,  ## Indicates if there is a left obstacle
                                          (left_obstacle_location.x - ego_location.x) / self.__longitude_norm,
                                          (left_obstacle_location.y - ego_location.y) / lateral]
            else:
                left_obstacle_location = [0., 0., 0.]  ##

        if not self.right_obstacle:
            right_obstacle_location = [0., 0., 0.]  ##
        else:
            right_obstacle_location = self.right_obstacle.get_location()
            if math.sqrt((right_obstacle_location.x - ego_location.x) ** 2 + (
                    right_obstacle_location.y - ego_location.y) ** 2) <= env_v1_config.farest_vehicle_consider:
                right_obstacle_location = [1.,
                                           (right_obstacle_location.x - ego_location.x) / self.__longitude_norm,
                                           (right_obstacle_location.y - ego_location.y) / lateral]
            else:
                right_obstacle_location = [0., 0., 0.]  ## Le véhicule le plus proche sur la droite est trop loin, car aucun

        ## [left_pos, left_size, right_pos, right_size]
        state = left_obstacle_location + right_obstacle_location

        ego_velocity = ego.get_velocity()
        if self.left_obstacle:
            left_velocity = self.left_obstacle.get_velocity()
            left_ego_v_x = (ego_velocity.x - left_velocity.x) / self.longitude_velocity_norm
            left_ego_v_y = (ego_velocity.y - left_velocity.y) / self.lateral_velocity_norm
        else:
            left_ego_v_x = 0.
            left_ego_v_y = 0.

        if self.right_obstacle:
            right_velocity = self.right_obstacle.get_velocity()
            right_ego_v_x = (ego_velocity.x - right_velocity.x) / self.longitude_velocity_norm
            right_ego_v_y = (ego_velocity.y - right_velocity.y) / self.lateral_velocity_norm
        else:
            right_ego_v_x = 0.
            right_ego_v_y = 0.

        state = state + [left_ego_v_x, left_ego_v_y, right_ego_v_x, right_ego_v_y]
        return state

    def get_lateral_limitation(self, ego):
        """Obtenir la distance des zones de conduite gauche et droite"""
        ego_location = ego.get_location()
        lateral = env_v1_config.lateral_pos_limitation[1] - env_v1_config.lateral_pos_limitation[0]
        left_dist = (ego_location.y - env_v1_config.lateral_pos_limitation[0]) / lateral
        right_dist = (env_v1_config.lateral_pos_limitation[1] - ego_location.y) / lateral
        return [left_dist, right_dist]

    def get_env_state(self):
        ego_state = self.get_ego_state(self.ego)
        obstacles_state = self.get_obstacles_state(self.ego, self.obstacles)
        # lateral_state = self.get_lateral_limitation(self.ego)

        # logger.info('ego_state -- ' + str(ego_state))
        # logger.info('obstacles_state -- ' + str(obstacles_state))
        # logger.info('lateral_state -- ' + str(lateral_state))

        state = ego_state + obstacles_state
        # state = ego_state + lateral_state
        # state = ego_state   ## this is ok for only balance driving
        return np.array(state)

    def respawn_vehicles(self):
        only_one_vehicle = False

        if not only_one_vehicle:
            if not env_v1_config.fix_vehicle_pos:
                self.vehicles_pos = generate_vehicles_pos(n_vehicles=random.randint(27, 27)) #list of coordonnate (x,y,z)array

            obstacles = []
            for idx, vehicle_pos in enumerate(self.vehicles_pos):
                transform = carla.Transform()
                transform.location.x = vehicle_pos[0]
                transform.location.y = vehicle_pos[1]
                transform.location.z = vehicle_pos[2]
                transform.rotation.yaw = -0.142975
                if idx == 0:    ## ego
                    self.ego = world_ops.try_spawn_random_vehicle_at(world=self.world, transform=transform,
                                                                     role_name='ego', autopilot=False,
                                                                     vehicle_type='vehicle.tesla.model3')
                    
                    self.world.tick()
                    pass
                else:  ##other vehicles
                    obstacle = world_ops.try_spawn_random_vehicle_at(world=self.world, transform=transform,
                                                                      role_name='other', autopilot=False)
                    
                    self.world.tick()
                    obstacles.append(obstacle)
            self.obstacles = obstacles
        else:
            vehicle_pos = [-30., (env_v1_config.lateral_pos_limitation[0] + env_v1_config.lateral_pos_limitation[1])/2., 1.81]
            transform = carla.Transform()
            transform.location.x = vehicle_pos[0]
            transform.location.y = vehicle_pos[1]
            transform.location.z = vehicle_pos[2]
            transform.rotation.yaw = -0.142975

            self.ego = world_ops.try_spawn_random_vehicle_at(world=self.world, transform=transform,
                                                             role_name='ego', autopilot=False,
                                                             vehicle_type='vehicle.tesla.model3')
            self.world.tick()
            self.obstacles = []

    def reattach_sensors(self):
        env_v1_config.collision_sensor_config['attach_to'] = self.ego
        self.collision_sensor = sensor_ops.collision_query(self.world, env_v1_config.collision_sensor_config)

        if self.test_mode:
            self.attach_camera_to_ego()

        # env_v1_config.invasion_sensor_config['attach_to'] = self.ego
        # self.lane_invasion_sensor = sensor_ops.lane_invasion_query(self.world, env_v1_config.invasion_sensor_config)

    def __lane_invasion(self):
        """imitate the lane invasion sensor"""
        ego_pos = self.ego.get_location()
        if ego_pos.y > env_v1_config.lateral_pos_limitation[1] or ego_pos.y < env_v1_config.lateral_pos_limitation[0]:
            return True
        else:
            return False

    def __is_finish_game(self):
        """judge agent whether finish game"""
        location = self.ego.get_location()
        return True if location.x >= self.end_point_x else False

    def __is_illegal_done(self):
        """ judge whether lane invasion or collision"""
        return self.__lane_invasion() or self.collision_sensor.get()

    def is_done(self):
        """query whether the game done"""
        max_step = self.step_counter >= self.max_step
        ##The or operator in the return statement combines these conditions and returns True if any one of them is True.
        #  So, if any of these conditions is True, the method will return True, 
        # which indicates that the game has ended
        print("-----------------collision ",self.collision_sensor.get())
        print("-----------------__is_finish_game()  ",self.__is_finish_game() )
        print("-----------------self.__lane_invasion()   ",self.__lane_invasion()  )
        print("-----------------max_step   ",max_step  )
        return self.__is_finish_game() or self.__lane_invasion() or self.collision_sensor.get() or max_step

    def wait_env_running(self, time):
        """ Attendez que l'environnement s'exécute pendant un moment
        Args:
            time: sec
        """
        if base_config.synchronous_mode:
            self.wait_carla_runing(time)
            #self.world.tick()
        else:
            sys_time.sleep(time)

    
    def random_action_test_v2(self):
        """ test script, syntronic mode
        Example:
            scenario =ObstacleAvoidanceScenario()
            scenario.reset()
            while True:
                state, done = scenario.random_action_test()
                if done:
                    scenario.reset()
                    continue
        """
        print("---------------executing the random_actionV2---------")
        state = None
        done = False

        # --- do action ---#
        self.step(action_idx=4)

        # ---- action holding ---- #
        #self.world.tick()
        self.wait_env_running(time=env_v1_config.action_holding_time)

        # ---- get state ---- #
        state = self.get_env_state()
        ego_state = state[:4]
        left_obstacle = state[4:7]
        right_obstacle = state[7:]
        print('ego_state:', ego_state)
        print('left_obstacle:', left_obstacle)
        print('right_obstacle:', right_obstacle)


        # --- reward --- # forward distance, velocity and center pos
        reward = self.get_reward_v4(state=state)
        logger.info('reward - %f'%(reward))

        done = self.is_done()

        self.vis_state_as_img(state)
        print("---------------live the random_actionV2---------")

        return state, done

    def vis_state_as_img(self, state):
        img = np.zeros(shape=(512, 512, 3), dtype=np.uint8)

        lateral_pos = state[0] ## [-1, 1]
        ego_pos_x_pixel = int(lateral_pos * 100)
        ego_pos_y_pixel = 512 // 2

        ## draw lane
        cv2.line(img, (ego_pos_y_pixel-100, 0), (ego_pos_y_pixel-100, 512),
                 color=(0, 255, 0), thickness=2)

        cv2.line(img, (ego_pos_y_pixel + 100, 0), (ego_pos_y_pixel + 100, 512),
                 color=(0, 255, 0), thickness=2)


        ## draw ego vehicle
        cv2.circle(img, (256 + ego_pos_x_pixel, ego_pos_y_pixel), color=(255, 0, 0),
                   thickness=3, radius=10)

        ##
        left_obstacle = state[4:7]
        if left_obstacle[0]:
            # lateral = env_v1_config.lateral_pos_limitation[1] - env_v1_config.lateral_pos_limitation[0]
            left_pos_x_pixel = int(left_obstacle[2] * 2 * 100 + 256 + ego_pos_x_pixel)
            left_pos_y_pixel = int(256 - (left_obstacle[1] * self.__longitude_norm / self.__longitude_norm * 256))
            cv2.circle(img, (left_pos_x_pixel, left_pos_y_pixel), color=(0, 0, 255),
                       thickness=3, radius=10)

        right_obstacle = state[7:10]
        if right_obstacle[0]:
            right_pos_x_pixel = int(right_obstacle[2] * 2 * 100 + 256 + ego_pos_x_pixel)
            right_pos_y_pixel = int(256 - (right_obstacle[1] * self.__longitude_norm / self.__longitude_norm * 256))

            cv2.circle(img, (right_pos_x_pixel, right_pos_y_pixel), color=(0, 0, 255),
                       thickness=3, radius=10)

        if left_obstacle[0] and right_obstacle[0]:
            reward_map = heat_map((512, 512), [(left_pos_x_pixel, left_pos_y_pixel), (right_pos_x_pixel, right_pos_y_pixel)], sigma=20, func=gaussian_2d)
        elif left_obstacle[0] and not bool(right_obstacle[0]):
            reward_map = heat_map((512, 512),
                                  [(left_pos_x_pixel, left_pos_y_pixel)],
                                  sigma=50, func=gaussian_2d)
        elif not bool(left_obstacle[0]) and right_obstacle[0]:
            reward_map = heat_map((512, 512),
                                  [(right_pos_x_pixel, right_pos_y_pixel)],
                                  sigma=50, func=gaussian_2d)
        else:
            reward_map = np.zeros(shape=(512, 512), dtype=np.uint8)

        # cv2.imshow('reward', reward_map)

        cv2.imshow('vis the position of ego and obstacles', img)
        cv2.waitKey(1)


    def __get_n_been_catched_so_far(self):
        #these fct permet de voir si on va attraper un obstacle 
        n_been_catched_so_far = 0
        ego_pos = self.ego.get_location()

        for obstacle in self.obstacles:
            obstacle_pos = obstacle.get_location()
            if ego_pos.x - obstacle_pos.x > 0.1:
                n_been_catched_so_far += 1
        return n_been_catched_so_far

    def attach_camera_to_ego(self):
        camera_config = {'data_type': 'sensor.camera.rgb', 'image_size_x': 640,
                              'image_size_y': 360, 'fov': 110, 'sensor_tick': 0.02,
                              'transform': carla.Transform(carla.Location(x=-0., y=-0.4, z=1.25)),
                              'attach_to': self.ego}

        self.camera = sensor_ops.bgr_camera(self.world, camera_config)

    def test(self):
        self.last_action = np.array([0., 0., 0.])
        self.test_mode = True

import gym 

def register():
    gym.register(
        id = 'ObstacleAvoidance-v0', 
        entry_point = 'env_v1:ObstacleAvoidanceScenario',
        reward_threshold = 100.
    )



if __name__ == '__main__':

    scenario = ObstacleAvoidanceScenario()
    #cpt=0
    print("--------------enter in reset scenario-------------------")
    scenario.reset()
    print("--------------live the reset scenario-------------------")
    t = True
    while True:
        _, done = scenario.random_action_test_v2()
        #cpt+=1
        if done:
            scenario.reset()
            continue
        #if cpt==2:
            #break
    
