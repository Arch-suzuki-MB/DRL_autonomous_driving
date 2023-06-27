""" script for specifying the saved file for generating the positions 
    configuration for invasion sensor , collision sensors 
"""
from pathlib import Path
import sys
path_root = Path(__file__).parents[3]
sys.path.append(str(path_root))
print("the path root in env_v1.py est : ",str(path_root))

import os
                # -------- scenario settings -------- #

#False : if : generate new random position else : you want to load the saved position file 
fix_vehicle_pos = True
vehicles_pos_file = r"C:\Users\safae\OneDrive\Bureau\my project\saves\positions\12_vehicle_positions.npz"

lateral_pos_limitation = (11.4, 17.7)
action_holding_time = 1./15.    ## 20hz #combien doit l'action prend de temps pour s'executer 

                # -------- sensor settings ----------- #
collision_sensor_config = {'data_type': 'sensor.other.collision','attach_to': None}
invasion_sensor_config = {'data_type': 'sensor.other.lane_invasion', 'attach_to': None}

farest_vehicle_consider = 50

# --------- action settings : actions  for static obstacles------- #

#action dict -- {idx: [throttle : acceleration, steer : diriger, brake: freiner ]}
actions = {0:[0.5, 0.5, 0.], # Bear Right & accelerate : RTL
           1:[0.5, -0.5, 0.],# Bear Left & accelerate : LTL
           2:[0.5, 0.1, 0.],#right turn with small numerical value RTS
           3:[0.5, -0.1, 0.],#left turn with small numerical value LTS
           4:[0.5, 0., 0.]}  # S :straight driving action without steering 


#throttle [0,1]     steer[-1,1]     brake [0,1]