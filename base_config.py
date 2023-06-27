
"""scripts for defining the location of egg file for carla """

import socket
hostname = socket.gethostname()


egg_config = {
    'jaouad':'C:/Users/jaoua/Desktop/CARLA_0.9.12/WindowsNoEditor/PythonAPI/carla/dist/carla-0.9.12-py3.7-win-amd64.egg'
            }

egg_file = egg_config[hostname]

world_ops_logger = False

no_render_mode = False
synchronous_mode =False
fixed_delta_seconds = 0.05
