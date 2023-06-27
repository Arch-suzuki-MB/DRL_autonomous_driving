from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
#print("the path root in world_ops.py : ",str(path_root))

import random
from utilities.logging import logger
import sys
import base_config

try:
    sys.path.append(base_config.egg_file)
except IndexError:
    pass

import carla

def try_spawn_random_vehicle_at(world, transform, role_name, autopilot=False, vehicle_type=None):
    if not vehicle_type:
        #get all the attributs of the actor and then filter on vehicule_type 
        blueprints = world.get_blueprint_library().filter('vehicle.mercedes.coupe')
    else:
        blueprints = world.get_blueprint_library().filter(vehicle_type)

    #choosing a generique model of vehicle_type at random 
    blueprint = random.choice(blueprints)
    if blueprint.has_attribute('color'):
        color = random.choice(blueprint.get_attribute('color').recommended_values)
        blueprint.set_attribute('color', color)

    blueprint.set_attribute('role_name', role_name)
    vehicle = world.try_spawn_actor(blueprint, transform)

    if (vehicle is not None) and (autopilot):
        vehicle.set_autopilot(True)
        # throttle = random.uniform(0.2, 0.45)
        # vehicle.apply_control(carla.VehicleControl(throttle=throttle,
        #                                            steer=0, brake=0.))
        # logger.info('spawned a autopilot %r at %s' % (vehicle.type_id, transform.location))
    elif (vehicle is not None) and (not autopilot):
        vehicle.set_autopilot(False)
        ## eviter le glissement en premier temps
        vehicle.apply_control(carla.VehicleControl(throttle=0,
                                            steer=0, brake=1.))
        # logger.info('spawned a egopilot %r at %s' % (vehicle.type_id, transform.location))
    return vehicle


def destroy_all_actors(world):
    """destroy all actors"""
    #destroy the vehicles 
    actor_list = world.get_actors()
    vehicles = list(actor_list.filter('vehicle*'))
    for vehicle in vehicles:
        vehicle.destroy()
    logger.info('Destroy all vehicles...')

    #destroy the sensors
    sensors = list(actor_list.filter('sensor*'))
    for sensor in sensors:
        sensor.destroy()
    logger.info('Destroy all sensors...')