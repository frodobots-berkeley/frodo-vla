import argparse
import datetime
import queue
import requests
import argparse
from io import BytesIO
from agents.navigation.basic_agent import BasicAgent
from PIL import Image
import numpy as np
import base64
import time
import os

import carla
import random
from carla import Location

CONTEXT_SIZE = 3

def test_carla_connection(host='localhost', port=2000):
    try:
        client = carla.Client(host, port)
        client.set_timeout(2.0)
        world = client.get_world()
        print("Connected to CARLA server successfully.")
    except Exception as e:
        print(f"Failed to connect to CARLA: {e}")

def image_to_base64(image):
    buffer = BytesIO()
    # Convert the image to RGB mode if it's in RGBA mode
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    image.save(buffer, format="JPEG")
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return img_str

def convert_action_to_control(actions, current_speed):
    # convert from speed and curve to throttle and
    controls = []
    actions = actions[0,...]
    for i in range(actions.shape[0]): 
        action = actions[i,...]
        control = carla.VehicleControl()
        if action[1] > 0:
            control.throttle = min(action[1]*10, 1.0)
        else:
            control.brake = max(action[1]*10, -1.0)
        yaw = action[0]*action[1]
        if yaw > 0:
            control.steer = min(yaw, 1.0)
        else:
            control.steer = max(yaw, -1.0)
        controls.append(control)
    return controls

def main(args):
    # Connect to the client and retrieve the world object
    client = carla.Client('localhost', 2000)
    world = client.get_world()

    # Load the map
    client.load_world('Town05')

    # Create world and vehicle
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.find('vehicle.audi.a2')
    carla_map = world.get_map()
    spawn_point = carla_map.get_spawn_points()[0]
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)

    # Create camera
    cam_bp = None
    cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    cam_bp.set_attribute("image_size_x",str(1920))
    cam_bp.set_attribute("image_size_y",str(1080))
    cam_bp.set_attribute("fov",str(105))
    cam_bp.set_attribute('sensor_tick', '1.0')
    cam_location = carla.Location(2,0,1)
    cam_rotation = carla.Rotation(0,0,0)
    cam_transform = carla.Transform(cam_location,cam_rotation)

    # Create camera 
    ego_cam = world.spawn_actor(cam_bp, cam_transform, attach_to=vehicle)
    image_queue = queue.Queue(maxsize=CONTEXT_SIZE)
    ego_cam.listen(image_queue.put)

    # Create agent
    agent = BasicAgent(vehicle)

    # Create goal location
    context_queue = []

    # Run the commands
    prev_control = carla.VehicleControl(throttle=0.0, steer=0.0)
    step = 0
    os.makedirs("test_images", exist_ok=True)
    while True: 
        if agent.done() == True:
            print("The target has been reached, stopping the simulation")
            break
        
        # Get the current observation
        image = image_queue.get()

        # Send request to server
        image.save_to_disk("tmp.png")
        image = Image.open("tmp.png")
        context_queue.append(image_to_base64(image))

        if len(context_queue) == CONTEXT_SIZE: 
            obs_base64 = context_queue
            req_str = args.server_address + str("/inference")
            response = requests.post(req_str, json={'obs': obs_base64, 'prompt': args.prompt, 'initial_speed': prev_control.throttle}, timeout=99999999)
            actions = np.array(response.json()['actions'])
            controls = convert_action_to_control(actions, prev_control.throttle)
            context_queue = context_queue[1:]
            
            for control_step in controls[:args.num_steps]:
                print("Throttle is: ", control_step.throttle)
                print("Brake is: ", control_step.brake)
                print("Steer is: ", control_step.steer)
                vehicle.apply_control(control_step)
                world.tick()
                image = image_queue.get()
                image.save_to_disk(f"test_images/step_{step}.png")
                step += 1
            prev_control = control_step


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_address", type=str, default="http://localhost:11311")
    parser.add_argument("--prompt", type=str, default="No stop")
    parser.add_argument("--num_steps", type=int, default=None)

    args = parser.parse_args()
    main(args)
        





