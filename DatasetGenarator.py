import os
import sys
import time
import random
from queue import Queue
import numpy as np
sys.path.append(
    'E:\\neg\\term2-Arshad\\machine-learning\\CARLA_project\\CARLA_0.9.11\\WindowsNoEditor\\PythonAPI\\carla\\dist\\carla-0.9.11-py%d.%d-win-amd64.egg' % (sys.version_info.major,
                                                             sys.version_info.minor))
import carla
def establish_connection():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    return client, world

# Apply a specific weather condition to the world
def apply_weather(world, weather_params, weather_name):
    world.set_weather(weather_params)
    print(f"Weather updated to {weather_name}: {weather_params}")
    world.wait_for_tick()
    time.sleep(3.0)

def capture_image(image, queue):
    queue.put(image)

# Create a dataset with images from different weather conditions
def create_dataset(world, save_dir='dataset', num_samples=80):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    weather_conditions = [
        (carla.WeatherParameters(
            cloudiness=80.0, precipitation=80.0, sun_altitude_angle=60.0, fog_density=0.0, wind_intensity=20.0
        ), 'rain'),

        (carla.WeatherParameters(
            cloudiness=80.0, precipitation=0.0, sun_altitude_angle=10.0, fog_density=0.7, wind_intensity=5.0
        ), 'fog'),

        (carla.WeatherParameters(
            cloudiness=60.0, precipitation=0.0, sun_altitude_angle=-15.0, fog_density=0.3, wind_intensity=5.0
        ), 'night'),

        (carla.WeatherParameters(
            cloudiness=20.0, precipitation=0.0, sun_altitude_angle=70.0, fog_density=0.0, wind_intensity=5.0
        ), 'day'),
    ]

    blueprint_library = world.get_blueprint_library()
    vehicle_blueprint = blueprint_library.filter('model3')[0]
    camera_blueprint = blueprint_library.find('sensor.camera.rgb')
    camera_blueprint.set_attribute('image_size_x', '224')
    camera_blueprint.set_attribute('image_size_y', '224')

    spawn_points = world.get_map().get_spawn_points()

    for weather, condition in weather_conditions:
        apply_weather(world, weather, condition)

        condition_dir = os.path.join(save_dir, condition)
        os.makedirs(condition_dir, exist_ok=True)

        for i in range(num_samples):
            try:
                spawn_point = random.choice(spawn_points)
                vehicle = world.spawn_actor(vehicle_blueprint, spawn_point)
                camera = world.spawn_actor(
                    camera_blueprint,
                    carla.Transform(carla.Location(x=2.5, z=1.2)),
                    attach_to=vehicle
                )

                image_queue = Queue()
                camera.listen(lambda image: capture_image(image, image_queue))

                world.tick()
                time.sleep(0.1)
                image = image_queue.get()

                # Save the captured image
                image_path = os.path.join(condition_dir, f'{i}.png')
                image.save_to_disk(image_path)
                print(f"Image saved at: {image_path}")

                # Clean up actors
                camera.stop()
                vehicle.destroy()
                time.sleep(0.1)

            except Exception as e:
                print(f"Error with sample {i} in {condition}: {e}")
                if 'vehicle' in locals():
                    vehicle.destroy()
                if 'camera' in locals():
                    camera.stop()
                continue

    print("Dataset creation finished!")

def run():
    try:
        client, world = establish_connection()
        create_dataset(world)
    finally:
        print("Cleanup completed.")

if __name__ == '__main__':
    run()
