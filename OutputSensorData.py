import os
import sys
import time
import random

sys.path.append(
    'E:\\neg\\term2-Arshad\\machine-learning\\CARLA_project\\CARLA_0.9.11\\WindowsNoEditor\\PythonAPI\\carla\\dist\\carla-0.9.11-py%d.%d-win-amd64.egg' % (sys.version_info.major,
                                                             sys.version_info.minor))
import carla

# Define output directory and subfolders for each sensor's data
data_directory = 'sensor_data'
if not os.path.exists(data_directory):
    os.makedirs(data_directory)

sensor_folders = ['lidar', 'radar', 'collision']
for folder in sensor_folders:
    sensor_folder_path = os.path.join(data_directory, folder)
    if not os.path.exists(sensor_folder_path):
        os.makedirs(sensor_folder_path)

def store_lidar_data(lidar_data):
    """Store LIDAR sensor data in a PLY file."""
    try:
        lidar_data.save_to_disk(os.path.join(data_directory, 'lidar', f'{lidar_data.frame}.ply'))
        print(f"LIDAR data for frame {lidar_data.frame} saved successfully.")
    except Exception as e:
        print(f"Error while saving LIDAR data: {e}")

def store_radar_data(radar_data):
    """Store RADAR sensor data in a text file."""
    try:
        radar_filename = os.path.join(data_directory, 'radar', f'{radar_data.frame}.radar')
        with open(radar_filename, 'w') as file:
            for detection in radar_data:
                file.write(f"{detection.azimuth},{detection.altitude},{detection.depth},{detection.velocity}\n")
        print(f"RADAR data for frame {radar_data.frame} saved successfully.")
    except Exception as e:
        print(f"Error while saving RADAR data: {e}")

def store_collision_data(collision_data):
    """Store collision sensor data in a text file."""
    try:
        collision_filename = os.path.join(data_directory, 'collision', f'collision_{collision_data.frame}.txt')
        with open(collision_filename, 'w') as file:
            file.write(f"Frame: {collision_data.frame}\n")
            file.write(f"Normal Impulse: {collision_data.normal_impulse.x}, {collision_data.normal_impulse.y}, {collision_data.normal_impulse.z}\n")
            file.write(f"Other Actor: {collision_data.other_actor.type_id if collision_data.other_actor else 'None'}\n")
        print(f"Collision detected! Data for frame {collision_data.frame} saved.")
    except Exception as e:
        print(f"Error while saving collision data: {e}")

def initialize_simulation():
    # Set up connection to the CARLA simulator
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    vehicle = None
    obstacle = None
    lidar_sensor = None
    radar_sensor = None
    collision_sensor = None

    try:
        world = client.get_world()
        blueprint_library = world.get_blueprint_library()

        # Create and spawn the vehicle
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        spawn_points = world.get_map().get_spawn_points()
        spawn_location = random.choice(spawn_points)
        vehicle = world.spawn_actor(vehicle_bp, spawn_location)
        print("Tesla Model 3 spawned successfully.")

        # Spawn an obstacle vehicle
        obstacle_bp = blueprint_library.find('vehicle.audi.a2')
        obstacle_transform = carla.Transform(
            carla.Location(x=spawn_location.location.x + 5.0, y=spawn_location.location.y, z=spawn_location.location.z),
            spawn_location.rotation
        )
        obstacle = world.spawn_actor(obstacle_bp, obstacle_transform)
        print("Obstacle vehicle spawned successfully.")

        # Attach LIDAR sensor to the vehicle
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('range', '50')
        lidar_transform = carla.Transform(carla.Location(x=0.0, z=2.5))
        lidar_sensor = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)
        lidar_sensor.listen(lambda data: store_lidar_data(data))
        print("LIDAR sensor attached.")

        # Attach RADAR sensor to the vehicle
        radar_bp = blueprint_library.find('sensor.other.radar')
        radar_transform = carla.Transform(carla.Location(x=2.0, z=1.0))
        radar_sensor = world.spawn_actor(radar_bp, radar_transform, attach_to=vehicle)
        radar_sensor.listen(lambda data: store_radar_data(data))
        print("RADAR sensor attached.")

        # Attach Collision sensor to the vehicle
        collision_bp = blueprint_library.find('sensor.other.collision')
        collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=vehicle)
        collision_sensor.listen(lambda data: store_collision_data(data))
        print("Collision sensor attached.")

        # Move the vehicle to collide with the obstacle
        vehicle.apply_control(carla.VehicleControl(throttle=0.7, steer=0.0))
        print("Vehicle moving forward for collision.")

        # Allow the simulation to run for 15 seconds to trigger the collision
        time.sleep(15)

    except Exception as e:
        print(f"Error occurred during simulation: {e}")
        raise

    finally:
        # Cleanup: stop sensors and destroy actors
        print("Cleaning up the actors and sensors...")
        for sensor in [lidar_sensor, radar_sensor, collision_sensor]:
            if sensor and sensor.is_alive:
                sensor.stop()
                sensor.destroy()
        for actor in [vehicle, obstacle]:
            if actor and actor.is_alive:
                actor.destroy()
        print("Cleanup completed.")

if __name__ == '__main__':
    initialize_simulation()
