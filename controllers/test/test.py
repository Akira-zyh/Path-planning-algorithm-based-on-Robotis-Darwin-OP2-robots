"""test controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Supervisor
from managers import RobotisOp2GaitManager,RobotisOp2MotionManager
import sys
# create the Robot instance.
robot = Supervisor()

NMOTORS = 20
motorNames = [
    "ShoulderR", "ShoulderL", "ArmUpperR", "ArmUpperL", "ArmLowerR",
    "ArmLowerL", "PelvYR", "PelvYL", "PelvR", "PelvL", "LegUpperR",
    "LegUpperL", "LegLowerR", "LegLowerL", "AnkleR", "AnkleL", "FootR",
    "FootL", "Neck", "Head"
]

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())
# lidar = robot.getDevice("lidar")
# lidar.enable(timestep)
# lidar.enablePointCloud()
# robot.simulationResetPhysics()
# You should insert a getDevice-like function in order to get the
# instance of a device of the robot. Something like:
#  motor = robot.getDevice('motorname')
#  ds = robot.getDevice('dsname')
#  ds.enable(timestep)
robot.step(timestep)
gait_manager = RobotisOp2GaitManager(robot, "config.ini")
gait_manager.start()
motion_manager = RobotisOp2MotionManager(robot)
gyro = robot.getDevice("Gyro")
gyro.enable(timestep)
led_head = robot.getDevice("HeadLed")
led_eye = robot.getDevice("EyeLed")
led_head.set(0xFFFF00)
led_eye.set(0xFF0400)
motors = []
print(timestep)
position_sensors = []
for name in motorNames:
    motor = robot.getDevice(name)
    sensor = robot.getDevice(name + "S")
    sensor.enable(timestep)
    motors.append(motor)
    position_sensors.append(sensor)
# 传感器初始化（网页4示例）
robot.accelerometer = robot.getDevice("Accelerometer")
robot.accelerometer.enable(timestep)

def wait(ms):
    start_time = robot.getTime()
    while (robot.getTime() - start_time) * 1000 < ms:
        my_step()
def my_step():
    if robot.step(timestep) == -1:
        sys.exit(0)
        
my_step()
# 初始化动作
motion_manager.playPage(9)  # 初始姿势
motors[-1].setPosition(0.8)
wait(200)
my_step()

    
# Main loop:
# - perform simulation steps until Webots is stopping the controller
while True:
    # Read the sensors:
    # Enter here functions to read sensor data, like:
    #  val = ds.getValue()
    # data = lidar.getPointCloud()
    # Process sensor data here.
    # gait_manager.setXAmplitude(1.0)
    gait_manager.setYAmplitude(0.5)
    gait_manager.setAAmplitude(0.0)
    
    
    # random_action = np.random.randint(0, 4)
    # robot.execute_action(random_action)
    # print(random_action)
    
    # 步态更新
    gait_manager.step(timestep)
    my_step()
    # Enter here functions to send actuator commands, like:

# Enter here exit cleanup code.
