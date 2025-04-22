"""test controller."""
import sys
# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot
from managers import RobotisOp2GaitManager
# create the Robot instance.
robot = Robot()

NMOTORS = 20
motorNames = [
    "ShoulderR", "ShoulderL", "ArmUpperR", "ArmUpperL", "ArmLowerR",
    "ArmLowerL", "PelvYR", "PelvYL", "PelvR", "PelvL", "LegUpperR",
    "LegUpperL", "LegLowerR", "LegLowerL", "AnkleR", "AnkleL", "FootR",
    "FootL", "Neck", "Head"
]

timestep = int(robot.getBasicTimeStep())
motors = []
position_sensors = []
for name in motorNames:
    motor = robot.getDevice(name)
    sensor = robot.getDevice(name + "S")
    sensor.enable(timestep)
    motors.append(motor)
    position_sensors.append(sensor)
# get the time step of the current world.
accelerometer = robot.getDevice("Accelerometer")
accelerometer.enable(timestep)
gyro = robot.getDevice("Gyro")
gyro.enable(timestep)
def my_step(self):
    if robot.step(self.timeStep) == -1:
        sys.exit(0)
# You should insert a getDevice-like function in order to get the
# instance of a device of the robot. Something like:
#  motor = robot.getDevice('motorname')
#  ds = robot.getDevice('dsname')
#  ds.enable(timestep)
gait_manager = RobotisOp2GaitManager(robot, "config.ini")
gait_manager.setBalanceEnable(True)
gait_manager.start()
# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step(timestep) != -1:
    # Read the sensors:
    # Enter here functions to read sensor data, like:
    #  val = ds.getValue()
    gait_manager.setXAmplitude(1.0)
    gait_manager.setAAmplitude(-0.5)
    i = 0
    if i >= 10:
        gait_manager.setXAmplitude(1.0)
        gait_manager.setAAmplitude(-0.5)
        i += 1
    # Process sensor data here.
    gait_manager.step(timestep)
    # Enter here functions to send actuator commands, like:
    #  motor.setPosition(10.0)
    pass

# Enter here exit cleanup code.
