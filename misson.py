from pymavlink import mavutil

import time
# Create the connection
master = mavutil.mavlink_connection('udpin:192.168.2.1:14550')
# Wait a heartbeat before sending commands
def heart():
    try:
        master.wait_heartbeat()
    except:
        time.sleep(1)
        heart()

heart()

# https://mavlink.io/en/messages/common.html#MAV_CMD_COMPONENT_ARM_DISARM

# Arm
# master.arducopter_arm() or:
def arm():
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,
        1, 0, 0, 0, 0, 0, 0)

    # wait until arming confirmed (can manually check with master.motors_armed())
    print("Waiting for the vehicle to arm")
    print('Armed!')


def forward():
    print("forward")
    master.mav.manual_control_send(
        master.target_system,
        500,
        0,
        0,
        0,
        0,
        0)


def backward():
    print("backward")
    master.mav.manual_control_send(
        master.target_system,
        -500,
        0,
        0,
        0,
        0,
        0)


def left():
    print("left")
    master.mav.manual_control_send(
        master.target_system,
        0,
        -500,
        0,
        0,
        0,
        0)


def right():
    print("right")
    master.mav.manual_control_send(
        master.target_system,
        0,
        500,
        0,
        0,
        0,
        0)


def up():
    print("up")
    master.mav.manual_control_send(
        master.target_system,
        0,
        0,
        250,
        0,
        0,
        0)
    time.sleep(1)

def down():
    print("down")
    master.mav.manual_control_send(
        master.target_system,
        0,
        0,
        -250,
        0,
        0,
        0)
    time.sleep(1)

def yow_cc():
    print("yow_cc")
    master.mav.manual_control_send(
        master.target_system,
        0,
        0,
        0,
        100,
        0,
        0)
    time.sleep(1)

def yow_c():
    print("yow_c")
    master.mav.manual_control_send(
        master.target_system,
        0,
        0,
        0,
        -100,
        0,
        0)
    time.sleep(1)

def pitch_forward():
    print("pitch_forward")
    master.mav.manual_control_send(
        master.target_system,
        0,
        0,
        0,
        0,
        100,
        0)
    time.sleep(1)

def pitch_backward():
    print("pitch_backward")
    master.mav.manual_control_send(
        master.target_system,
        0,
        0,
        0,
        0,
        -100,
        0)
    time.sleep(1)


def roll_right():
    print("roll_right")
    master.mav.manual_control_send(
        master.target_system,
        0,
        0,
        0,
        0,
        0,
        100)
    time.sleep(1)

def roll_left():
    print("roll_left")
    master.mav.manual_control_send(
        master.target_system,
        0,
        0,
        0,
        0,
        0,
        -100)
    time.sleep(1)

def compass_value():
    msg = master.recv_match(type='VFR_HUD',blocking=True)
    print('heading',end='==>')
    print((msg.heading))
    return msg.heading

def pid(head):
    try:
        current = int(compass_value())
        while head == current:
            forward()
            current = int(compass_value())
        else:

            while head != current:
                current = int(compass_value())
                if head > current:
                    right()
                elif head<current:
                    left()
                current = int(compass_value())
    except:

        arm()
        heart()
        pass


compass_value()
head = compass_value()
while True:

    pid(head)

