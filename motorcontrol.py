import RPi.GPIO as io

def l(val):
    left.ChangeDutyCycle(val)

def r(val):
    right.ChangeDutyCycle(val)

def b(val):
    left.ChangeDutyCycle(val)
    right.ChangeDutyCycle(val)

io.setmode(io.BCM)
io.setwarnings(False)

left_pinout = 12
right_pinout = 13

io.setup(left_pinout, io.OUT)
io.setup(right_pinout, io.OUT)

left = io.PWM(left_pinout, 100)
right = io.PWM(right_pinout, 100)

left.start(0)
right.start(0)

