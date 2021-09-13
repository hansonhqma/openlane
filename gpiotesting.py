import RPi.GPIO as io

io.setmode(io.BCM)

a_pinout = int(input("Motor A GPIO: "))
b_pinout = int(input("Motor B GPIO: "))

io.setup(a_pinout, io.OUT)
io.setup(b_pinout, io.OUT)

a = io.PWM(a_pinout, 100)
b = io.PWM(b_pinout, 100)


