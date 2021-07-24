from  serial import Serial
import time

ser = Serial('/dev/ttyACM0', 9600, timeout=1)
ser.flush()
#while True:
time.sleep(10)
num = 1
ser.write(bytes(f'{num}\n', 'utf-8'))
time.sleep(5)
