import time
import numpy as np
import tensorflow as tf
from picamera import PiCamera
import picamera.array as array
from picamera.array import PiRGBArray
from PIL import Image
import RPi.GPIO as GPIO

# Pin Definitions for sensors and H-bridge connections
sensor_front = 3  
sensor_rear = 26    #sensors will be ganged in pairs; only two pins required
sensor_bottom_front = 2
sensor_bottom_rear = 19

in1_motor = 7
in2_motor = 1     #output to H-bridge, in1/2 as labeled on H-bridge
en_motor = 12      #pwm pin to H-bridge

in3_led = 8  
in4_led = 25
en_led = 13       #pwm pin to H-bridge 

# Other globals for motor and LED power levels
Power_motor = 90  
power_led = 100    

move_time=0.2
after_picture_wait= 0.2
Stop = False 

motor_direc = 'stop'    #valad values "stop", "forward", "backward"

# Probability array for decision making based on model output
prob_count=0               #The probability array counter
probability_array_length=3
prob_array=np.zeros(probability_array_length)
prob_factor=0.4



# GPIO Setup for input and output pins
GPIO.setmode(GPIO.BCM)

# Setting up GPIO pins for various sensors and motors
GPIO.setup(sensor_front, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(sensor_rear, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(sensor_bottom_front, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(sensor_bottom_rear, GPIO.IN, pull_up_down=GPIO.PUD_UP)

GPIO.setup(in1_motor, GPIO.OUT)
GPIO.setup(in2_motor, GPIO.OUT)
GPIO.setup(en_motor, GPIO.OUT)

GPIO.setup(in3_led, GPIO.OUT)
GPIO.setup(in4_led, GPIO.OUT)
GPIO.setup(en_led, GPIO.OUT)


# PWM Setup for motor and LED control
pwm_motor = GPIO.PWM(en_motor, 100)
pwm_motor.start(0)
pwm_led = GPIO.PWM(en_led, 100)
pwm_led.start(0)                        #begin both off (pwm val of 0)
GPIO.output(in3_led, GPIO.HIGH)
GPIO.output(in4_led, GPIO.LOW)
pwm_led.ChangeDutyCycle(0)




def control_led(brightness):
    """ Adjust LED brightness """
    pwm_led.ChangeDutyCycle(brightness)
 

# Camera setup for capturing images
camera= PiCamera()
camera.resolution = (1648,1232)
camera.framerate = 30
raw_capture = PiRGBArray(camera, size = (1648,1232))

# TensorFlow Lite model setup
model_path = '/home/pi/Desktop/notebook_identification/dec10arch1.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def take_picture():
    """ Capture and preprocess an image for model inference """
    camera.capture(raw_capture, format='bgr', use_video_port=True)
    img_array = raw_capture.array
    preprocessed_image = preprocess_image(img_array)
    raw_capture.truncate(0)
    return preprocessed_image


  
def preprocess_image(image):
    """ Preprocess image for the TensorFlow model """
    IMG_WIDTH, IMG_HEIGHT = 320, 240
    image = Image.fromarray(image)
    image = image.convert('L')  # Convert to grayscale
    image = image.resize((IMG_WIDTH, IMG_HEIGHT), Image.ANTIALIAS)  # Ensure correct size
    image = np.array(image, dtype=np.float32)  # Ensure dtype is float32
    image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

def run_tflite(img):
    """ Run TensorFlow Lite model inference """
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output_val = interpreter.get_tensor(output_details[0]['index'])
    return output_val

def move_step():
    """ Control motor movement and LED based on sensor input """
    global Stop, motor_direc
    if Stop:
        GPIO.output(in1_motor, GPIO.LOW)
        GPIO.output(in2_motor, GPIO.LOW)
        pwm_motor.ChangeDutyCycle(0)
        control_led(100)
    else:
        Stop= False
        if motor_direc == 'forward':
            GPIO.output(in1_motor, GPIO.HIGH)
            GPIO.output(in2_motor, GPIO.LOW)
            pwm_motor.ChangeDutyCycle(Power_motor)
            control_led(0)
        elif motor_direc == 'backward':
            GPIO.output(in1_motor, GPIO.LOW)
            GPIO.output(in2_motor, GPIO.HIGH)
            pwm_motor.ChangeDutyCycle(Power_motor)
            control_led(0)
        pwm_motor.ChangeDutyCycle(Power_motor)


def move(move_duration):
    """ Perform a movement step """
    check_sensors()
    printalert()
    move_step()
    time.sleep(move_duration/2)
    check_sensors()
    printalert()
    move_step()
    time.sleep(move_duration/2)
    pwm_motor.ChangeDutyCycle(0)  # Stop after moving
    check_sensors()
    printalert()


def printalert():
    """ Debugging function to print sensor status """
    if GPIO.input(sensor_rear) == GPIO.LOW:
        print("rear")
    elif GPIO.input(sensor_bottom_rear) == GPIO.HIGH:
        print("bottom rear")
    elif GPIO.input(sensor_front) == GPIO.LOW:
        print( "front")
    elif GPIO.input(sensor_bottom_front):
        print("bottom front")

    


def check_sensors():
    """ Check sensor inputs to determine movement direction """
    global motor_direc, Stop
    if (GPIO.input(sensor_rear) == GPIO.LOW or GPIO.input(sensor_bottom_rear) == GPIO.HIGH) and (GPIO.input(sensor_front) == GPIO.LOW or GPIO.input(sensor_bottom_front) == GPIO.HIGH):
        Stop = True 
    elif  GPIO.input(sensor_rear) == GPIO.LOW or GPIO.input(sensor_bottom_rear) == GPIO.HIGH:
        motor_direc = 'forward'
    
    elif GPIO.input(sensor_front) == GPIO.LOW or GPIO.input(sensor_bottom_front) == GPIO.HIGH:
        motor_direc = 'backward'

def picture():
    """ Take a picture, run model inference, and set Stop flag if notebook is detected """
    global Stop, prob_count
    cur_pic = take_picture()
    model_output = run_tflite(cur_pic)
    prob_array[prob_count % probability_array_length] = model_output
    prob_count+=1
    Stop = True if prob_array.sum() > (prob_factor * probability_array_length) else False
    
    return 
try:
    while True:
        move(move_time)  # Move a step
        picture()  # Take and process picture after moving
        time.sleep(after_picture_wait)

except KeyboardInterrupt:
    # Cleanup routine for safe shutdown
    print("except run")
    camera.close()
    pwm_motor.stop()
    pwm_led.stop()
    GPIO.cleanup()

