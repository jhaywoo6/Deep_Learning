import multiprocessing
from multiprocessing import Event, Queue, Value, Process
import gpiozero
import board
import digitalio
import adafruit_max31855
import busio
import adafruit_ads1x15.ads1115 as ADS
import RPi.GPIO as GPIO
from adafruit_ads1x15.analog_in import AnalogIn
from time import sleep
import os
from os import system
import csv
import gi
import time

# MAX31855
# Pin # 23, 24, 21, 11, 13, 15
# GPIO # 11, 8, 9, 17, 27, 22
# SCK, CS, S0, T0, T1, T2
SCK = 11
CS = 8
S0 = 9
T0 = 17
T1 = 13
T2 = 15

# ADS1115
# Pin # 3 5
# GPIO # 2 3
# SDA1, SCL1
# Note, doesn't seem to mix well with screen GPIO, but screen works without these so idk.

# Gas Input
# Pin # 40
# GPIO # 21

Gas_Square_In = 21

gi.require_version("Gtk", "3.0")
from gi.repository import Gtk, GLib
"""
This is a library for the octo MAX31855 thermocouple breakout board.

MIT License

Copyright (c) 2020 Mitchell Herbert

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files
(the "Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import RPi.GPIO as GPIO
from time import sleep

class MAX31855:
	"""
	Initializes GPIO pins and instance variables.
	CS initializes to high because it is active low.
	SO is an input pin.

	:param SCK: the BCM pin number of the SCK line
	:param CS: the BCM pin number of the CS line
	:param SO: the BCM pin number of the SO line
	:param T0: the BCM pin number of the T0 line
	:param T1: the BCM pin number of the T1 line
	:param T2: the BCM pin number of the T2 line
	"""
	def __init__(self, SCK, CS, SO, T0, T1, T2):
		GPIO.setmode(GPIO.BCM)
		# Setup all of the GPIO pins
		for pin_number in [SCK, T0, T1, T2]:
			GPIO.setup(pin_number, GPIO.OUT)
			GPIO.output(pin_number, 0)
		GPIO.setup(CS, GPIO.OUT)
		GPIO.output(CS, 1)
		GPIO.setup(SO, GPIO.IN)
		# Initialize instance variables
		self.SCK = SCK
		self.CS = CS
		self.SO = SO
		self.T0 = T0
		self.T1 = T1
		self.T2 = T2
		# Initialize the poll data to zero
		self.latest_data = 0b0

	"""
	Communicates with the octo MAX31855 board to retrieve
	temperature and fault data. The data is stored in
	self.latest_data for later reference.

	:param therm_id: id of the thermocouple (0 - 7)
	"""
	def read_data(self, therm_id):
		# Select the thermocouple using multiplexer
		GPIO.output(self.T2, therm_id & 0b100)
		GPIO.output(self.T1, therm_id & 0b10)
		GPIO.output(self.T0, therm_id & 0b1)
		# Wait for the multiplexer to update
		sleep(0.125)
		# Select the chip and record incoming data
		data = 0b0
		GPIO.output(self.CS, 0)
		# Shift in 32 bits of data
		for bitshift in reversed(range(0, 32)):
			GPIO.output(self.SCK, 1)
			data += GPIO.input(self.SO) << bitshift
			GPIO.output(self.SCK, 0)
		GPIO.output(self.CS, 1)
		self.latest_data = data

	"""
	Gets the temperature of the most recently polled
	thermocouple.

	:returns: float representing the temperature in celsius
	"""
	def get_thermocouple_temp(self):
		data = self.latest_data
		# Select appropriate bits
		data = data >> 18
		# Handle twos complement
		if data >= 0x2000:
			data = -((data ^ 0x3fff) + 1)
		# Divide by 4 to handle fractional component
		return data / 4

	"""
	Gets the temperature of the reference junction from
	the most recent poll.

	:returns: float representing the temperature in celsius
	"""
	def get_reference_temp(self):
		data = self.latest_data
		# Select appropriate bits
		data = (data & 0xfff0) >> 4
		# Handle twos complement
		if data & 0x800:
			data = -((data ^ 0xfff) + 1)
		# Divide by 16 to handle fractional component
		return data / 16

	"""
	Returns a value signififying a particular fault in the most
	recent poll.

	0 indicates that no faults exist
	1 indicates an SCV fault (thermocouple is shorted to VCC)
	2 indicates an SCG fault (thermocouple is shorted to GND)
	3 indicates an OC fault (the thermocouple is not connected)

	:returns: an integer representing the fault
	"""
	def get_faults(self):
		data = self.latest_data
		if data & 0x00010000:
			if data & 0b100:
				return 1
			if data & 0b10:
				return 2
			if data & 0b1:
				return 3
		return 0

	"""
	Should be called at the end of program execution to bring
	all GPIO pins to a 'safe' state.
	"""
	def cleanup(self):
		GPIO.cleanup()


	"""
	Returns the value of latest_data

	:returns: the value of latest_data
	"""
	def get_latest_data(self):
		return self.latest_data


def getData(queue, timeTotal, endDataCollect):
	
	Temperature = MAX31855(SCK, CS, S0, T0, T1, T2)
	
	gasFlow = queue.get()
	tempAvg = queue.get()
	wattage = queue.get()
	timeCurTest = queue.get()
	timeTotal = queue.get()
	gasUsage = queue.get()
	gasTotalUsage = queue.get()
	RotateRead = -1
	
	gasCurrentTest = len(timeTotal)-1
	DataCollectFrequency = 0.5 # Collects data every DataCollectFrequency seconds. Note: low values will increase the data collection speed, but currently may freeze the program. Modify with care.
	timeCurTest[-1] = 0

	while not endDataCollect.is_set():
		
		if len(gasFlow) >= 600:
			gasFlow.pop(0)
			tempAvg.pop(0)
			wattage.pop(0)
			timeCurTest.pop(0)
			timeTotal.pop(0)
			gasUsage.pop(0)
			gasTotalUsage.pop(0)
		
		# gasFlow.append()
		RotateRead += 1
		if RotateRead == 8:
			RotateRead = 0
		Temperature.read_data(0)
		tempAvg.append(Temperature.get_thermocouple_temp() * 9 / 5 + 32)
		# Temperature.get_thermocouple_temp()
		wattage.append(wattChan.value) # Requires ADS1115 to run
		
		timeCurTest.append(timeCurTest[-1] + DataCollectFrequency)
		timeTotal.append(timeTotal[-1] + DataCollectFrequency)
		
		gasUsage.append(sum(gasFlow[gasCurrentTest:])*DataCollectFrequency)   # This is measured as ft^3/s. As the data is collected every second, the total can be gotten through summation
		gasTotalUsage.append(sum(gasFlow)*DataCollectFrequency)

		queue.put(gasFlow)
		queue.put(tempAvg)
		queue.put(wattage)
		queue.put(timeCurTest)
		queue.put(timeTotal)
		queue.put(gasUsage)
		queue.put(gasTotalUsage)
		
		sleep(DataCollectFrequency)



def count_rising_edges(queue):
	edge_count = 0
	last_state = GPIO.LOW
	gasUsage = 0.00

	try:
		while True:
			current_state = GPIO.input(Gas_Square_In)
			# Detect rising edge
			if current_state == GPIO.HIGH and last_state == GPIO.LOW:
				edge_count += 1

				# Increment the counter every 3 rising edges
				if edge_count == 3:
					gasUsage += 0.01
					queue.put(gasUsage)
					print("0.01 added to the queue")
					edge_count = 0

			last_state = current_state
			time.sleep(0.001)  # Small delay to debounce
	except KeyboardInterrupt:
		print("Stopping process...")
	finally:
		GPIO.cleanup()

class ProgramLoop(Gtk.Window):
	
	def __init__(self, queue):
		super().__init__(title="Looping App")
		self.endDataCollect = Event()
		self.set_default_size(800, 480)
		self.set_border_width(8)

		self.dataProcess = None
		self.queue = queue
		
		GLib.timeout_add(100, self.check_queue)
        
		self.gasFlow = [0]
		self.tempAvg = [0]
		self.wattage = [0]
		self.timeCurTest = [0]
		self.timeTotal = [0]
		self.gasUsage = [0]
		self.gasTotalUsage = [0]
		
		
		
		self.motor = digitalio.DigitalInOut(board.D7)
		
		self.motor.direction = digitalio.Direction.OUTPUT
		
		self.motor.value = 0
		
		self.gasTotal = 0
        
		self.fileName = "Empty String"

		self.stack = Gtk.Stack()
		self.add(self.stack)
		
		self.output_directory = '/home/pengo/VULKAN_FRY/Outputs'
        
		# Screen 1: Naming the file
		self.nameFile1 = Gtk.Box(spacing=10, orientation=Gtk.Orientation.VERTICAL)
		self.nameFile1.set_vexpand(True)
		self.nameFile1.set_valign(Gtk.Align.START)
		
		self.nameFilelabel = Gtk.Label(label="Welcome to the simulated ASTM F1361 test apparatus.\nPlease read the user manual prior to setting up this test.\nEnsure that the sensors are affixed to the frier being tested.\nEnter a file name for saving the test.\nIf the file already exists, it will be overwritten.\nPress Enter to continue.")
		self.nameFilelabel.set_line_wrap(True)
		self.nameFilelabel.set_xalign(0)
		self.nameFilelabel.set_yalign(0)
		
		self.nameFileEntry = Gtk.Entry()
		self.nameFileEntry.connect("key-press-event", self.saveFileName)
		
		self.nameFile1.pack_start(self.nameFileEntry, False, False, 10)
		self.nameFile1.pack_start(self.nameFilelabel, False, False, 10)
		self.stack.add_named(self.nameFile1, "nameFile1")

		 # Screen 2: Waits for user input to begin test
		self.waitToBegin2 = Gtk.Box(spacing=10, orientation=Gtk.Orientation.VERTICAL)

		self.waitToBeginlabel = Gtk.Label(label="Press the button to begin the test.\nThis should start and run the motors for the duration of the test.\nIf the motors are running outside of the test, use the switches in the electrical cabinet to turn them off.\nDo not attempt another test and contact the VULKAN_FRY team for assistance.")
		self.waitToBeginlabel.set_line_wrap(True)

		self.waitToBeginbutton = Gtk.Button(label="Begin Test")
		self.waitToBeginbutton.connect("clicked", self.beginTest)

		self.waitToBegin2.pack_start(self.waitToBeginlabel, True, True, 0)
		self.waitToBegin2.pack_start(self.waitToBeginbutton, True, True, 0)
		self.stack.add_named(self.waitToBegin2, "waitToBegin2")

		# Screen 3: Waits for motors to turn on
		self.motorStartup3 = Gtk.Box(spacing=10, orientation=Gtk.Orientation.VERTICAL)

		self.motorStartuplabel = Gtk.Label(label="The motors should be turing on. If they do not, end the test and contact the VULKAN_FRY team.")
		self.motorStartuplabel.set_line_wrap(True)

		self.motorStartup3.pack_start(self.motorStartuplabel, True, True, 0)
		self.stack.add_named(self.motorStartup3, "motorStartup3")

		# Screen 4: Displays Data untill user Input
		self.dataCollection4 = Gtk.Box(spacing=10, orientation=Gtk.Orientation.VERTICAL)

		self.dataCollectionlabel = Gtk.Label(label="")
		self.dataCollectionlabel.set_line_wrap(True)

		self.dataCollectionbutton = Gtk.Button(label="End Test")
		self.dataCollectionbutton.connect("clicked", self.endTest)

		self.dataCollection4.pack_start(self.dataCollectionlabel, True, True, 0)
		self.dataCollection4.pack_start(self.dataCollectionbutton, True, True, 0)
		self.stack.add_named(self.dataCollection4, "dataCollection4")

		# Screen 5: Waits for motors to turn off
		self.motorWindDown5 = Gtk.Box(spacing=10, orientation=Gtk.Orientation.VERTICAL)

		self.motorWindDownlabel = Gtk.Label(label="The motors should be turing off. If they do not,end the test and contact the VULKAN_FRY team.")
		self.motorWindDownlabel.set_line_wrap(True)

		self.motorWindDown5.pack_start(self.motorWindDownlabel, True, True, 0)
		self.stack.add_named(self.motorWindDown5, "motorWindDown5")

		# Screen 6: Continue Testing Querry
		self.continueTestingQuerry6 = Gtk.Box(spacing=10, orientation=Gtk.Orientation.VERTICAL)

		self.TQbuttonNextTest = Gtk.Button(label="Click to begin the next test.")
		self.TQbuttonEndTesting = Gtk.Button(label="Click to end testing.")
		self.TQbuttonNextTest.connect("clicked", self.beginTest)
		self.TQbuttonEndTesting.connect("clicked", self.saveDataQuerry)

		self.continueTestingQuerry6.pack_start(self.TQbuttonNextTest, True, True, 0)
		self.continueTestingQuerry6.pack_start(self.TQbuttonEndTesting, True, True, 0)

		self.stack.add_named(self.continueTestingQuerry6, "continueTestingQuerry6")

		# Screen 7: Save Data to file Querry
		self.saveDataQuerry7 = Gtk.Box(spacing=10, orientation=Gtk.Orientation.VERTICAL)

		self.DQbuttonSave = Gtk.Button(label="Click to save test results to /home/pengo/VULKAN_FRY/Outputs. Program will restart.")
		self.DQbuttonReset = Gtk.Button(label="Click to restart program without saving.")
		self.DQbuttonSave.connect("clicked", self.saveData)
		self.DQbuttonReset.connect("clicked", self.resetProgram)

		self.saveDataQuerry7.pack_start(self.DQbuttonSave, True, True, 0)
		self.saveDataQuerry7.pack_start(self.DQbuttonReset, True, True, 0)

		self.stack.add_named(self.saveDataQuerry7, "saveDataQuerry7")

		# Screen 8: Saving data to file
		self.savingData8 = Gtk.Box(spacing=10, orientation=Gtk.Orientation.VERTICAL)

		self.savingDatalabel = Gtk.Label(label="Saving Data...")
		self.savingDatalabel.set_line_wrap(True)

		self.savingData8.pack_start(self.savingDatalabel, True, True, 0)
		self.stack.add_named(self.savingData8, "savingData8")

		# Screen 9: Data saved to file
		self.dataSaved9 = Gtk.Box(spacing=10, orientation=Gtk.Orientation.VERTICAL)

		self.dataSavelabel = Gtk.Label(label="Data Saved.")
		self.dataSavelabel.set_line_wrap(True)

		self.dataSaved9.pack_start(self.dataSavelabel, True, True, 0)
		self.stack.add_named(self.dataSaved9, "dataSaved9")
        
        
	
	def saveFileName(self, widget, event):
		from gi.repository import Gdk
		print(self.gasFlow)
		if event.keyval == Gdk.KEY_Return:
			self.fileName = self.nameFileEntry.get_text()
			self.stack.set_visible_child_name("waitToBegin2")
		
	
	def beginTest(self, *args):
		self.stack.set_visible_child_name("motorStartup3")
		self.motor.value = 1
		GLib.timeout_add(5000, self.startDataCollection)
		
	def startDataCollection(self):
		self.queue.put(self.gasFlow)
		self.queue.put(self.tempAvg)
		self.queue.put(self.wattage)
		self.queue.put(self.timeCurTest)
		self.queue.put(self.timeTotal)
		self.queue.put(self.gasUsage)
		self.queue.put(self.gasTotalUsage)
		self.endDataCollect.clear()
		self.dataProcess = multiprocessing.Process(
			target=getData, args=(self.queue, timeTotal, self.endDataCollect), daemon=True
		)
		self.GasProcess = multiprocessing.Process(
			target=count_rising_edges, args=(self.queue,)
		)
		self.dataProcess.start()
		self.GasProcess.start()
		self.stack.set_visible_child_name("dataCollection4")
		return False
		
	def check_queue(self):
		while not self.queue.empty():
			self.gasFlow = self.queue.get()
			self.tempAvg = self.queue.get()
			self.wattage = self.queue.get()
			self.timeCurTest = self.queue.get()
			self.timeTotal = self.queue.get()
			self.gasUsage = self.queue.get()
			self.gasTotalUsage = self.queue.get()
			if self.stack.get_visible_child_name() == "dataCollection4":
				dataUpdate = (
					f"gasFlow: {self.gasFlow[-1]}\n"
					f"tempAvg: {self.tempAvg[-1]}\n"
					f"wattage: {self.wattage[-1]}\n"
					f"timeCurTest: {self.timeCurTest[-1]}\n"
					f"timeTotal: {self.timeTotal[-1]}\n"
					f"gasUsage: {self.gasUsage[-1]}\n"
					f"gasTotalUsage: {self.gasTotalUsage[-1]}\n"
				)
				self.dataCollectionlabel.set_text(dataUpdate)
		return True 
        
	def endTest(self, *args):
		self.endDataCollect.set()
		self.dataProcess.join()
		self.GasProcess.join()
		self.stack.set_visible_child_name("motorWindDown5")
		self.motor.value = 0
		GLib.timeout_add(5000, self.continueTestingQuerry)
		
	def continueTestingQuerry(self):
		self.stack.set_visible_child_name("continueTestingQuerry6")
		return False
		
	def saveDataQuerry(self, *args):
		self.stack.set_visible_child_name("saveDataQuerry7")
		
	def saveData(self, *args):
		self.stack.set_visible_child_name("savingData8")
		
		os.makedirs(self.output_directory, exist_ok=True)
            

		with open(os.path.join(self.output_directory, self.fileName + '.csv'), 'w', newline='') as file:   # NOQA
			writer = csv.writer(file)

			writer.writerow([
				"Gas Flow Rate",
				"Temperature Average",
				"Wattage",
				"Current Test Time (sec)",
				"Total Time (sec)",
				"Gas Usage",
				"Gas Total Usage"
			])

			writer.writerows(zip(
				self.gasFlow,
				self.tempAvg,
				self.wattage,
				self.timeCurTest[1:],
				self.timeTotal[1:],
				self.gasUsage,
				self.gasTotalUsage
			))
                
		self.stack.set_visible_child_name("dataSaved9")
		GLib.timeout_add(5000, self.resetProgram)
		
	def resetProgram(self, *args):
		self.gasFlow = []
		self.tempAvg = []
		self.wattage = []
		self.timeCurTest = [0]
		self.timeTotal = [0]
		self.gasUsage = []
		self.gasTotalUsage = []
		self.stack.set_visible_child_name("nameFile1")
		return False
        
        
def main():
	global gasFlow, tempAvg, wattage, timeTotal, timeCurTest, gasUsage, gasTotalUsage
	global gasAnalogIn, max31855, wattChan
	gasFlow = []
	gasFlowTotal = []
	gasUsage = []
	gasTotalUsage = []
	tempAvg = []
	wattage = []
	timeCurTest = [0]
	timeTotal = [0]
	
	# Preparing Pins
	GPIO.setmode(GPIO.BCM)

	spi = board.SPI()   # Set up the Thermocoupler
	cs = digitalio.DigitalInOut(board.D5)
	max31855 = adafruit_max31855.MAX31855(spi, cs)

	i2c = busio.I2C(board.SCL, board.SDA)   # Set up the Wattage Sensor
	ads = ADS.ADS1115(i2c) # Requires ADS1115 to run
	wattChan = AnalogIn(ads, ADS.P0) # Requires ADS1115 to run
	
	GPIO.setup(Gas_Square_In, GPIO.IN)
	gasAnalogIn = digitalio.DigitalInOut(board.D6)   # TBD When Gas Flow Meter compatible with pi is found

	# Sets up two relays to control two pumps

	global motor
	motor = digitalio.DigitalInOut(board.D7)
	
	motor.direction = digitalio.Direction.OUTPUT

	motor.value = 0
	
	queue = multiprocessing.Queue()
	
	app = ProgramLoop(queue)
	app.connect("destroy", Gtk.main_quit)
	app.show_all()
	Gtk.main()
	
if __name__ == "__main__":
    main()
