
"""
Created on Wed Oct 18 15:22:03 2023

@author: Jayakrishna.S.S
"""

# Program To Read video 
# and Extract Frames 

import cv2 

# Function to extract frames 
def FrameCapture(path): 

	# Path to video file 
	vidObj = cv2.VideoCapture(path) 

	# Used as counter variable 
	count = 500

	# checks whether frames were extracted 
	success = 1

	while success: 

		# vidObj object calls read 
		# function extract frames 
		success, image = vidObj.read() 

		# Saves the frames with frame-count 
		cv2.imwrite("frame%d.png" % count, image) 

		count += 1


# Driver Code 
if __name__ == '__main__': 

	# Calling the function 
	FrameCapture("C:/Users/jayak/WORK FLOW/solo/VID00026.AVI") 
