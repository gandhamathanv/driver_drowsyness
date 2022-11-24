# import the opencv library
import cv2


# define a video capture object
vid = cv2.VideoCapture(5)
print(vid)
while(True):
	# Capture the video frame
	# by frame
	ret, frame = vid.read()
	print(ret,frame)
	cv2.imshow('frame', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
								break
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
