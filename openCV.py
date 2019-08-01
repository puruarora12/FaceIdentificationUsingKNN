import cv2
import numpy as np
cap=cv2.VideoCapture(0)
skip = 0
face_Data =[]
face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
datapath = './data/'
file_name = input('Enter your name : ')
while True:

	ret , frame = cap.read()
	if ret == False:
		continue
	faces = face_classifier.detectMultiScale(frame , 1.3 , 5)
	#print(faces)
	for face in faces :
		x , y, w , h  =face
		cv2.rectangle(frame , (x ,y) , ((x+w) , (y+h)) , (255 , 0 , 255) , 2)
	faces = sorted(faces ,key = lambda f: f[2]*f[3] )#,reverse=True)
	face_section = frame[y-10  : y+10+h  , x-10 : x+10+w ]
	face_section=cv2.resize(face_section , (100,100))
	skip+=1
	if skip%10==0:
		face_Data.append(face_section)
		#print(face_section)
		print(len(face_Data))
	cv2.imshow('webcam test' ,frame)
	cv2.imshow('Selected faces ' , face_section)
	#print('/////////////////////////////////////////////////////////////////////////////////////')
	#print(face_section.shape)
	#print('/////////////////////////////////////////////////////////////////////////////////////')
	key  =cv2.waitKey(1) & 0xFF
	
	if key == ord('q'):
		break
		
face_Data = np.asarray(face_Data)
face_Data=face_Data.reshape((face_Data.shape[0] , -1))		
print(face_Data.shape)
np.save(datapath+file_name+'.npy' , face_Data)		
cap.release()
cv2.destroyAllWindows()
print("data saved succesfuly")








