import cv2
import numpy as np
import os

def dist(x1,x2):
    return np.sqrt(sum((x1-x2)**2))

def knn(X,Y,queryPoint,k=5):
    
    vals = []
    m = X.shape[0]
    
    for i in range(m):
        d = dist(queryPoint,X[i])
        vals.append((d,Y[i]))
        
    
    vals = sorted(vals)
    # Nearest/First K points
    vals = vals[:k]
    
    vals = np.array(vals)
    
    #print(vals)
    
    new_vals = np.unique(vals[:,1],return_counts=True)
    #print(new_vals)
    
    index = new_vals[1].argmax()
    pred = new_vals[0][index]
    
    return pred
    

	
	

dataset_path = './data/'
face_Data=[]
class_id =0
label = []
names= {}




for fx in os.listdir(dataset_path):
	if fx.endswith('.npy'):
		names[class_id]=fx[:-4]
		data_load = np.load(dataset_path+fx)
		face_Data.append(data_load)
		target = class_id*np.ones((data_load.shape[0] , ))
		class_id+=1
		label.append(target)

face_dataset = np.concatenate(face_Data ,axis =0)
labels = np.concatenate(label ,axis =0)





cap=cv2.VideoCapture(0)
skip = 0
face_Data =[]
face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

while True:
	ret , frame = cap.read()
	if ret ==False:
		continue 
	faces = face_classifier.detectMultiScale(frame , 1.3 , 5)
	
	for face in faces:
		x , y ,w ,h  =face
		face_section = frame[y-10 : y+h+10 , x-10 : x+w+10]
		face_section = cv2.resize(face_section , (100 , 100))
		out = knn(face_dataset , labels  , face_section.flatten() )
		predname = names[int(out)]
		cv2.putText(frame , predname , (x , y-10) , cv2.FONT_HERSHEY_SIMPLEX  , 1 ,  (255 , 255 , 255) , 	 2 , cv2.LINE_AA)
		cv2.rectangle(frame , (x ,y ) , (x+w , y+h) , (0 , 0 ,0 ) , 2)
		cv2.imshow('frame',frame)
	
	key = cv2.waitKey(1) & 0xFF
	if(key == ord('q')):
		break
cap.release()
cv2.destroyAllWindows()



