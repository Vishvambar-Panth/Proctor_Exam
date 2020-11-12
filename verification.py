from matplotlib import pyplot as plt
from mtcnn.mtcnn import MTCNN
from matplotlib.patches import Rectangle
import numpy as np
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from scipy.spatial.distance import cosine
import cv2
from PIL import Image
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def extract_face(filename, required_size=(224, 224)):
	# load image from file
	pixels = plt.imread(filename)
	# create the detector, using default weights
	detector = MTCNN()
	# detect faces in the image
	results = detector.detect_faces(pixels)
	# extract the bounding box from the first face
	x1, y1, width, height = results[0]['box']
	x2, y2 = x1 + width, y1 + height
	# extract the face
	face = pixels[y1:y2, x1:x2]
	# resize pixels to the model size
	image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = np.array(image)
	return face_array

def get_embeddings(filenames):
	# extract faces
	faces = [extract_face(f) for f in filenames]
	# convert into an array of samples
	samples = np.array(faces, 'float32')
	# prepare the face for the model, e.g. center pixels
	samples = preprocess_input(samples, version=2)
	# create a vggface model
	model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
	# perform prediction
	yhat = model.predict(samples)
	return yhat

def is_match(known_embedding, candidate_embedding, thresh=0.4):
	# calculate distance between embeddings
	score = cosine(known_embedding, candidate_embedding)
	if score <= thresh:
		return True
		print('>MATCH (%.3f <= %.3f)' % (score, thresh))
	else:
		return False
		print('>NOT A MATCH (%.3f > %.3f)' % (score, thresh))


def take_picture():
    press = False
    while True:
        ret, frame = cam.read()
        frame = cv2.flip(frame,1)
        s_img = cv2.imread("outline.png", -1)
        s_img = cv2.resize(s_img, (frame.shape[1]//2, frame.shape[1]//2))
        x_offset = frame.shape[1]//5
        y_offset = frame.shape[0]//5
        y1, y2 = y_offset, y_offset + s_img.shape[0]
        x1, x2 = x_offset, x_offset + s_img.shape[1]

        alpha_s = s_img[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s

        for c in range(0, 3):
            frame[y1:y2, x1:x2, c] = (alpha_s * s_img[:, :, c] +
                                    alpha_l * frame[y1:y2, x1:x2, c])
        if not ret:
            print("failed to grab frame")
            break

        k = cv2.waitKey(1)
        if k%256 == 32:
            # SPACE pressed
            img_name = "capture.jpg"
            _, cap_file = cam.read()
            cv2.imwrite(img_name, cap_file)
            print("{} written!".format(img_name))
            filenames = ['id.jpg', 'capture.jpg']
            embeddings = get_embeddings(filenames)
            press = True
        
        if press:
            if is_match(embeddings[0], embeddings[1]):
                cv2.putText(frame, "MATCH " , (10,50), cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,0), 1)
            else:
                cv2.putText(frame, "NOT A MATCH: " ,(10,50),  cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,255), 1)
                
        cv2.imshow("test", frame)
        if k & 0xFF == ord('q') :
            break

    return

cam = cv2.VideoCapture(0)
cv2.namedWindow("test")
take_picture()


cam.release()
cv2.destroyAllWindows()
