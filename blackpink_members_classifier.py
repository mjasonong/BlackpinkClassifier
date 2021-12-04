import cv2
from tensorflow.keras.models import load_model
import numpy as np

model = load_model('./model/bp_cnn.h5')
face_clsfr = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_frontalface_default.xml')
classes = ["Jennie", "Jisoo", "Lisa", "Rose"]
color_dict = {0: (51,153,255), 1: (255,105,180)}

print(model.summary())

source = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    success, image = source.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_clsfr.detectMultiScale(gray, 1.32, 4)

    for (x, y, w, h) in faces:
        face_image = image[y:y+w, x:x+h]
        resize_image = cv2.resize(face_image, (180, 180))
        normalized_image = resize_image / 255.0
        reshaped_image = np.reshape(normalized_image, (1, 180, 180, 3))
        result = model.predict(reshaped_image)
        label = np.argmax(result)
        confidence = round((100 * (np.max(result[0]))),0)
        percentage = f"{confidence}%"

        if confidence > 60:
            cv2.rectangle(image, (x,y), (x+w, y+h), color_dict[1], 2)
            cv2.putText(image, classes[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(image, str(percentage), (x+70, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow("Blackpink Classifier", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
source.release()