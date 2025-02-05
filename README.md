## Languages and Technologies used:

1. Software - Python 
   - run add_faces.py adds faces in the dataset folder and trains the model and stores the embedding and saves the model as 	pkl file in data folder.
   - run test.py tests the model in real time using the trained model by taking into consideration HaarCascade model. Loads 	the representations of the new faces captured, if any.
2. Hardware - Embedded C
3. Algorithms - KNN and DeepFace - KNN does not accurately predict and detects the faces but DeepFace model detects and recognizes faces very well and returns a bounding box and confidence scores for each face detected.
4. Model - FaceNet - Pre-trained model developed by Google and stord in a haarcascaded file. Which is used to train our 	model from scratch.
5. Hardware Implementation 
   - Dual Motor Drivers connected to Arduino Mega 
   - Ultrasonic sensors are used for navigation
   - Led Acid Bateries - 2*6V
   - DC Motors
   - Rasberry Pi 4B for integration of Face detection model along with pi camera module.
   - IR sensor for boundary detection and self navigation
   - Programmed through Embedded C using Arduino IDE 2.3.4

6. Final Product - Self navigating robot along with locally streamed face detection.

