from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img
import random,os,glob
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Testing Image
imgPath = '/Users/ruthvikpedibhotla/Documents/Python/imgTest.jpg'
img=mpimg.imread('/Users/ruthvikpedibhotla/Documents/Python/imgTest.jpg')
imgplot = plt.imshow(img)

img = image.load_img(imgPath, target_size=(64, 64))
img = image.img_to_array(img, dtype=np.uint8)

# Loading the Model
model=load_model('/Users/ruthvikpedibhotla/Documents/Python/model.h5')
model.summary()
p=model.predict(img[np.newaxis, ...])

# Printing the Class
print("Maximum Probability: ",np.max(p[0], axis=-1))
predicted_class=np.argmax(p[0], axis=-1)
if predicted_class == 0:
	item = "Cardboard"
elif predicted_class == 1:
	item = "Glass"
elif predicted_class == 2:
	item = "Metal"
elif predicted_class == 3:
	item = "Paper"
elif predicted_class == 4:
	item = "Plastic"
else:
	item = "Landfill"

print('Classified:', item)

# Showing Image
plt.show()
plt.title('Loaded Image')
plt.axis('off')
plt.imshow(img.squeeze())
