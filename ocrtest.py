import pytesseract as ts
from PIL import Image


image = Image.open("/Users/liwenbo/Desktop/hanzi.jpg",'r')
s = ts.image_to_string(image)
print(s)
