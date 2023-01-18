from PIL import Image
from IPython.display import display
import cv2
img = cv2.imread('C:\Users\LENOVO\Documents', 1)
img = cv2.resize(img, (640, 400))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
display(Image.fromarray(img))
img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
display(Image.fromarray(img_grey))
img_inv = cv2.bitwise_not(img_grey)
display(Image.fromarray(img_inv))
img_smooth = cv2.GaussianBlur(img_inv, (5, 5), 0, 0)
display(Image.fromarray(img_smooth))
def dodge(img_grey, img_smooth):
  img_smooth_inv = 255 - img_smooth
  return cv2.divide(img_grey, img_smooth_inv, scale=256.0)
img_final = dodge(img_grey, img_smooth)
display(Image.fromarray(img_final))
display(Image.fromarray(img))
display(Image.fromarray(img_final))
