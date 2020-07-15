from matplotlib import pyplot
from matplotlib.patches import Rectangle
from mtcnn import MTCNN
import tensorflow as tf

def draw_image_with_boxes(filename,result_list):
    data = pyplot.imread(filename)
    pyplot.imshow(data)
    ax = pyplot.gca()

    for result in result_list:
        x,y,width,height = result['box']
        rect = Rectangle((x,y),width,height,fill=False,color = 'blue')
        ax.add_patch(rect)
    pyplot.show()

filename = 'abc.jpg'

pixels = pyplot.imread(filename)
detector = MTCNN()

faces = detector.detect_faces(pixels)
draw_image_with_boxes(filename,faces)
