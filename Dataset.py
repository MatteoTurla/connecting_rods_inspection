import cv2
import glob
from matplotlib import pyplot as plt

# menage all the images to analyse
class Dataset():

    def __init__(self, folder_path="./immagini/*.BMP"):
        """
        :param folder_path: path of the images to load, pass it as glob expression
        """
        self.images = self.init_dataset(folder_path)

    def init_dataset(self, folder_path):
        """
        :param folder_path: path of the images to load
        :return: array of numpy images
        """
        dataset_path = glob.glob(folder_path)
        images = []
        for path in dataset_path:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            images.append(img)
        return images

    def show(self):
        """
        show all the images in the dataset
        """
        for index, img in enumerate(self.images):
            plt.figure(index)
            plt.imshow(img, cmap="gray", vmin="0", vmax="255")
            plt.xticks([]), plt.yticks([])
            plt.show()

