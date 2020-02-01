import cv2
from matplotlib import pyplot as plt
import numpy as np
from queue import PriorityQueue
from MatchingProblem import MatchingProblem
from BlobAnalyzer import BlobAnalyzer
from search import astar_search

class Segmenter:

    def __init__(self):
        """
        segment a given image:
        1. gaussian denoising
        2. otsu's threshold
        3. morphological operation to remove noise and erode iron powder
        5. remove connection between attached objects
        6. labelling of connected components
        """
        self.analyzer = BlobAnalyzer()

    def segmentation(self, image, show):
        """
        denoise the image with a gaussian 5x5 kernel and then apply otsu's segmentation
        :param image: gray image to segment
        :param show: if true show the step's result of the function
        :return: threshold image, 1 foregorund, 0 background
        """

        blur = cv2.GaussianBlur(image, (5, 5), 0)
        ret, th = cv2.threshold(blur, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # 0 background, 1 foreground
        th = 1 - th
        if show:
            plt.figure()
            plt.imshow(image, 'gray')
            plt.title('original image')
            plt.xticks([]), plt.yticks([])
            plt.show()

            print("-"*30)
            print("threshold step")
            print("threshold used: ", ret)
            print("range of values of threshold image: ", np.unique(th))
            plt.figure(1)
            plt.subplot(2, 2, 1)
            plt.title("original image")
            plt.imshow(image, 'gray')
            plt.xticks([]), plt.yticks([])
            plt.subplot(2, 2, 2)
            plt.title("hist of original image")
            plt.hist(image.ravel(), 256)
            plt.subplot(2, 2, 3)
            plt.imshow(blur, 'gray')
            plt.xticks([]), plt.yticks([])
            plt.subplot(2, 2, 4)
            plt.hist(blur.ravel(), 256)
            plt.show()

            plt.figure(2)
            plt.title("binary image")
            plt.imshow(th, 'gray')
            plt.xticks([]), plt.yticks([])
            plt.show()
        return th

    def fill_hole(self, im, show):
        """
        fill holes of binary image, in order to erode iron powders
        :param im: binary image, 1 foreground, 0 background
        :param show: if true show results of computation
        :return: binary image with filled holes
        """

        # idea: connected component on background, then set to 0 the label of the background, if there are more labels, they referer
        # to holes in the image
        M,N = im.shape
        for i in range(M):
            for j in range(N):
                if im[i,j] == 0:
                    bg_pos = (i,j)
                    flag = False
                    break

        background = 1 - im
        n_labels, labels = cv2.connectedComponents(background, 8)
        holes = np.zeros(labels.shape, np.uint8)
        background_label = labels[bg_pos]
        # set to 1 the pixel belonging to a hole
        for i in range(1, n_labels):
            if not i == background_label:
                holes[np.where(labels == i)] = 1
        filled = np.bitwise_or(im, holes)

        if show:
            print("-"*30)
            print("filling holes")
            print("range of values of filled image: ", np.unique(filled))
            plt.figure(3)
            plt.subplot(1, 2, 1), plt.title("binary image"), plt.imshow(im, "gray")
            plt.xticks([]), plt.yticks([])
            plt.subplot(1, 2, 2), plt.title("filled holes image"), plt.imshow(filled, "gray")
            plt.xticks([]), plt.yticks([])
            plt.show()
        return filled

    def remove_unwanted_object(self, im, show):
        """
        remove small object from image due to iron powder or segmented noise
        we can use opening, and if iron power persist we can then do blob anlysis and classify them as not relevant
        :param im: binarized image, range 0-1, 0 background, 1 foreground, holes of image are filled
        :param show: if true show step's result
        :return: cleaned image from iron power and noise
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned_im = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel, iterations=2)
        if show:
            print("-"*30)
            print("removing noise and iron power")
            plt.figure(4)
            plt.subplot(1, 2, 1), plt.title("binary image"), plt.imshow(im, "gray")
            plt.xticks([]), plt.yticks([])
            plt.subplot(1, 2, 2), plt.title("cleaned image"), plt.imshow(cleaned_im, "gray")
            plt.xticks([]), plt.yticks([])
            plt.show()
        return cleaned_im

    def remove_attached_object(self, cleaned_im, or_im, show):
        """
        find pairs of possible points connection, and then cut in the most promising points
        :param cleaned_im: cleaned binary image , 0 foreground, 1 background with filled holes
        :param or_im: binary imagewith no iron powder or noise, holes are not filled
        :param show: if true show result of the function
        :return: binary cleaned image with no attached object and holes not filled
        """

        removed = or_im.copy()

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        blackhat = cv2.morphologyEx(cleaned_im, cv2.MORPH_BLACKHAT, kernel)
        i_dil = cv2.dilate(blackhat, kernel, iterations=2)


        # find pairs of points
        n_labels, labels = cv2.connectedComponents(i_dil, 8)
        centroids =[]
        for blob in self.analyzer.compute_masks(i_dil,labels,n_labels):
            centroids.append(self.analyzer.baricenter(blob,self.analyzer.area(blob)))
        if show:
            print("-"*30)
            print("removing connection between different object")
            plt.figure(5), plt.title("possible connection points")
            plt.imshow(cleaned_im)
            plt.xticks([]), plt.yticks([])
            for c in centroids:
                print("possible points of connection: ", c)
                plt.plot(c[1], c[0], 'o')
            plt.show()

        # find possible connection between point, take the pairs of point that have less distance one to another
        # to avoid false connection we could put a threshold on distance
        found = []
        matched = []
        for i in range(0, len(centroids) - 1):
            q = PriorityQueue()
            pairs = []
            for j in range(i + 1, len(centroids)):
                if centroids[i] not in found and centroids[j] not in found:
                    pairs.append((centroids[i], centroids[j]))
            for pair in pairs:
                a = pair[0]
                b = pair[1]
                d = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
                q.put((d, pair))
            if q.qsize() > 0:
                match = q.get()
                if match[0] < 1000:
                    found.append(match[1][0])
                    found.append(match[1][1])
                    matched.append(match[1])

        if show:
            plt.figure(6)
            plt.imshow(removed)
            plt.xticks([]), plt.yticks([])
            plt.title("line cut")

        matching_image = np.bitwise_or(or_im, i_dil)
        # find all possible discrete point along the line that connect two point
        for match in matched:
            if show:
                print("point p1: ", match[0], "matched with point p2: ", match[1])
            x0, y0 = match[0]
            x1, y1 = match[1]
            x0, y0 = int(x0), int(y0)
            x1, y1 = int(x1), int(y1)

            points = self.find_discrete_line_points2(matching_image, (x0, y0), (x1, y1), show)
            for point in points:
                removed[point[0], point[1]] = 0
                if show:
                    plt.plot(point[1], point[0], 'o')


        if show:
            plt.show()
            print("range of values of image with no connection: ", np.unique(removed))
            plt.figure(7)
            plt.title("result of removing connnection between object")
            plt.imshow(removed, 'gray')
            plt.xticks([]), plt.yticks([])
            plt.show()
        return removed



    def find_discrete_line_points2(self, matching_image, p1, p2, show):
        if show:
            print("finding discrete line point between ", p1, p2)
        mp = MatchingProblem(p1,p2,matching_image)
        sol = astar_search(mp)
        path = self.path_states(sol)
        return path

    def path_states(self, node):
        if node == None:
            return []
        return self.path_states(node.parent) + [node.state]

    def connected_component(self, im, show):
        """
        find connected component in binary image
        :param im: binary image, range 0-1
        :param show: if true show result of computations
        :return: labeled image
        """
        n_labels, labels = cv2.connectedComponents(im, 8, cv2.CV_32S)
        if show:
            print("-"*30)
            print("finding connected components")
            print("labels found: ", n_labels)
            plt.figure(8),
            plt.title("connected components")
            plt.imshow(labels)
            plt.xticks([]), plt.yticks([])
            plt.show()
            print("-" * 30)
            print("END SEGMENTATION")
        return labels, n_labels

    def run(self, image, show=False):
        """
        segmetn image, remove iro power, delete connecttion between object, connected componens
        :param image: image to process
        :param show: if true show the computation process
        :return: binary image, labeled image
        """
        binary = self.segmentation(image, show)
        filled = self.fill_hole(binary, show)
        cleaned = self.remove_unwanted_object(filled, show)
        original_image_nopower = np.bitwise_and(binary, cleaned)
        segmented_image = self.remove_attached_object(cleaned, original_image_nopower, show)
        labels, n_labels = self.connected_component(segmented_image, show)
        return segmented_image, labels, n_labels
