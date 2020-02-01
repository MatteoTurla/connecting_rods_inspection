import numpy as np
import cv2
from matplotlib import pyplot as plt
import math

class BlobAnalyzer():

    def __init__(self):
        """
        perform measuraments on masked binary image
        """
        pass

    def compute_masks(self, im, labels, n_labels):
        """
        extract blobs from image
        :param im: binary image, 0 backgorund, 1 blob
        :param labels: labels image of connected component
        :param n_labels: number of labels
        :return: array of blob image 0 background, 1 foregorund
        """
        im = im.copy()
        labels = labels.copy()
        results = []
        for i in range(1, n_labels):
            mask = np.zeros(im.shape, np.uint8)
            mask[np.where(labels == i)] = 1
            blob = np.bitwise_and(im, mask)
            results.append(blob)
        return results

    def area(self, image):
        """
        compute area of the blob
        :param image: binary image, 0 background, 1 foregorund
        :return: area
        """
        return np.sum(image)

    def baricenter(self, image, area):
        """
        compute baricenter of the blob
        :param image: binary image, 0 background, 1 foregorund
        :param area: area of the blob
        :return: point (i,j) that rapresent the baricenter
        """
        i, j = np.where(image==1)
        bi = np.sum(i)/area
        bj = np.sum(j)/area
        return bi, bj

    def invariant_moments(self, image, baricenter, m, n):
        """
        compute central moments of order m,n
        :param image: binary image, 0 background, 1 foregorund
        :param baricenter: baricenter of the image
        :param m: order m of moments
        :param n: order n of moments
        :return: central moment of order m,n
        """
        index_i, index_j = np.where(image==1)
        i_b, j_b = baricenter
        diff_i = index_i - i_b
        diff_j = index_j - j_b
        power_i = np.power(diff_i, m)
        power_j = np.power(diff_j, n)
        mult = np.multiply(power_i, power_j)
        moment = np.sum(mult)

        return moment

    def orientation(self, M11, M20, M02):
        """
        compute orientation of the blob
        :param M11: central moment order 1,1
        :param M20: central moment order 2,0
        :param M02: central moment order 0,2
        :return: theta 0-pi in deg
        """
        if M02-M20 == 0:
            arctan = math.pi/2
        else:
            M = 2 * M11 / (M02 - M20)
            arctan = math.atan(M)
        theta = -1*0.5*arctan
        # check second derivatives
        second_der = math.cos(2*theta)*2*(M02-M20)-2*math.sin(2*theta)*2*M11
        if second_der < 0:
            theta = theta+math.pi/2
        if theta < 0:
            theta = theta + np.pi
        return theta

    def rad_conversion(self, rad):
        """
        convert deg to radius
        :param rad: orientation
        :return: orientation converted in deg
        """
        return rad/math.pi*180

    def axis_parameter(self, theta):
        """
        compute direction vector of major axis
        :param theta: orientation of the image
        :return:  alpha, beta
        """
        alpha = -1*math.sin(theta)
        beta = math.cos(theta)
        return alpha, beta

    def major_axis(self, alpha, beta, baricenter):
        """
        compute parameter of the major axis aj+bi+c
        :param alpha: first component of major axis direction
        :param beta: second component of major axis direction
        :param baricenter: baricenter of the image (i,j)
        :return: parameter of the major axis: a,b,c
        """
        bi, bj = baricenter
        a = alpha
        b = -1*beta
        c = beta*bi-1*alpha*bj
        return a,b,c

    def minor_axis(self, alpha, beta, baricenter):
        """
        compute parameter of the minor axis a'j+b'i+c'
        :param alpha: first component of minor axis direction
        :param beta: second component of minor axis direction
        :param baricenter: baricenter of the image (i,j)
        :return: parameter of the minor axis: a',b',c'
        """
        bi, bj = baricenter
        a = beta
        b = alpha
        c = -1*beta*bj-1*alpha*bi
        return a,b,c

    def line_j(self,a,b,c, i):
        """
        find j variable of the parametric line aj+bi+c = 0 given i
        :param a: first parameter of line
        :param b: second parameter of the line
        :param c: third parameter of the line
        :param i: i coordinates
        :return: j coordinate given i
        """
        return (-1*i*b-c)/a

    def line_i(self,a,b,c, j):
        """
        find i variable of the parametric line aj+bi+c = 0 given j
        :param a: first parameter of line
        :param b: second parameter of the line
        :param c: third parameter of the line
        :param i: j coordinates
        :return: i coordinate given j
        """
        return (-1*j*a-c)/b

    def find_contour(self, image):
        """
        find external contour using dilation
        :param image: binary blob, 0 background, 1 foreground
        :return: image of the contour, 0 background, 1 contour
        """

        # C4 contour
        kernel = np.ones((3,3), np.uint8)
        kernel[0,0], kernel[0,2], kernel[2,0], kernel[2,2] = 0,0,0,0
        im = image.copy()
        dilatate = cv2.dilate(im, kernel)
        return dilatate - im

    def minimum_rectangle(self, contour, a,b,c,a_,b_,c_):
        """
        find minimum oriented enclosing rectangle of the blob
        given major axis: aj+bi+c
        given minor axis: a'+b'+c'
        :param contour: contour of the blob
        :param a: first parameter of major axis
        :param b: second parameter of major axis
        :param c: third componet of major axis
        :param a_: first component of minor axis
        :param b_: second componet of minor axis
        :param c_: third component of minor axis
        :return: vertecies of oriented bounding box
        """
        nr, nc = contour.shape
        dMAmin = 10000
        dMAmax = -10000
        dMImin = 10000
        dMImax = -10000
        normMA = math.sqrt(a*a+b*b)
        normMI = math.sqrt(a_*a_ + b_*b_)
        # c1 = (i1,j1)
        # c2 = (i2,j2)
        # c3 = (i3,j3)
        # c4 = (i4,j4
        i1,j1,i2,j2,i3,j3,i4,j4 = 0,0,0,0,0,0,0,0
        for i in range(0,nr):
            for j in range(0,nc):
                if contour[i,j] == 1:
                    dMA = (a*j+b*i+c)/normMA
                    dMI = (a_*j+b_*i+c_)/normMI
                    if dMA < dMAmin:
                        dMAmin = dMA
                        i1 = i
                        j1 = j
                    if dMA > dMAmax:
                        dMAmax = dMA
                        i2 = i
                        j2 = j
                    if dMI < dMImin:
                        dMImin = dMI
                        i3 = i
                        j3 = j
                    if dMI > dMImax:
                        dMImax = dMI
                        i4 = i
                        j4 = j

        cl1 = -1*(a*j1+b*i1)
        cl2 = -1*(a*j2+b*i2)
        cw1 = -1*(a_*j3+b_*i3)
        cw2 = -1*(a_*j4+b_*i4)

        vj1 = (b * cw1 - b_ * cl1)/(a*b_-b*a_)
        vi1 = (a_ * cl1 - a * cw1)/(a*b_-b*a_)

        vj2 = (b * cw2 - b_ * cl1) / (a * b_ - b * a_)
        vi2 = (a_ * cl1 - a * cw2) / (a * b_ - b * a_)

        vj3 = (b * cw1 - b_ * cl2) / (a * b_ - b * a_)
        vi3 = (a_ * cl2 - a * cw1) / (a * b_ - b * a_)

        vj4 = (b * cw2 - b_ * cl2) / (a * b_ - b * a_)
        vi4 = (a_ * cl2 - a * cw2) / (a * b_ - b * a_)

        return (vi1,vj1), (vi2,vj2), (vi3,vj3), (vi4,vj4)

    def width_at_baricenter(self, contour, a, b, c, a_, b_, c_):
        """
        find the width at baricenter:
        find two points in opposite side (using signed distance with respect to major axis) and minimum distance with
        respect to the minor axis
        :param contour: image of contour, 0 background, 1 foreground
        :param a: first parameter of major axis
        :param b: second parameter of major axis
        :param c: third parameter of major axis
        :param a_: first parameter of minor axis
        :param b_: second parameter of minor axis
        :param c_: third parameter of minor axis
        :return: extreme points of the segment that rapresent the width at the baricenter
        """
        nr, nc = contour.shape
        normMA = math.sqrt(a * a + b * b)
        normMI = math.sqrt(a_ * a_ + b_ * b_)
        dWB1 = 10000
        dWB2 = 10000

        # cWB1 = (i5,j5)
        # cWB2 = (i6, j6)

        i5, j5, i6, j6 = 0, 0, 0, 0

        for i in range(0, nr):
            for j in range(0, nc):
                if contour[i, j] == 1:
                    dMA = (a * j + b * i + c) / normMA
                    dMI = (a_ * j + b_ * i + c_) / normMI
                    if dMA > 0 and abs(dMI) < dWB1:
                        dWB1 = abs(dMI)
                        i5 = i
                        j5 = j
                    if dMA < 0 and abs(dMI) < dWB2:
                        dWB2 = abs(dMI)
                        i6 = i
                        j6 = j

        return (i5,j5), (i6,j6)


    def find_holes_and_measure(self, image):
        """
        find holes in the blob and analyze them:
        in order to find the holes we find connected components on the inverted image blob,
        then assign to 0 the point that belong to the original background
        :param image: blob, 0 background, 1 foreground
        :return: return stats about the founded holes
        """
        M, N = image.shape
        for i in range(M):
            for j in range(N):
                if image[i, j] == 0:
                    bg_pos = (i, j)
                    break

        background = 1 - image
        n_labels, labels = cv2.connectedComponents(background, 8)
        holes = np.zeros(labels.shape, np.uint8)

        background_label = labels[bg_pos]
        # set to 1 the pixel belonging to a hole
        for i in range(1, n_labels):
            if not i == background_label:
                holes[np.where(labels == i)] = 1


        n_holes, holes_labels = cv2.connectedComponents(holes, 8)
        holes_analytics = []
        if n_holes > 1:
            holes_analytics = self.hole_analysis(holes, holes_labels, n_holes)
        return holes_analytics

    def find_segments(self, blob, measure):
        """
        find lines that connect the vertex in order to draw the bounding box,
        find the intersection between the major axis and the parallel line to the minor axis through v2
        find the intersection between the minor axis and the parallel line to the majox axis through v2
        this last 2 step done for draw the major and minor axis
        :param blob: binary image, 0 background, 1 foreground
        :param measure: dictionary that contains vertecies, baricenter, major and minor axes
        :return: a list of segment between two points
        """
        # segment between vertex
        baricenter = measure["baricenter"]
        bi,bj = baricenter
        v1,v2,v3,v4 = measure["vertecies"]
        segments = []
        segments.append(np.vstack((v1,v2)))
        segments.append(np.vstack((v2,v4)))
        segments.append(np.vstack((v4, v3)))
        segments.append(np.vstack((v3, v1)))

        a,b,c = measure["major_axis"]
        a_,b_,c_ = measure["minor_axis"]
        # find segment major axis and v2 = (i,j)
        # the parametric line parallel to minor axis is a'*j + b'*i + cv2 -> cv2 = -a'*j + b'*i
        # then find intersection between two lines using classical formulation

        cv1 = -1*a_*v2[1]-1*b_*v2[0]
        i = (c*a_-1*a*cv1)/(a*b_-1*b*a_)
        j = (-b*i-c)/a
        major_axis_v2 = [i,j]
        segment_MAaxis = np.vstack((baricenter, major_axis_v2))
        segments.append(segment_MAaxis)

        cv2 = -1 * a * v2[1] - 1 * b * v2[0]
        i = (cv2 * a_ - 1 * a * c_) / (a * b_ - 1 * b * a_)
        j = (-b * i - cv2) / a
        minor_axis_v2 = [i, j]
        segment_MIaxis = np.vstack((baricenter, minor_axis_v2))
        segments.append(segment_MIaxis)

        return segments

    def euclidean_distance(self, p1, p2):
        """
        euclidean distance between two points
        :param p1: (i,j)
        :param p2: (i',j')
        :return: euclidean distance between p1 and p2
        """
        x1, y1 = p1
        x2, y2 = p2
        return math.sqrt((x1-x2)**2+(y1-y2)**2)

    def hole_analysis(self, holes, labels, n_labels):
        blobs = self.compute_masks(holes, labels, n_labels)
        stats = []
        for blob in blobs:
            measure = {}
            area = self.area(blob)
            measure["area"] = area
            bi, bj = self.baricenter(blob, area)
            baricenter = (bi, bj)
            measure["baricenter"] = baricenter
            cont = self.find_contour(blob)
            x,y = np.where(cont==1)
            diameter = 0
            for i in range(len(x)):
                diameter += self.euclidean_distance((x[i],y[i]), baricenter)
            diameter = 2 * diameter / len(x)
            measure["diameter"] = diameter
            segments = []
            v1 = (np.min(x), np.min(y))
            v2 = (np.min(x), np.max(y))
            v3 = (np.max(x), np.min(y))
            v4 = (np.max(x), np.max(y))
            segments.append(np.vstack((v1, v2)))
            segments.append(np.vstack((v2, v4)))
            segments.append(np.vstack((v4, v3)))
            segments.append(np.vstack((v3, v1)))
            measure["segments"] = segments
            stats.append(measure)
        return stats



    def run(self, im, labels, n_labels, show=False):
        """
        perfrom analysis given an image
        :param im: binary image, 0 background, 1 foregorund
        :param labels: image of connected components
        :param n_labels: number of unique lables
        :param hole_analysis: if true perform holes analysis on each blob
        :param show: if true, plot the result of the computation
        :return: list of dictionaries, one for each components
        """
        blobs = self.compute_masks(im, labels, n_labels)
        if show:
            print("-" * 30)
            print("ANALYZING BLOBS")
            print("number of blobs to analize: ", len(blobs))
        blobs_anlytics = []
        for index, blob in enumerate(blobs):
            if show:
                print("-" * 20)
                print("analyzing blob number ", index, " :")
            measure = {}
            measure["index"] = index
            area = self.area(blob)
            measure["area"] = area
            bi,bj = self.baricenter(blob, area)
            baricenter = (bi,bj)
            measure["baricenter"] = baricenter
            M11 = self.invariant_moments(blob, baricenter, 1, 1)
            M20 = self.invariant_moments(blob, baricenter, 2, 0)
            M02 = self.invariant_moments(blob, baricenter, 0, 2)
            theta = self.orientation(M11, M20, M02)
            measure["orientation"] = self.rad_conversion(theta)
            alpha, beta = self.axis_parameter(theta)
            a,b,c = self.major_axis(alpha, beta, baricenter)
            measure["major_axis"] = [a,b,c]
            a_,b_,c_ = self.minor_axis(alpha, beta, baricenter)
            measure["minor_axis"] = [a_,b_,c_]
            contour = self.find_contour(blob)
            v1,v2,v3,v4 = self.minimum_rectangle(contour,a,b,c,a_,b_,b_)
            measure["vertecies"] = [v1,v2,v3,v4]
            height = self.euclidean_distance(v1, v2)
            measure["height"] = [height, v1, v2]
            width = self.euclidean_distance(v1, v3)
            measure["width"] = [width, v1, v3]
            compactness = width / height
            measure["compactness"] = compactness
            if compactness <= 0.6:
                # hole anlysis
                holes_stats = self.find_holes_and_measure(blob)
                n_holes = len(holes_stats)
                if n_holes > 0:
                    measure["n_holes"] = n_holes
                    measure["holes"] = holes_stats
                    segments = self.find_segments(blob, measure)
                    measure["segments"] = segments
                    wb1,wb2 = self.width_at_baricenter(contour,a,b,c,a_,b_,c_)
                    wb_segment = np.vstack((wb1,wb2))
                    measure["distance_baricenter"] = [self.euclidean_distance(wb1,wb2), wb1,wb2]
                    if n_holes == 1:
                        measure["type"] = "A"
                    elif n_holes == 2:
                        measure["type"] = "B"
                    else:
                        raise Exception('Problem in analysing blob:', index)
                    blobs_anlytics.append(measure)

                    # plot result
                    if show:
                        plt.figure()
                        plt.imshow(blob)
                        plt.plot(bj, bi, 'o')
                        for segment in segments:
                            plt.plot(segment[:, 1], segment[:, 0], 'b-')

                        plt.plot(wb_segment[:, 1], wb_segment[:, 0], 'go-')

                        for hole in measure["holes"]:
                            for segment in hole["segments"]:
                                plt.plot(segment[:, 1], segment[:, 0], 'r--')
                            baricenter = hole["baricenter"]
                            plt.plot(baricenter[1], baricenter[0], 'ro')
                        plt.xticks([]), plt.yticks([])
                        plt.show()

        return blobs_anlytics

    def show(self, image, stats):
        f1 = plt.figure(1)
        plt.imshow(image, "gray")
        for stat in stats:
            print("-"*30)
            print("index: ", stat["index"])
            print("type: ", stat["type"])
            print("position: ", stat["baricenter"])
            print("orientation: ", stat["orientation"])
            print("length: ", stat["height"][0])
            print("width: ", stat["width"][0])
            print("width at baricenter: ", stat["distance_baricenter"][0])

            baricenter = stat["baricenter"]
            bi,bj=baricenter
            segments = stat["segments"]
            for segment in segments:
                plt.plot(segment[:, 1], segment[:, 0], 'bo-.')

            label = "index : "+str(stat["index"])+" type: "+stat["type"]
            plt.plot(bj, bi, 'o', label=label)
            wb = stat["distance_baricenter"]
            plt.plot(wb[1][1], wb[1][0], 'mo', markersize="3")
            plt.plot(wb[2][1], wb[2][0], 'mo', markersize="3")

            print("holes:")
            for hole in stat["holes"]:
                print("\tbaricenter: ", hole["baricenter"])
                print("\tdiameter: ", hole["diameter"])
                for index, segment in enumerate(hole["segments"]):
                    if index < 4:
                        plt.plot(segment[:, 1], segment[:, 0], 'r--')
                bi,bj = hole["baricenter"]
                plt.plot(bj, bi, 'ro', markersize=5)

            plt.xticks([]), plt.yticks([])
            plt.legend()
        plt.show()





