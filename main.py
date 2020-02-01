from Dataset import Dataset
from Segmenter import Segmenter
from BlobAnalyzer import BlobAnalyzer

# reading dataset
dataset = Dataset("./immagini/*.BMP")

segmenter = Segmenter()
analyzer = BlobAnalyzer()


# set true: show the computation of all the process
show=False

for image in dataset.images:
    binary, labels, n_labels = segmenter.run(image, show=show)
    stats = analyzer.run(binary, labels, n_labels, show=show)
    analyzer.show(image, stats)


