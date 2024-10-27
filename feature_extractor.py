import tqdm

class FeatureExtractor:

    def __init__(self, src, dest):
        self.src = src
        self.dest = dest

    def extract(self):
        # for all images in src list
        for img, dst in zip(tqdm(self.src), self.dest):
            # load image
            image = 