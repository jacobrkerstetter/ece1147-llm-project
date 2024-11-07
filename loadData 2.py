import os
from PIL import Image

class ImageData:
    def __init__(self):
        self.abortionImages = self.loadImages('data/images/abortion')
        self.gunControlImages = self.loadImages('data/images/gun_control')

    def loadImages(self, path):
        images = []
        for filename in os.listdir(path):
            filepath = os.path.join(path, filename)
            try:
                img = Image.open(filepath)
                images.append(img)
            except Exception as e:
                print(f"Error loading image {filepath}: {e}")
        return images