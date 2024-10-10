class ImageMasks:
    def maskImageAboveY(image, y_index):
        image[0:y_index, :] = 0