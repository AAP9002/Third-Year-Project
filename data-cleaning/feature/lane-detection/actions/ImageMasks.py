import cv2


class ImageMasks:
    def maskImageAboveY(image:cv2.typing.MatLike, y_index:int):
        """mask all pixels above a given y value

        Args:
            image (MatLike): image passed by reference
            y_index (Integer): y value to mask at 
        """
        if(y_index > image.shape[0]):
            raise Exception(f'maskImageAboveY y_index out of rank for mask (y_index: {y_index}, image height: {image.shape[0]})')
        image[0:y_index, :] = 0