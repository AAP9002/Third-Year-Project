import numpy as np

class ImageStacker:
    def get_image_grid(image_array, row_length = 3):
        x_count = len(image_array)
        image_rows = []
        white_image = np.ones(image_array[0].shape, dtype=image_array[0].dtype) * 255
        
        for i in range(0, x_count, row_length):
            row_images = image_array[i:i + row_length]
            if len(row_images) < row_length:
                row_images.extend([white_image] * (row_length - len(row_images)))

            row_stack = np.hstack(row_images)
            image_rows.append(row_stack)
        
        output_image = np.vstack(image_rows)
        return output_image