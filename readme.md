# Image Compression using K-Means Clustering

This Python script implements image compression using the K-Means clustering algorithm. It takes an input image and reduces the number of unique colors (clusters) in the image, thus compressing the image while preserving its visual content.

## Dependencies
- NumPy
- OpenCV (cv2)
- Matplotlib


## How it Works

The image compression process involves the following steps:

1. **Preprocessing**: The input image is preprocessed by converting it from the BGR color space (default format of OpenCV) to the RGB color space. The image is then flattened to a pixel grid, and pixel values are normalized.

2. **K-Means Clustering**: The K-Means algorithm is applied to the preprocessed image. Random initial cluster centroids are selected from the pixel array. The algorithm iteratively assigns each pixel to the nearest cluster centroid and updates the centroids based on the mean of the pixels assigned to each cluster.

3. **Compression**: Each pixel in the original image is replaced with the color of the nearest cluster centroid, resulting in a compressed image.

4. **Saving the Compressed Image**: The compressed image is saved to a file.

## Usage

1. **Requirements**: Make sure you have Python installed, along with the necessary libraries: NumPy, OpenCV (cv2), and Matplotlib.

2. **Input Image**: Provide the input image (`image.jpg`) in the same directory as the script.

3. **Running the Script**: Execute the Python script `image_compression.py`. You can adjust the number of clusters and iterations in the `KMeans` object initialization to control the compression quality.

4. **Output**: The compressed image will be displayed and saved as `output.jpg` in the same directory.

## Results
Results of 16 clusters over 100 iterations are shown below
### Original image
![input](image.jpg)
### Compresed image
![output](output.jpg)
