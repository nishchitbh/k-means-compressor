import numpy as np
import cv2
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, array, n_clusters, n_iters):
        """
        Initializes the KMeans object with input parameters.

        Parameters:
            array (numpy.ndarray): Input image array.
            n_clusters (int): Number of clusters (i.e., colors) to segment the image into.
            n_iters (int): Number of iterations for the KMeans algorithm.
        """
        self.array = array
        self.n_clusters = n_clusters
        self.n_iters = n_iters

    def preprocessing(self):
        """
        Preprocesses the input image array.

        Converts the image from BGR format (default format of cv2) to RGB format,
        and flattens the image to a pixel grid while normalizing pixel values.

        Returns:
            numpy.ndarray: Preprocessed and flattened image array.
        """
        array = self.array
        converted = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        flat = converted.reshape(-1, 3)  # Flatten the image to a pixel grid
        return flat / 255  # Normalize pixel values by dividing by 255.

    def main(self):
        """
        Main method implementing the KMeans algorithm.

        Returns:
            numpy.ndarray: Final cluster centroids.
        """
        array = self.preprocessing()  # Preprocess the input image
        n_clusters = self.n_clusters

        # Initialize cluster centroids randomly from the pixel array
        initial_Ks_index = np.random.randint(0, len(array), n_clusters)
        Ks = array[initial_Ks_index]

        # Main loop for KMeans algorithm
        for _ in range(self.n_iters):
            # Calculate distances of each pixel from each cluster centroid
            distances = np.linalg.norm(array[:, np.newaxis] - Ks, axis=2)

            # Assign each pixel to the closest cluster centroid
            assignments = np.argmin(distances, axis=1)

            # Update cluster centroids
            for i in range(n_clusters):
                cluster_points = array[assignments == i]
                if len(cluster_points) > 0:
                    Ks[i] = np.mean(cluster_points, axis=0)

        return Ks

    def compress(self):
        """
        Compresses the image by assigning each pixel to its nearest cluster centroid.

        Returns:
            numpy.ndarray: Compressed image array.
        """
        Ks = self.main()  # Get final cluster centroids
        array = self.preprocessing()  # Preprocess the input image
        distances = np.linalg.norm(array[:, np.newaxis] - Ks, axis=2)
        assignments = np.argmin(distances, axis=1)
        processed_pixels = Ks[assignments]
        compressed_image = processed_pixels.reshape(self.array.shape)
        return compressed_image

    def save_image(self, filename, compressed_image):
        """
        Saves the compressed image to a file.

        Parameters:
            filename (str): Name of the file to save.
            compressed_image (numpy.ndarray): Compressed image array.
        """
        # Convert compressed image back to uint8 format and BGR color space
        compressed_image = (compressed_image * 255).astype(np.uint8)
        bgr_image = cv2.cvtColor(compressed_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename, bgr_image)


# Load input image
img = cv2.imread("image.jpg")

# Initialize KMeans object and perform compression
kmeans = KMeans(img, 16, 100)
compressed_image = kmeans.compress()

# Display compressed image
plt.imshow(compressed_image)
plt.show()

# Save compressed image
kmeans.save_image("output.jpg", compressed_image)
