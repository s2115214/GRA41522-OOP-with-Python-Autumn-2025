import os
import numpy as np
import wget
import tensorflow as tf


class DataLoader:
    def __init__(self, dset="mnist_color", test = False):
        """Initialize DataLoader with specified dataset.

        Args:
            dset (str): Dataset name. 'mnist_color' or 'mnist_bw'.
            test (bool): If True, load test dataset. Otherwise, load training dataset.
        """
        self._dset = dset
        self._test = test
        # If loading test data
        if self._test:
            # Defines the filename. E.g. mnist_color.npy
            # Used to find the file in local folder or give name to file being downloaded
            self._filename = f"{self._dset}.npy"
            # Links to test datasets
            self._links = {"mnist_bw": 'https://www.dropbox.com/scl/fi/dj8vbkfpf5ey523z6ro43/mnist_bw_te.npy?rlkey=5msedqw3dhv0s8za976qlaoir&st=nmu00cvk&dl=1',
                        "mnist_color": 'https://www.dropbox.com/scl/fi/w08xctj7iou6lqvdkdtzh/mnist_color_te.pkl?rlkey=xntuty30shu76kazwhb440abj&st=u0hd2nym&dl=1'
                        }
  
        else:
            self._filename = f"{self._dset}_train.npy"
            # Links to training datasets
            self._links = {"mnist_bw": "https://www.dropbox.com/scl/fi/fjye8km5530t9981ulrll/mnist_bw.npy?rlkey=ou7nt8t88wx1z38nodjjx6lch&st=5swdpnbr&dl=1",
                        "mnist_color": 'https://www.dropbox.com/scl/fi/w7hjg8ucehnjfv1re5wzm/mnist_color.pkl?rlkey=ya9cpgr2chxt017c4lg52yqs9&st=ev984mfc&dl=1'
                        }
        
        # Check if file exists locally, if not download
        self._check_if_file_exists()

        
            
    def load_data(self, color = "m1"):
        #Load dataset from local file
        self._data = np.load(f"./data/{self._filename}", allow_pickle=True)
        
        # For mnist_color, select the appropriate color 
        if self._dset == "mnist_color":
            self._data = self._data[color]
        
        # Reshape for mnist_bw to add channel dimension. Now of shape (num_samples, 28, 28, 1) to work with convolutional layers
        if self._dset == "mnist_bw":
            self._data = self._data.reshape((-1, 28, 28, 1))
        
        # Normalize and float32 conversion. Returns a tf.data.Dataset
        self._data = self._process_data(self._data)
        return self._data
    
    
    
    def _check_if_file_exists(self):
         # Check if file exists locally. If not, download it.
        if not os.path.exists(f"./data/{self._filename}"):
            print(f"Downloading {self._dset} dataset...")
            # Finds the correct link based on dataset name. Whether the links are for test or train data is handled in the __init__
            url = self._links[self._dset]
            wget.download(url, out=f"./data/{self._filename}")
            
            print("Download complete.")
        
    
    
    def _process_data(self, data):
        # Processes the data: convert to float32 and normalize pixel values to [0, 1](if bw)
        data = data.astype('float32')
        
        # Normalize pixel values to [0, 1]
        if self._dset == "mnist_bw":
            data = data / 255.0
        print(f"Data normalized: min={np.min(data)}, max={np.max(data)}")
        # Return processed data as a TensorFlow Dataset
        return tf.data.Dataset.from_tensor_slices(data)
    



        