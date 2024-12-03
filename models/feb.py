from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
import torch
import torch.nn as nn

class FMinMaxScaler:
    # The FMinMaxScaler class is a custom implementation of a scaling technique,
    # similar to scikit-learn's MinMaxScaler, with additional functionality to handle outlier robustness.
    def __init__(self, ratio=0.01):
        # The goal is to scale data to a range of
        # [0,1] while being robust to outliers by focusing on the central portion of the data and ignoring extreme values.
        # This is achieved using a trimming ratio, specified as ratio, to exclude a small percentage of extreme data points before calculating the minimum and maximum values for scaling.
        self.ratio = ratio
        self.min = None
        self.max = None
        # self.ratio: Specifies the proportion of extreme values to exclude from both ends of the data during fitting. For example, ratio=0.01 trims 1% of extreme data on each end.
        # self.min: The minimum value after trimming outliers, used for scaling.
        # self.max: The maximum value after trimming outliers, used for scaling.
        
    def fit(self, data):
        # The np.partition function is used to find the thresholds for the lowest and highest values, excluding the specified ratio of outliers.
        m0 = np.partition(data, int(data.shape[0] * self.ratio), axis=0)[int(data.shape[0] * self.ratio)-1]
        m1 = np.partition(data, -int(data.shape[0] * self.ratio), axis=0)[-int(data.shape[0] * self.ratio)]
        # For a ratio of 0.01, it identifies the values below which the lowest 1% of the data lies (m0) and above which the highest 1% of the data lies (m1).
        data = data[(data>=m0) & (data<=m1)]
        # only the data points with the range m0 and m1 kept for scaling
        self.min = data.min(0).item()
        self.max = data.max(0).item()
        
    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
    
    def transform(self, data):
        # Normalizes all data points to a uniform range, which is crucial for many machine learning models to perform effectively.
        if isinstance(data, np.ndarray):
            data = np.clip(data, self.min, self.max)
        elif isinstance(data, torch.Tensor):
            data = torch.clamp(data, self.min, self.max)
        data = (data - self.min) / (self.max - self.min)
        return data


class ForegroundEstimateBranch(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        # This PyTorch module creates a neural network layer
        # (Conv2d) and initializes its weights using pre-trained parameters (
        # derived from LDA and a Min-Max scaler). The layer learns to predict whether
        # each pixel in the input image belongs to the foreground
        self.conv1x1 = torch.nn.Conv2d(in_channels, 1, 1, 1).requires_grad_(False)
        # Initializes a 1x1 convolution layer (Conv2d) with one output channel.
        # .requires_grad_(False) freezes the layer parameters (useful for inference without training the layer).
    
    def initialize_weights(self, lda: LinearDiscriminantAnalysis, normalizer: FMinMaxScaler):
        # Converts LDA weights and bias to PyTorch tensors and normalizes them using the provided FMinMaxScaler.
        # These weights are assigned to the convolution layer.
        self.conv1x1.weight.data = torch.from_numpy(lda.coef_.T).float()[None, :, :, None] / torch.tensor(normalizer.max - normalizer.min).float()
        # This scales the weights to match the normalized range of the features.
        # The minimum value of the normalizer isn't subtracted because the weights themselves are directional multipliers (slopes), and subtracting the minimum would distort this directional property.
        self.conv1x1.bias.data = (torch.from_numpy(lda.intercept_).float() - torch.tensor(normalizer.min).float()) / torch.tensor(normalizer.max - normalizer.min).float()
        # Subtracting the minimum ensures the bias aligns with the normalized input space. This step adjusts the intercept so that it matches the scale of the normalized data (where features are normalized to lie between 0 and 1).
        return self
    
    @torch.no_grad()
    def forward(self, x):
        # Passes the input tensor x through the convolution layer and clamps its output between 0 and 1.
        return torch.clamp(self.conv1x1(x), 0, 1)
    
    
def get_feb(train_features) -> ForegroundEstimateBranch:
    """
    Get the foreground estimate branch.
    This code defines the function get_feb that aims to estimate foreground features in a batch
    of images using a combination of clustering (K-Means) and classification (Linear Discriminant Analysis).
    Args:
        train_features (torch.Tensor): The train features of shape (B, C, H, W), 
            where B is the batch size, C is the number of channels, H is the height, and W is the width.
    Returns:
        ForegroundEstimateBranch
    """
    kmeans_f_num      = 50000 # kmeans_f_num = 50000: This specifies how many sample points (pixels) from the image will be used to train the K-Means algorithm.
    lda_f_num         = 15000 # This defines how many foreground and background features will be used for training Linear Discriminant Analysis (LDA).
    '''
    oringal vlaues 
        foreground_ratio  = 1/5
    background_ratio  = 3/80
    '''
    foreground_ratio  = 0.1 #  Specifies that the foreground is expected to occupy 10% of the image.
    background_ratio  = 3/80 # pecifies that the background is expected to occupy 3/80ths of the image (about 3.75%).
    background_id_num = 1 # This indicates the number of background IDs to use when defining the background mask.
    n_clusters        = 2 # n_clusters = 2: This sets the number of clusters in the K-Means algorithm, which will divide the image pixels into two groups (likely foreground and background).
    random_state      = np.random.RandomState(66) # A random number generator for reproducibility.
    # these 2 functions are part of sci learn function
    lda               = LinearDiscriminantAnalysis()
    kmeans            = KMeans(n_clusters, n_init=10, random_state=random_state)
    normalizer        = FMinMaxScaler()
    # train_features: b x c x h x w
    # you have 5330 ,256,80,80
    B, C, H, W = train_features.shape
    image_features = train_features.permute(0, 2, 3, 1).cpu().numpy()  # b x h x w x 512
    # The function first gets the shape of the train_features tensor (B, C, H, W)
    # and reshapes it from (B, C, H, W) to (B, H, W, C) for easier manipulation.
        # It then converts this tensor into a NumPy array (image_features).
    # you permute everything
    # you reshape everything based on thec so you have huge vector by 512
    # you have a set of H*W*B = 3392000
    # you get from them kmeans_f_num
    # you extract solo kmeans_f_num
    # This code snippet applies K-Means clustering to the features extracted from an image and assigns each pixel in the image to a cluster.
    kmeans.fit(image_features.reshape(-1, C)[random_state.permutation(B*H*W)[:kmeans_f_num]])
    # The code selects kmeans_f_num random pixels from the reshaped image_features and fits a K-Means model to these pixels.
    # The K-Means model learns the cluster centroids based on the features in each pixel.
    # you classify this pixel in chanel 0 is cloer to which cluster centroid
    # the question in here when you use kmeans_f_num that means you are not taking along all the images but rather first
    # set of images
    # is this bad ?????

    image_codes = kmeans.predict(image_features.reshape(-1, C)).reshape(B, H, W)
    # After the model is trained, it predicts the cluster assignment for each pixel in the image.
    # The feature vectors for all pixels (image_features.reshape(-1, C)) are passed to the K-Means model, and it assigns each pixel to one of the kmeans_f_num clusters based on the learned centroids.
    # you have predicted imagecode of size of (530,80,80)
    # based on the 512 channel you used k mean to create some kind of clusters for each channels
    # the cluster is based on the batch and based on size
    # background
    # remember that H and W are 80 by 80
    background_mask = np.zeros((B, H, W), dtype=bool)
    # Includes a border region:
    # Top background_ratio * H rows.
    # Bottom background_ratio * H rows.
    # Left background_ratio * W columns.
    # Right background_ratio * W columns.
    background_mask[:, :int(background_ratio * H), :] = True
    background_mask[:, -int(background_ratio * H):, :] = True
    background_mask[:, int(background_ratio * H):-int(background_ratio * H), :int(background_ratio * W)] = True
    background_mask[:, int(background_ratio * H):-int(background_ratio * H), -int(background_ratio * W):] = True
    # foreground
    # Focuses on a central square region:
    # Width and height defined as foreground_ratio of the image dimensions.
    foreground_mask = np.zeros((B, H, W), dtype=bool)
    foreground_mask[:, int(image_codes.shape[1] / 2 - image_codes.shape[1] * foreground_ratio):int(image_codes.shape[1] / 2 + image_codes.shape[1] * foreground_ratio),
                    int(image_codes.shape[2] / 2 - image_codes.shape[2] * foreground_ratio):int(image_codes.shape[2] / 2 + image_codes.shape[2] * foreground_ratio)] = True
    # background id
    # This is often used for one-hot encoding, where a numerical value is represented as a vector with all zeros except for a 1 at the index of the numerical value.
    # image_codes is a 1D NumPy array with shape (489720,).
        # background_mask is a Boolean array (or index array) of the same length as image_codes, used to select specific elements.
    # # xtracts only the elements of image_codes where background_mask is True
    # very smart way to idenify things
    # Since the background_id_num is 1 in your example, it identifies the most frequent background cluster(s) in the background_mask as the background ID(s).
    # In a complex scene, clusters may separately identify different areas of the background (e.g., sky, water, walls).
    background_ids = np.eye(kmeans.n_clusters)[image_codes[background_mask]].sum(0).argsort()[kmeans.n_clusters-background_id_num:]
    # omputes the sum along the rows of the 2D array. you sum allong axi 0
    # so you jhave (489720,2)
    # you sum along so you get something else
    #np.eye(kmeans.n_clusters)[image_codes[background_mask]].sum(0) give you array([139557., 350163.])
    # argsort return the indices that sort an array
    #### Mask Refinement:
    # Determine which cluster IDs correspond to the background:
    # Count cluster frequencies in the background mask.
    # Take the most common cluster(s) as the background ID(s)
    # leave background id
    # background mask has shape of (530, 80, 80)
    # you and it with place where images_codes have also the same background id
    background_mask = background_mask & (np.stack([image_codes == background_id for background_id in background_ids]).sum(0) > 0)
    # you got the write background id and accordingly you got what you want !

    # remove background id
    foreground_mask = foreground_mask & (np.stack([image_codes != background_id for background_id in background_ids]).sum(0) >= len(background_ids))
    # Refine masks to match cluster assignments:
    # Background mask keeps only pixels belonging to the background cluster(s).
    # Foreground mask excludes pixels in the background cluster(s).
    # This step ensures:
    # Pixels labeled as foreground are not part of any background cluster.
    # Foreground is refined to exclude regions dominated by background.
    background_features = image_features[background_mask]
    foreground_features = image_features[foreground_mask]

    ### LDA Training
    # Purpose: Train a classifier to separate foreground and background features.
    lda_f_num = min(lda_f_num, len(background_features), len(foreground_features))  # accelerate
    # lda_f_num: Number of features to sample for LDA (e.g., 15,000).
    lda.fit(np.concatenate([
        background_features[random_state.permutation(len(background_features))[:lda_f_num]], 
        foreground_features[random_state.permutation(len(foreground_features))[:lda_f_num]]]), 
        np.concatenate([np.zeros((lda_f_num), dtype=int), np.ones((lda_f_num), dtype=int)]))
    # Train LDA with:
    # Background pixels labeled as 0.
    # Foreground pixels labeled as 1.
    # Sample features from the foreground and background masks.
    normalizer.fit(lda.decision_function(image_features.reshape(-1, C)))
    # C is the 530 that you already have
    return ForegroundEstimateBranch(C).initialize_weights(lda, normalizer)
