# RecSys Approaches

In this code I explores various recommendation system (RecSys) approaches, including Singular Value Decomposition (SVD) decomposition, collaborative filtering, and deep learning techniques.

### SVD Decomposition

SVD decomposition is a matrix factorization technique that aims to represent a user-item interaction matrix as the product of three matrices: user matrix, singular value matrix, and item matrix. This approach captures latent features and patterns in the data, enabling us to make personalized recommendations. In my code it was implemented with numpy library.

### Collaborative Filtering

Collaborative filtering is a classic RecSys approach that recommends items to users based on the preferences and behaviors of similar users. User-based and item-based collaborative filtering will be implemented. This method leverages user-item interaction data to identify relationships and similarities between users and items.

### Deep Learning

Deep learning techniques, such as neural networks, can be applied to recommendation systems to capture complex patterns in user-item interactions. I experimented with neural network architectures like feedforward neural networks for recommendation tasks. PyTorch was used for deep learning implementations.

## Usage

To use and explore the different RecSys approaches implemented in this code, follow these steps:

1. Clone the repository: `git clone https://github.com/UniverseIsMyCreation/RecSys_approaches.git`
2. `dl_recsys.py` contains neural network implementation, `collaborative_filtering.py` contains collaborative filtering implementation, `svd_recsys.py` contains svd decomposition implementation, `tools.py` contains some tool function and necessary libraries
3. `RecSys.ipynb` contains step by step code of these approaches
## Dependencies

The code relies on the following libraries and packages:

- NumPy
- Scikit-learn
- PyTorch
- tqdm
- matplotlib
- (Add other dependencies as needed)
