from nn_from_scratch.dataset import Dataset
from nn_from_scratch.network import Network

dataset = Dataset(split='train')
X, Y = dataset.get_features_and_labels()

nn = Network()
nn.train(X, Y, iterations=1000, training_rate=0.1)
