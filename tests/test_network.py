import numpy as np
from nn_from_scratch.network import Network

def test_network_initialization():
    network = Network(input_size=784, hidden_size=128, output_size=10)

    assert network.input_size == 784, "Network should set input size correctly"
    assert network.hidden_size == 128, "Network should set hidden layer size correctly"
    assert network.output_size == 10, "Network should set output size correctly"
    assert network.input_layer is not None, "Input layer should be initialized"
    assert network.hidden_layer is not None, "Hidden layer should be initialized"
    assert network.output_layer is not None, "Output layer should be initialized"

def test_network_forward_method():
    network = Network(input_size=3, hidden_size=4, output_size=2)
    
    X = np.random.rand(3, 5)
    Y, _ = network.forward(X)
    
    assert Y.shape == (2, 5), "Forward method should return correct output dimensions"
    assert (Y >= 0).all(), "Softmax output should be non-negative"
    assert np.allclose(np.sum(Y, axis=0, keepdims=True), 1), "Softmax output should sum to 1 for each column"

def test_network_forward_training_mode():
    network = Network(input_size=3, hidden_size=4, output_size=2)
    
    X = np.random.rand(3, 5)
    Y, _ = network.forward(X, training=True)
    
    assert Y.shape == (2, 5), "Forward method should return correct output dimensions in training mode"

def test_network_train_method():
    network = Network(input_size=3, hidden_size=10, output_size=10)

    X = np.random.rand(3, 10)
    Y = np.random.randint(0, 2, (10,))
    
    for i, accuracy in network.train(X, Y, iterations=5, learning_rate=0.01):
        assert i == 1, "The train method must return the iteration counter"
        assert isinstance(accuracy, float), "The train method must return the accuracy at each step"

        break

def test_network_backward_method():
    network = Network(input_size=3, hidden_size=10, output_size=10)
    
    X = np.random.rand(3, 10)
    Y = np.random.randint(0, 2, 10)
    Y_pred, steps = network.forward(X, training=True)
    
    initial_output_weights = network.output_layer.W.copy()
    initial_hidden_weights = network.hidden_layer.W.copy()
    
    network.backward(X, Y, Y_pred, steps, learning_rate=0.01)
    
    assert not np.array_equal(network.output_layer.W, initial_output_weights), "Backward method should update output layer weights"
    assert not np.array_equal(network.hidden_layer.W, initial_hidden_weights), "Backward method should update hidden layer weights"

def test_network_update_params():
    network = Network(input_size=3, hidden_size=4, output_size=2)
    
    dW_o = np.random.rand(2, 4)
    db_o = np.random.rand(2, 1)
    dW_h = np.random.rand(4, 3)
    db_h = np.random.rand(4, 1)
    
    initial_output_weights = network.output_layer.W.copy()
    initial_output_biases = network.output_layer.b.copy()
    initial_hidden_weights = network.hidden_layer.W.copy()
    initial_hidden_biases = network.hidden_layer.b.copy()
    
    network.update_params(dW_o, db_o, dW_h, db_h, learning_rate=0.01)
    
    assert not np.array_equal(network.output_layer.W, initial_output_weights), "Output layer weights should be updated"
    assert not np.array_equal(network.output_layer.b, initial_output_biases), "Output layer biases should be updated"
    assert not np.array_equal(network.hidden_layer.W, initial_hidden_weights), "Hidden layer weights should be updated"
    assert not np.array_equal(network.hidden_layer.b, initial_hidden_biases), "Hidden layer biases should be updated"
