import numpy as np
import json
from nn_from_scratch.network import Network
from nn_from_scratch.repository import Repository

def test_store_and_load_weights(tmp_path):
    # Prepare
    network = Network(
        input_size=10,
        hidden_size=10,
        output_size=10,
    )

    repository = Repository()

    with open('resources/weights_and_biases.json', 'r') as f:
        weights_and_biases = json.load(f)

    network.hidden_layer.W = np.array(weights_and_biases['hidden_layer']['weights'])
    network.hidden_layer.b = np.array(weights_and_biases['hidden_layer']['biases']).reshape(-1, 1)
    network.output_layer.W = np.array(weights_and_biases['output_layer']['weights'])
    network.output_layer.b = np.array(weights_and_biases['output_layer']['biases']).reshape(-1, 1)

    stored_weights_file = tmp_path / "weights.json"

    # Act
    repository.store(network, str(stored_weights_file))

    new_network = Network(
        input_size=10,
        hidden_size=10,
        output_size=10
    )

    repository.load(new_network, str(stored_weights_file))

    # Assert
    np.testing.assert_array_almost_equal(
        network.hidden_layer.W, 
        new_network.hidden_layer.W, 
        decimal=7,
        err_msg="Hidden layer weights do not match after store and load"
    )

    # Verify hidden layer biases
    np.testing.assert_array_almost_equal(
        network.hidden_layer.b, 
        new_network.hidden_layer.b, 
        decimal=7,
        err_msg="Hidden layer biases do not match after store and load"
    )

    # Verify output layer weights
    np.testing.assert_array_almost_equal(
        network.output_layer.W, 
        new_network.output_layer.W, 
        decimal=7,
        err_msg="Output layer weights do not match after store and load"
    )

    # Verify output layer biases
    np.testing.assert_array_almost_equal(
        network.output_layer.b, 
        new_network.output_layer.b, 
        decimal=7,
        err_msg="Output layer biases do not match after store and load"
    )
