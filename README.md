
# Simple Neural Network Implementation

## Overview

This repository contains a simple feedforward neural network implementation from scratch using Python and NumPy. The network has:
- 1 hidden layer with configurable size
- Sigmoid activation function
- Binary cross-entropy loss
- Gradient descent optimization

## Requirements

### Python Packages
```bash
pip install numpy matplotlib
```

## File Structure
```
.
├── neural_network.py    # Main neural network implementation
└── instructions.md      # This file
```

## How It Works

### Neural Network Architecture

The network has three layers:
1. **Input Layer**: Takes in features (2 for the example)
2. **Hidden Layer**: Configurable number of neurons (10 by default)
3. **Output Layer**: 1 neuron for binary classification

### Forward Propagation
1. Input → Hidden Layer (sigmoid activation)
2. Hidden → Output Layer (sigmoid activation)
3. Output is probability between 0 and 1

### Backward Propagation
- Computes gradients using chain rule
- Updates weights and biases using gradient descent

## Usage

### Quick Start

Run the example:
```bash
python neural_network.py
```

### Creating Your Own Model

```python
from neural_network import SimpleNeuralNetwork

# Create model
model = SimpleNeuralNetwork(
    input_size=2,      # Number of features
    hidden_size=10,    # Neurons in hidden layer
    output_size=1,     # Binary classification
    learning_rate=0.5  # Learning rate
)

# Train
model.train(X_train, y_train, epochs=1000)

# Predict
predictions = model.predict(X_test)

# Evaluate
accuracy = model.accuracy(X_test, y_test)
```

## Training on Your Own Data

### Data Preparation

Your data should be in the following format:
- **X**: NumPy array of shape (n_samples, n_features)
- **y**: NumPy array of shape (n_samples, 1) with values 0 or 1

Example:
```python
import numpy as np

# Generate sample data
X = np.random.randn(100, 2)  # 100 samples, 2 features
y = (X[:, 0] + X[:, 1] > 0).astype(int).reshape(-1, 1)  # Binary labels
```

### Training Custom Data

```python
# Split data
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Create and train model
model = SimpleNeuralNetwork(
    input_size=X_train.shape[1],  # Auto-detect features
    hidden_size=20,
    output_size=1,
    learning_rate=0.1
)

model.train(X_train, y_train, epochs=2000)
```

## Key Functions

### Class Methods

| Method | Description |
|--------|-------------|
| `forward(X)` | Perform forward propagation |
| `backward(X, y, cache)` | Compute gradients via backpropagation |
| `update_parameters(gradients)` | Apply gradient descent |
| `train(X, y, epochs)` | Train the network |
| `predict(X)` | Make predictions |
| `accuracy(X, y)` | Calculate accuracy |
| `compute_loss(y_true, y_pred)` | Calculate binary cross-entropy loss |

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `input_size` | Number of input features | Required |
| `hidden_size` | Neurons in hidden layer | Required |
| `output_size` | Number of outputs | Required |
| `learning_rate` | Learning rate for optimization | 0.1 |

## Understanding the Output

### Training Progress

When training, you'll see:
```
Epoch 100/1000, Loss: 0.4523, Accuracy: 82.50%
Epoch 200/1000, Loss: 0.3412, Accuracy: 86.25%
...
```

### Visualization

The example code generates two plots:
1. **Decision Boundary**: Shows how the model separates classes
2. **Training Loss**: Shows loss decreasing over epochs

## Common Issues and Solutions

### Issue: Loss Not Decreasing
- **Solution**: Try increasing the learning rate
- **Solution**: Increase hidden layer size
- **Solution**: Train for more epochs

### Issue: Overfitting
- **Solution**: Decrease hidden layer size
- **Solution**: Add regularization (not implemented)
- **Solution**: Use more training data

### Issue: Slow Training
- **Solution**: Decrease hidden layer size
- **Solution**: Use fewer training epochs
- **Solution**: Vectorize operations (already done)

## Extending the Network

### Adding More Hidden Layers

```python
class DeeperNetwork(SimpleNeuralNetwork):
    def __init__(self, layer_sizes, learning_rate=0.1):
        # Initialize multiple layers
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            W = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01
            b = np.zeros((1, layer_sizes[i+1]))
            self.layers.append({'W': W, 'b': b})
        self.learning_rate = learning_rate
```

### Changing Activation Functions

Replace `sigmoid` with:
```python
def relu(self, x):
    return np.maximum(0, x)

def tanh(self, x):
    return np.tanh(x)
```

## Testing

The code includes a test function that:
1. Generates synthetic data with a non-linear boundary
2. Trains the network
3. Visualizes results

Run it with:
```python
python neural_network.py
```

## Performance Notes

- **Complexity**: O(n_samples * n_features * hidden_size) per epoch
- **Memory**: Stores gradients and activations during training
- **Recommendation**: Scale features to [0, 1] or [-1, 1] for better convergence

## License

This code is provided as-is for educational purposes. Feel free to use and modify it.

## Further Reading

- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)
- [CS231n: Convolutional Neural Networks](http://cs231n.stanford.edu/)
- [NumPy Documentation](https://numpy.org/doc/stable/)
