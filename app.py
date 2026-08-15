import numpy as np
import matplotlib.pyplot as plt

class SimpleNeuralNetwork:
    """
    A simple 2-layer neural network for binary classification.
    """
    
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        """
        Initialize the neural network with random weights and biases.
        
        Args:
            input_size: Number of input features
            hidden_size: Number of neurons in hidden layer
            output_size: Number of output neurons
            learning_rate: Learning rate for gradient descent
        """
        # Initialize weights and biases with small random values
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
        self.learning_rate = learning_rate
        
        # Store for visualization
        self.loss_history = []
    
    def sigmoid(self, x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))  # Clip to prevent overflow
    
    def sigmoid_derivative(self, x):
        """Derivative of sigmoid function."""
        return x * (1 - x)
    
    def forward(self, X):
        """
        Forward propagation through the network.
        
        Args:
            X: Input data of shape (m, input_size)
            
        Returns:
            A2: Final output predictions
            cache: Stored values for backpropagation
        """
        # Layer 1
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = self.sigmoid(Z1)
        
        # Layer 2 (output)
        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = self.sigmoid(Z2)
        
        cache = {
            'Z1': Z1, 'A1': A1,
            'Z2': Z2, 'A2': A2
        }
        
        return A2, cache
    
    def compute_loss(self, y_true, y_pred):
        """
        Compute binary cross-entropy loss.
        
        Args:
            y_true: True labels
            y_pred: Predicted probabilities
            
        Returns:
            loss: Binary cross-entropy loss
        """
        # Add small epsilon to prevent log(0)
        epsilon = 1e-8
        m = y_true.shape[0]
        loss = -np.mean(y_true * np.log(y_pred + epsilon) + 
                       (1 - y_true) * np.log(1 - y_pred + epsilon))
        return loss
    
    def backward(self, X, y_true, cache):
        """
        Backward propagation to compute gradients.
        
        Args:
            X: Input data
            y_true: True labels
            cache: Stored values from forward propagation
            
        Returns:
            gradients: Dictionary of weight and bias gradients
        """
        m = X.shape[0]
        
        # Extract cached values
        A1 = cache['A1']
        A2 = cache['A2']
        Z1 = cache['Z1']
        
        # Output layer gradient
        dZ2 = A2 - y_true  # derivative of binary cross-entropy with sigmoid
        dW2 = (1/m) * np.dot(A1.T, dZ2)
        db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)
        
        # Hidden layer gradient
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.sigmoid_derivative(A1)
        dW1 = (1/m) * np.dot(X.T, dZ1)
        db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)
        
        gradients = {
            'dW1': dW1, 'db1': db1,
            'dW2': dW2, 'db2': db2
        }
        
        return gradients
    
    def update_parameters(self, gradients):
        """
        Update weights and biases using gradient descent.
        
        Args:
            gradients: Dictionary of gradients
        """
        self.W1 -= self.learning_rate * gradients['dW1']
        self.b1 -= self.learning_rate * gradients['db1']
        self.W2 -= self.learning_rate * gradients['dW2']
        self.b2 -= self.learning_rate * gradients['db2']
    
    def train(self, X, y, epochs=1000, verbose=True):
        """
        Train the neural network.
        
        Args:
            X: Input data
            y: True labels
            epochs: Number of training iterations
            verbose: Whether to print progress
        """
        self.loss_history = []
        
        for epoch in range(epochs):
            # Forward propagation
            y_pred, cache = self.forward(X)
            
            # Compute loss
            loss = self.compute_loss(y, y_pred)
            self.loss_history.append(loss)
            
            # Backward propagation
            gradients = self.backward(X, y, cache)
            
            # Update parameters
            self.update_parameters(gradients)
            
            # Print progress
            if verbose and (epoch + 1) % 100 == 0:
                accuracy = self.accuracy(X, y)
                print(f'Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.2%}')
    
    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X: Input data
            
        Returns:
            predictions: Binary predictions (0 or 1)
        """
        y_pred, _ = self.forward(X)
        return (y_pred > 0.5).astype(int)
    
    def accuracy(self, X, y):
        """
        Calculate accuracy.
        
        Args:
            X: Input data
            y: True labels
            
        Returns:
            accuracy: Accuracy score
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)


def generate_data(n_samples=1000):
    """
    Generate synthetic data for testing.
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        X: Features
        y: Labels
    """
    np.random.seed(42)
    
    # Generate random points in [0, 1] range
    X = np.random.randn(n_samples, 2)
    
    # Create a non-linear boundary (circle)
    radius = np.sqrt(X[:, 0]**2 + X[:, 1]**2)
    y = (radius < 1.0).astype(int).reshape(-1, 1)
    
    # Add some noise
    noise_idx = np.random.choice(n_samples, size=int(0.1 * n_samples), replace=False)
    y[noise_idx] = 1 - y[noise_idx]
    
    return X, y


def plot_decision_boundary(model, X, y):
    """
    Plot the decision boundary of the model.
    
    Args:
        model: Trained neural network
        X: Input data
        y: True labels
    """
    # Create a meshgrid
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    h = 0.02
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Predict on meshgrid
    X_mesh = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(X_mesh)
    Z = Z.reshape(xx.shape)
    
    # Plot
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
    plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap=plt.cm.RdYlBu, edgecolors='black', alpha=0.6)
    plt.title('Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    plt.subplot(1, 2, 2)
    plt.plot(model.loss_history)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()


def main():
    """Main function to run the neural network."""
    print("=" * 50)
    print("SIMPLE NEURAL NETWORK")
    print("=" * 50)
    
    # Generate data
    print("\nGenerating synthetic data...")
    X, y = generate_data(1000)
    print(f"Data shape: {X.shape}, Labels shape: {y.shape}")
    
    # Split into train and test sets
    split_idx = int(0.8 * X.shape[0])
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Create and train the model
    print("\nCreating neural network...")
    model = SimpleNeuralNetwork(
        input_size=2,
        hidden_size=10,
        output_size=1,
        learning_rate=0.5
    )
    
    print("\nTraining neural network...")
    model.train(X_train, y_train, epochs=1000, verbose=True)
    
    # Evaluate
    train_acc = model.accuracy(X_train, y_train)
    test_acc = model.accuracy(X_test, y_test)
    
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"Training Accuracy: {train_acc:.2%}")
    print(f"Test Accuracy: {test_acc:.2%}")
    
    # Plot results
    print("\nGenerating plots...")
    plot_decision_boundary(model, X, y)


if __name__ == "__main__":
    main()
