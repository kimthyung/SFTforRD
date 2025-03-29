import torch
import os
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ELM:
    def __init__(self, input_size, hidden_size, output_size, device=device):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.device = device

        torch.manual_seed(42)
        self.weights_input = torch.randn(input_size, hidden_size, device=self.device)
        self.biases_hidden = torch.randn(hidden_size, device=self.device)
        self.weights_output = torch.randn(hidden_size, output_size, device=self.device)

    def _activation(self, x):
        return torch.sigmoid(x)

    def train(self, X, y):
        H = self._activation(torch.matmul(X, self.weights_input) + self.biases_hidden)
        H_pseudo_inverse = torch.pinverse(H)
        self.weights_output = torch.matmul(H_pseudo_inverse, y)

    def predict(self, X):
        H = self._activation(torch.matmul(X, self.weights_input) + self.biases_hidden)
        return torch.matmul(H, self.weights_output)

    def update_weights(self, X, y, learning_rate=0.01):
        predictions = self.predict(X)
        error = y - predictions
        H = self._activation(torch.matmul(X, self.weights_input) + self.biases_hidden)
        self.weights_output += learning_rate * torch.matmul(H.T, error)

    def save_model(self, path):
        model_data = {
            'weights_input': self.weights_input,
            'biases_hidden': self.biases_hidden,
            'weights_output': self.weights_output
        }
        torch.save(model_data, path)
        print(f'Model saved to {path}')

    def load_model(self, path):
        model_data = torch.load(path)
        self.weights_input = model_data['weights_input']
        self.biases_hidden = model_data['biases_hidden']
        self.weights_output = model_data['weights_output']
        print(f'Model loaded from {path}')

def create_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Elbow Method
def plot_elbow_method(X, max_clusters=20):
    wcss = []
    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, max_clusters + 1), wcss, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.grid(True)
    plt.show()

# Silhouette Analysis
def plot_silhouette_scores(X, max_clusters=10):
    silhouette_scores = []
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    
    plt.figure(figsize=(8, 6))
    plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
    plt.title('Silhouette Analysis')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.grid(True)
    plt.show()