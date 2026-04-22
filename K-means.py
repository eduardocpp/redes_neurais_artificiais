import numpy as np
import matplotlib.pyplot as plt

def k_means(X, k, max_epochs = 100):
    plt.scatter(X[:, 0], X[:, 1], c='black', marker='o')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid()
    plt.title('Dados originais')
    plt.show()

    centroids = X[np.random.choice(X.shape[0], k, replace=False)]

    for epoch in range(max_epochs):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])

        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids
        # plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o')
        # plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200)
        # plt.xlabel('X')
        # plt.ylabel('Y')
        # plt.grid()
        # plt.title(f'Resultado do K-Means iteracao {epoch+1}')
        # plt.show()

    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid()
    plt.title(f'Resultado do K-Means iteracao {epoch+1}')
    plt.show()
    return centroids, labels

if __name__ == "__main__":
    x1 = np.random.multivariate_normal(mean = [1,1], cov = [[1, 0], [0, 1]], size = 25)
    x2 = np.random.multivariate_normal(mean = [5,5], cov = [[1, 0], [0, 1]], size = 25)
    x3 = np.random.multivariate_normal(mean = [1,5], cov = [[1, 0], [0, 1]], size = 25)
    x4 = np.random.multivariate_normal(mean = [5,1], cov = [[1, 0], [0, 1]], size = 25)
    X = np.vstack((x1, x2, x3, x4))
    for i in range(4):
        k = i + 2
        centroids, labels = k_means(X, k)
        print(f"Centroides finais para k={k}:\n", centroids)



