import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


print("\n===== IMAGE PROCESSING =====")

image = np.random.rand(64, 64)

blur_kernel = np.ones((3, 3)) / 9

blurred = np.zeros_like(image)

for i in range(1, image.shape[0] - 1):
    for j in range(1, image.shape[1] - 1):
        region = image[i - 1:i + 2, j - 1:j + 2]
        blurred[i, j] = np.sum(region * blur_kernel)

plt.imshow(blurred, cmap="gray")
plt.title("Blurred Image")
plt.axis("off")
plt.show()


print("\n===== COMPUTER VISION EDGE DETECTION =====")

sobel_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

edges = np.zeros_like(image)

for i in range(1, image.shape[0] - 1):
    for j in range(1, image.shape[1] - 1):
        region = image[i - 1:i + 2, j - 1:j + 2]
        edges[i, j] = np.sum(region * sobel_x)

plt.imshow(edges, cmap="gray")
plt.title("Edge Detection")
plt.axis("off")
plt.show()


print("\n===== CNN DEMO =====")

input_image = np.random.rand(28, 28)

filter_kernel = np.random.rand(3, 3)

feature_map = np.zeros((26, 26))

for i in range(26):
    for j in range(26):
        region = input_image[i:i + 3, j:j + 3]
        feature_map[i, j] = np.sum(region * filter_kernel)

relu_output = np.maximum(0, feature_map)

print("Feature Map Shape:", relu_output.shape)

plt.imshow(relu_output, cmap="gray")
plt.title("CNN Feature Map")
plt.axis("off")
plt.show()


print("\n===== IMAGE CLASSIFICATION =====")

X = np.random.rand(300, 256)
y = np.random.randint(0, 2, 300)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

clf = RandomForestClassifier()

clf.fit(X_train, y_train)

pred = clf.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, pred))


print("\n===== SIMPLE RNN =====")

input_size = 3
hidden_size = 5

Wxh = np.random.randn(hidden_size, input_size)
Whh = np.random.randn(hidden_size, hidden_size)

h = np.zeros((hidden_size, 1))

sequence = [
    np.random.randn(input_size, 1)
    for _ in range(5)
]

for x in sequence:
    h = np.tanh(
        np.dot(Wxh, x) +
        np.dot(Whh, h)
    )

print("Final Hidden State:")
print(h)