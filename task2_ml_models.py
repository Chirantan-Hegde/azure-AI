import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score


print("\n===== DECISION TREE =====")

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

pred = dt.predict(X_test)

print("Decision Tree Accuracy:", accuracy_score(y_test, pred))

plt.figure(figsize=(10, 6))
plot_tree(
    dt,
    filled=True,
    feature_names=iris.feature_names,
    class_names=iris.target_names
)
plt.title("Decision Tree")
plt.show()


print("\n===== RANDOM FOREST =====")

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

pred = rf.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, pred))

plt.figure(figsize=(8, 5))
plt.bar(
    iris.feature_names,
    rf.feature_importances_
)
plt.title("Random Forest Feature Importance")
plt.xticks(rotation=20)
plt.show()


print("\n===== SUPPORT VECTOR MACHINE =====")

svm = SVC(kernel="rbf")

svm.fit(X_train, y_train)

pred = svm.predict(X_test)

print("SVM Accuracy:", accuracy_score(y_test, pred))


print("\n===== PCA =====")

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X)

plt.figure(figsize=(8, 5))
plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=y,
    cmap="viridis"
)
plt.title("PCA Projection")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()


print("\n===== REINFORCEMENT LEARNING (Q TABLE) =====")

grid_size = 5

q_table = np.zeros((grid_size, grid_size, 4))

alpha = 0.1
gamma = 0.9
epsilon = 0.2

goal = (4, 4)

for episode in range(500):

    state = [0, 0]

    while tuple(state) != goal:

        if np.random.rand() < epsilon:
            action = np.random.randint(4)
        else:
            action = np.argmax(
                q_table[state[0], state[1]]
            )

        next_state = state.copy()

        if action == 0 and state[0] > 0:
            next_state[0] -= 1
        elif action == 1 and state[0] < 4:
            next_state[0] += 1
        elif action == 2 and state[1] > 0:
            next_state[1] -= 1
        elif action == 3 and state[1] < 4:
            next_state[1] += 1

        reward = 10 if tuple(next_state) == goal else -1

        old_q = q_table[state[0], state[1], action]

        next_max = np.max(
            q_table[next_state[0], next_state[1]]
        )

        q_table[state[0], state[1], action] = (
            old_q +
            alpha * (
                reward +
                gamma * next_max -
                old_q
            )
        )

        state = next_state

print("Q Learning Completed")


print("\n===== SIMPLE LSTM CELL =====")

input_size = 3
hidden_size = 4

x = np.random.randn(input_size)
h_prev = np.random.randn(hidden_size)
c_prev = np.random.randn(hidden_size)

Wf = np.random.randn(hidden_size, input_size + hidden_size)

combined = np.concatenate([h_prev, x])

forget_gate = 1 / (
    1 + np.exp(-np.dot(Wf, combined))
)

print("Forget Gate Output:")
print(forget_gate)


print("\n===== SIMPLE Q NETWORK =====")

input_nodes = 4
hidden_nodes = 16
output_nodes = 2

W1 = np.random.randn(
    input_nodes,
    hidden_nodes
)

W2 = np.random.randn(
    hidden_nodes,
    output_nodes
)

sample_state = np.random.randn(1, input_nodes)

hidden = np.maximum(
    0,
    np.dot(sample_state, W1)
)

q_values = np.dot(hidden, W2)

print("Q Values:")
print(q_values)