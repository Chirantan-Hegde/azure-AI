import numpy as np

print("\n===== SIMPLE CHATBOT USING RNN + LSTM CONCEPT =====")

pairs = {
    "hello": "hi there",
    "hi": "hello",
    "how are you": "i am fine",
    "what is ai": "artificial intelligence",
    "bye": "goodbye"
}

vocab = set()

for q, a in pairs.items():
    vocab.update(q.split())
    vocab.update(a.split())

vocab = sorted(list(vocab))

word_to_idx = {
    word: idx
    for idx, word in enumerate(vocab)
}

idx_to_word = {
    idx: word
    for word, idx in word_to_idx.items()
}

vocab_size = len(vocab)

Wxh = np.random.randn(16, vocab_size)
Whh = np.random.randn(16, 16)
Why = np.random.randn(vocab_size, 16)

def one_hot(word):
    vec = np.zeros((vocab_size, 1))
    if word in word_to_idx:
        vec[word_to_idx[word]] = 1
    return vec

def rnn_forward(words):
    h = np.zeros((16, 1))

    for word in words:
        x = one_hot(word)

        h = np.tanh(
            np.dot(Wxh, x) +
            np.dot(Whh, h)
        )

    y = np.dot(Why, h)

    return y

def chatbot_response(text):
    text = text.lower()

    if text in pairs:
        return pairs[text]

    words = text.split()

    if len(words) == 0:
        return "please say something"

    rnn_forward(words)

    return "sorry, i do not understand"

print("\nChatbot Ready")
print("Type 'exit' to quit\n")

while True:

    user = input("You: ")

    if user.lower() == "exit":
        print("Bot: Goodbye!")
        break

    print("Bot:", chatbot_response(user))