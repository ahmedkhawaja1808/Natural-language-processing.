# AI Spam Email Classifier

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Sample dataset
messages = [
    "Win a free iPhone now",
    "Limited offer just for you",
    "Congratulations you won a prize",
    "Meeting at 10 am tomorrow",
    "Let's complete the assignment",
    "Project discussion at university"
]

labels = [1, 1, 1, 0, 0, 0]  # 1 = Spam, 0 = Not Spam

# Convert text into numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(messages)

# Train model
model = MultinomialNB()
model.fit(X, labels)

def predict_message():
    msg = input("Enter your message: ")
    msg_vector = vectorizer.transform([msg])
    prediction = model.predict(msg_vector)

    if prediction[0] == 1:
        print("🚫 This is SPAM\n")
    else:
        print("✅ This is NOT SPAM\n")

# Menu
while True:
    print("=== AI Spam Classifier ===")
    print("1. Check Message")
    print("2. Exit")

    choice = input("Enter choice: ")

    if choice == "1":
        predict_message()
    elif choice == "2":
        print("Goodbye!")
        break
    else:
        print("Invalid choice\n")
