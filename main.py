import tensorflow as tf
import argparse
from clean_data import cleaner

# get the params from cmd line
parser = argparse.ArgumentParser(description='Movie classifier')
parser.add_argument("--title", default=None, type=str, help="Movie title")
parser.add_argument("--description", default=None, type=str, help="Movie description")

args = parser.parse_args()
# Lables of the classification we are using
lables = ['Animation', 'Adventure', 'Romance', 'Comedy', 'Action', 'Family', 'History',
          'Drama', 'Crime', 'Fantasy', 'Science Fiction', 'Thriller', 'Music', 'Horror',
          'Documentary', 'Mystery', 'Western', 'TV Movie', 'War', 'Foreign']


# Clean the descrption using our clean function
clean_description = cleaner(args.title + " " + args.description)

inputs = [clean_description]

# inputs = [
#     clean_description,
#     "During a dangerous mission to stop a drug cartel operating between the US and Mexico, Kate Macer, an FBI agent, is exposed to some harsh realities.",
#     "Tony Montana and his close friend Manny, build a strong drug empire in Miami. However as his power begins to grow, so does his ego and his enemies, and his own paranoia begins to plague his empire",
#     "swing life characters movie described swing day week"
# ]

# Load the best model saved suring training
new_model = tf.keras.models.load_model('model')
# Predict the class using the model
predicted_scores = new_model.predict(inputs)
predicted_labels = tf.argmax(predicted_scores, axis=1)
for input, label in zip(inputs, predicted_labels):
  print("Question: ", input)
  print("Predicted label: ", label.numpy())
genre = lables[label.numpy()]
# Return the result
result = {"title": args.title, "description": args.description, "genre": genre}
print(result)

a = input('Press a key to exit')
if a:
  exit(0)
