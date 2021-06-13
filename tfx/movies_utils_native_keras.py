from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import List, Text

import absl
import tensorflow as tf
from tensorflow import keras
import tensorflow_transform as tft

from tfx.components.trainer.fn_args_utils import FnArgs

_FEATURE_KEY = 'overview'
_LABEL_KEY = 'genres'

_DROPOUT_RATE = 0.2
_EMBEDDING_UNITS = 64
_EVAL_BATCH_SIZE = 5
_HIDDEN_UNITS = 64
_LEARNING_RATE = 1e-4
_LSTM_UNITS = 64
_VOCAB_SIZE = 8000
_MAX_LEN = 400
_TRAIN_BATCH_SIZE = 10


def _transformed_name(key, is_input=False):
  return key + ('_xf_input' if is_input else '_xf')


def _gzip_reader_fn(filenames):
  """Small utility returning a record reader that can read gzip'ed files."""
  return tf.data.TFRecordDataset(filenames, compression_type='GZIP')


def _tokenize_review(review):
  """Tokenize the overview by spliting the overview.

  Constructing a vocabulary. Map the words to their frequency index in the
  vocabulary.

  Args:
    review: tensors containing the overview. (batch_size/None, 1)

  Returns:
    Tokenized and padded review tensors. (batch_size/None, _MAX_LEN)
  """
  review_sparse = tf.strings.split(tf.reshape(review, [-1])).to_sparse()
  # tft.apply_vocabulary doesn't reserve 0 for oov words. In order to comply
  # with convention and use mask_zero in keras.embedding layer, set oov value
  # to _VOCAB_SIZE and padding value to -1. Then add 1 to all the tokens.
  review_indices = tft.compute_and_apply_vocabulary(
      review_sparse, default_value=_VOCAB_SIZE, top_k=_VOCAB_SIZE)
  dense = tf.sparse.to_dense(review_indices, default_value=-1)
  # TFX transform expects the transform result to be FixedLenFeature.
  padding_config = [[0, 0], [0, _MAX_LEN]]
  dense = tf.pad(dense, padding_config, 'CONSTANT', -1)
  padded = tf.slice(dense, [0, 0], [-1, _MAX_LEN])
  padded += 1
  return padded


# TFX Transform will call this function.
def preprocessing_fn(inputs):
  """tf.transform's callback function for preprocessing inputs.

  Args:
    inputs: map from feature keys to raw not-yet-transformed features.

  Returns:
    Map from string feature key to transformed feature operations.
  """
  print(inputs[_LABEL_KEY])
  return {
      _transformed_name(_LABEL_KEY):
          inputs[_LABEL_KEY],
      _transformed_name(_FEATURE_KEY, True):
          _tokenize_review(inputs[_FEATURE_KEY])
  }


def _input_fn(file_pattern: List[Text],
              tf_transform_output: tft.TFTransformOutput,
              batch_size: int = 200) -> tf.data.Dataset:
  """Generates features and label for tuning/training.

  Args:
    file_pattern: List of paths or patterns of input tfrecord files.
    tf_transform_output: A TFTransformOutput.
    batch_size: representing the number of consecutive elements of returned
      dataset to combine in a single batch.

  Returns:
    A dataset that contains (features, indices) tuple where features is a
      dictionary of Tensors, and indices is a single Tensor of label indices.
  """
  transformed_feature_spec = (
      tf_transform_output.transformed_feature_spec().copy())

  dataset = tf.data.experimental.make_batched_features_dataset(
      file_pattern=file_pattern,
      batch_size=batch_size,
      features=transformed_feature_spec,
      reader=_gzip_reader_fn,
      label_key=_transformed_name(_LABEL_KEY))

  return dataset.repeat()


def _build_keras_model() -> keras.Model:
  model = keras.Sequential([
      keras.layers.Embedding(
          _VOCAB_SIZE + 2,
          _EMBEDDING_UNITS,
          name=_transformed_name(_FEATURE_KEY)),
      keras.layers.Bidirectional(
          keras.layers.LSTM(_LSTM_UNITS, dropout=_DROPOUT_RATE)),
      keras.layers.Dense(_HIDDEN_UNITS, activation='relu'),
      keras.layers.Dense(20)
  ])

  model.compile(
      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      optimizer=keras.optimizers.Adam(_LEARNING_RATE),
      metrics=['accuracy'])

  model.summary(print_fn=absl.logging.info)
  return model


def _get_serve_tf_examples_fn(model, tf_transform_output):
  """Returns a function that parses a serialized tf.Example."""
  model.tft_layer = tf_transform_output.transform_features_layer()

  @tf.function
  def serve_tf_examples_fn(serialized_tf_examples):
    """Returns the output to be used in the serving signature."""
    feature_spec = tf_transform_output.raw_feature_spec()
    feature_spec.pop(_LABEL_KEY)
    parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
    transformed_features = model.tft_layer(parsed_features)
    return model(transformed_features)

  return serve_tf_examples_fn


# TFX Trainer will call this function.
def run_fn(fn_args: FnArgs):
  """Train the model based on given args.

  Args:
    fn_args: Holds args used to train the model as name/value pairs.
  """
  tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

  train_dataset = _input_fn(
      fn_args.train_files, tf_transform_output, batch_size=_TRAIN_BATCH_SIZE)

  eval_dataset = _input_fn(
      fn_args.eval_files, tf_transform_output, batch_size=_EVAL_BATCH_SIZE)

  mirrored_strategy = tf.distribute.MirroredStrategy()
  with mirrored_strategy.scope():
    model = _build_keras_model()

  model.fit(
      train_dataset,
      steps_per_epoch=fn_args.train_steps,
      validation_data=eval_dataset,
      validation_steps=fn_args.eval_steps)

  signatures = {
      'serving_default':
          _get_serve_tf_examples_fn(model,
                                    tf_transform_output).get_concrete_function(
                                        tf.TensorSpec(
                                            shape=[None],
                                            dtype=tf.string,
                                            name='examples')),
  }

  model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)
