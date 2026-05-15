import argparse
import datasets
import pandas
import transformers
import numpy
from tf_keras import layers, optimizers, losses, metrics, Input, Model, callbacks, models
from transformers import TFAutoModel, TFRobertaModel

tokenizer = transformers.AutoTokenizer.from_pretrained("distilroberta-base")

def tokenize(examples):
    """
    Converts the text of each example to "input_ids", a sequence of integers
    representing 1-hot vectors for each token in the text.
    """
    return tokenizer(examples["text"], truncation=True, max_length=64,
                     padding="max_length")


def train(model_path="model_final.keras", train_path="train.csv", dev_path="dev.csv"):
  hf_dataset = datasets.load_dataset("csv", data_files={
        "train": train_path, "validation": dev_path})
  labels = hf_dataset["train"].column_names[1:]

  def gather_labels(example):
        """Converts the label columns into a list of 0s and 1s"""
        # the float here is because F1Score requires floats
        return {"labels": [float(example[l]) for l in labels]}

  # convert text and labels to format expected by model
  hf_dataset = hf_dataset.map(gather_labels)
  hf_dataset = hf_dataset.map(tokenize, batched=True)

  # convert Huggingface datasets to Tensorflow datasets
  train_dataset = hf_dataset["train"].to_tf_dataset(
      #tranformers like attention masks so I added some
      columns=["input_ids", "attention_mask"],
      label_cols="labels",
      batch_size=32,
      shuffle=True)
  dev_dataset = hf_dataset["validation"].to_tf_dataset(
      columns=["input_ids", "attention_mask"],
      label_cols="labels",
      batch_size=32)
  #For full transparency, I was inspired for this method by scrolling through Keras tutorials https://keras.io/examples/nlp/semantic_similarity_with_bert/ 
  input_ids = Input(shape=(64,), dtype="int32", name="input_ids")
  attention_mask = Input(shape=(64,), dtype="int32", name="attention_mask")
  transformer = TFAutoModel.from_pretrained("distilroberta-base")
  output = transformer(input_ids, attention_mask=attention_mask)
  last_hidden_state = output.last_hidden_state
  pooled = layers.GlobalAveragePooling1D()(last_hidden_state)
  final_output = layers.Dense(len(labels), activation="sigmoid")(pooled)
  model = Model(inputs=[input_ids, attention_mask], outputs=final_output)
  model.compile(
      #lowered the optimizer because finetuning a large model
      optimizer=optimizers.Adam(1e-5),
      loss=losses.binary_crossentropy,
      metrics=[metrics.F1Score(average="micro", threshold=0.5)])
  # fit the model to the training data, monitoring F1 on the dev data
  model.fit(
      train_dataset,
      epochs=10,
      validation_data=dev_dataset,
      callbacks=[
        callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor="val_f1_score",
            mode="max",
            save_best_only=True),
        callbacks.EarlyStopping(
            # Added early stopping to prevent overfitting
            monitor="val_f1_score",
            mode="max",
            patience=2)
    ])
  
  def predict(model_path="model_final.keras", input_path="test-in.csv"):
    # load the saved model
    model = models.load_model(
        model_path, custom_objects={"TFRobertaModel": TFRobertaModel})

    # load the data for prediction
    # use Pandas here to make assigning labels easier later
    df = pandas.read_csv(input_path)

    # create input features in the same way as in train()
    hf_dataset = datasets.Dataset.from_pandas(df)
    hf_dataset = hf_dataset.map(tokenize, batched=True)
    tf_dataset = hf_dataset.to_tf_dataset(
        columns=["input_ids", "attention_mask"],
        batch_size=64)

    # generate predictions from model
    predictions = numpy.where(model.predict(tf_dataset) > 0.5, 1, 0)

    # assign predictions to label columns in Pandas data frame
    df.iloc[:, 1:] = predictions

    # write the Pandas dataframe to a zipped CSV file
    df.to_csv("submission_final.zip", index=False, compression=dict(
        method='zip', archive_name=f'submission.csv'))

if __name__ == "__main__":
    # parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices={"train", "predict"})
    args = parser.parse_args()

    # call either train() or predict()
    globals()[args.command]()