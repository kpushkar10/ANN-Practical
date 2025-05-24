import tensorflow as tf

# Load the model (even if it was saved with TF 2.15+)
model = tf.keras.models.load_model("model.h5")  # or model.keras

# Re-save with TF 2.13 to downgrade serialization
model.save("model_tf13.keras", save_format="keras")