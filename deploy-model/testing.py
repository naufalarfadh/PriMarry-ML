from flask import Flask, request
import tensorflow as tf
import numpy as np

model_path = 'D:\\OPUNK!\\Bangkit Machine Learning Batch 2\\Capstone\\deploy-model\\model\\model_20231219-163109\\'
model = tf.keras.models.load_model(model_path)

print(model.summary())

data = "apa sih bodoh"
print(model.predict(np.expand_dims(data, 0)))