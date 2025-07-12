MNIST Handwritten Digit Classifier Using JAX
🔍 Project Title
MNIST Handwritten Digit Classifier Using JAX

📄 Description
This is a simple neural network built from scratch using JAX , trained to recognize handwritten digits
from the MNIST dataset (0–9).

Framework: JAX
Dataset: MNIST
Model: 1 hidden layer neural network
Purpose: Learn how neural networks work using forward pass, ReLU, softmax, loss, and gradient
descent — all coded manually.
🖼️ What is MNIST?
The MNIST dataset contains 70,000 grayscale images of handwritten digits:

60,000 training images
10,000 testing images
Each image: 28 × 28 pixels, values range from 0 to 255
🧠 How Neural Networks Work (with Forward Pass)
👣 Step-by-step:
Input Image :
Each image is 28x28, flattened into a vector of 784 values (pixels).
Normalized between 0 and 1 by dividing by 255.
Weights Initialization :
W1 : (784 × 128) — connects input to hidden layer
b1 : (128,) — bias for hidden layer
W2 : (128 × 10) — connects hidden to output (10 digits)
b2 : (10,) — bias for output
• • • • • • •
1.
2.
3.
4.
5.
6.
7.
8.
Forward Pass:
# Hidden Layer
z1 = X @ W1 + b
a1 = relu(z1)
# Output Layer
z2 = a1 @ W2 + b
y_hat= softmax(z2)
relu(x): ReLU (Rectified Linear Unit) turns negative values to 0.
softmax(x): Converts logits into probabilities across 10 classes.
Prediction :
The digit with the highest probability is the predicted class.
🧪 Loss Function
defcross_entropy(preds, labels):
one_hot = jnp.eye(10)[labels] # one-hot encode
return -jnp.sum(one_hot * jnp.log(preds+ 1e-7))
Cross-entropy compares predicted probabilities to the actual label.
Lower loss = better prediction.
🔁 Training Loop
Loop for 1000 epochs
Each step:
Pick a sample (X_train[i])
Predict and calculate loss
Compute gradients using jax.grad
Update weights using simple gradient descent
grads= grad(loss_fn)(params, x, y)
params = {k: params[k] - lr * grads[k]for k inparams}
Every 100 epochs: Print accuracy on test data
9.
• • • • • •
1.
2.
3.
4.
5.
6.
1.
✅ Accuracy Output
Example:

Epoch 0, Accuracy: 0.
Epoch 100, Accuracy: 0.
Epoch 200, Accuracy: 0.
...
🛠️ Requirements
pipinstall jaxjaxlib tensorflow-datasets
📁 File Overview
smalljaxproject.py: Main script containing model, training, and evaluation
README.md: Documentation (this file)
✍️ Author
Tejas S

🔮 What to Try Next?
Add batching instead of one sample per step
Add more hidden layers
Use jax.jit to speed up training
Visualize weights and predictions
Build a Gradio web interface
