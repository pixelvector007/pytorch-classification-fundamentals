# PyTorch Workflow Fundamentals

A hands-on introduction to the **core PyTorch workflow**, starting with a linear regression model built from scratch and then rebuilding the same model using PyTorch's built-in `nn.Linear` layer.

The goal is to understand what happens inside a PyTorch training workflow before moving on to more complex neural networks and deep learning projects.

---

## 📂 Notebooks

### 01 — PyTorch Workflow Fundamentals

`01_pytorch_workflow_fundamentals.ipynb`

Builds a linear regression model manually using `nn.Parameter`.

Topics covered:

- Creating synthetic data
- Training and testing data split
- Data visualization
- Building a model with `nn.Module`
- Manually creating trainable `weight` and `bias`
- Understanding `nn.Parameter`
- `state_dict()`
- `L1Loss`
- `SGD` optimizer
- Training loop
- Forward pass
- Backpropagation
- Parameter updates
- `model.train()`
- `model.eval()`
- `torch.inference_mode()`
- Loss curves
- Saving model weights
- Loading a saved `state_dict`
- Verifying model predictions after loading

This notebook focuses on understanding **what PyTorch is doing under the hood**.

---

### 02 — PyTorch Workflow with `nn.Linear`

`02_pytorch_workflow_with_nn_linear.ipynb`

Rebuilds the same linear regression problem using PyTorch's built-in `nn.Linear` layer.

Topics covered:

- Synthetic dataset creation
- Train/test split
- Device-agnostic PyTorch code
- `torch.cuda.is_available()`
- Building models with `nn.Linear`
- Model parameters and `state_dict`
- `L1Loss`
- `SGD`
- Training loop
- Model evaluation
- `torch.inference_mode()`
- Saving and loading model weights
- Comparing predictions before and after loading

This notebook demonstrates the **standard and more practical way** of defining a linear model in PyTorch.

---

## 🔄 PyTorch Workflow

Both notebooks follow the fundamental PyTorch workflow:

```text
Data
 ↓
Train / Test Split
 ↓
Build Model
 ↓
Define Loss Function
 ↓
Define Optimizer
 ↓
Training Loop
 ↓
Evaluate Model
 ↓
Make Predictions
 ↓
Save Model
 ↓
Load Model
 ↓
Verify Predictions
```

The main difference is how the model is created:

```text
Notebook 01
nn.Module
   ↓
nn.Parameter
   ↓
Manual weight + bias
```

```text
Notebook 02
nn.Module
   ↓
nn.Linear
   ↓
Built-in weight + bias
```

---

## 🧠 Key Concepts Learned

### `nn.Parameter`

`nn.Parameter` tells PyTorch that a tensor should be treated as a **trainable model parameter**.

```python
self.weights = nn.Parameter(torch.randn(1))
self.bias = nn.Parameter(torch.randn(1))
```

This is essentially what happens internally when using layers such as `nn.Linear`.

### `nn.Linear`

Instead of manually defining weights and bias:

```python
self.linear_layer = nn.Linear(
    in_features=1,
    out_features=1
)
```

`nn.Linear` handles the parameters and linear transformation for us.

### Training Loop

The fundamental training cycle is:

```text
Forward Pass
     ↓
Calculate Loss
     ↓
Zero Gradients
     ↓
Backpropagation
     ↓
Update Parameters
```

Implemented using:

```python
y_pred = model(X_train)

loss = loss_fn(y_pred, y_train)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

### `train()` and `eval()`

```python
model.train()
```

puts the model into training mode.

```python
model.eval()
```

puts the model into evaluation mode.

These become especially important when using layers such as **Dropout** and **BatchNorm**.

### `torch.inference_mode()`

Used when making predictions without needing gradients:

```python
with torch.inference_mode():
    predictions = model(X_test)
```

This avoids unnecessary gradient tracking during inference.

### `state_dict()`

A model's learned parameters can be saved using:

```python
torch.save(model.state_dict(), MODEL_SAVE_PATH)
```

and loaded into a new model using:

```python
model.load_state_dict(torch.load(MODEL_SAVE_PATH))
```

---

## 💻 Technologies Used

- Python
- PyTorch
- NumPy
- Matplotlib
- Jupyter Notebook

---

## 🎯 Learning Progression

These notebooks establish the foundation for moving toward more advanced PyTorch concepts:

```text
Linear Regression
       ↓
PyTorch Training Workflow
       ↓
nn.Linear
       ↓
Datasets & DataLoaders
       ↓
Neural Networks
       ↓
Computer Vision
       ↓
CNNs
       ↓
Deep Learning Projects
```

---

## 📌 Purpose

This repository is part of my journey toward building a strong foundation in **Machine Learning, Deep Learning, and Artificial Intelligence**.

The notebooks focus on understanding the fundamentals through implementation rather than simply using high-level APIs.

---

## ⭐ Future Topics

Planned progression includes:

- PyTorch `Dataset` and `DataLoader`
- Neural network classification
- `nn.Sequential`
- Activation functions
- Loss functions for classification
- Optimizers
- Model evaluation
- Computer vision
- CNNs
- Transfer learning
- Experiment tracking

---

## 🙌 Thanks for Visiting!

I'm continuously learning and building projects in **Machine Learning, Deep Learning, and Artificial Intelligence**.

If you find this repository useful, consider giving it a ⭐!

**Made with ❤️ by Arpit Kushwaha**
