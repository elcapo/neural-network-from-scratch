# Building a Neural Network from Scratch

## The Math

### Forward Propagation

#### Input Layer

If $m$ is the number of samples in a batch, then our input matrix $\mathbf{X}_i$ has one sample per column.

$$
    \mathbf{X}_i
$$

The input layer represents this tensor without any modification. 

##### Dimensions

* $\mathbf{X}_i$ has dimensions $(784, m)$

#### Hidden Layer

Once the input value is passed to the hidden layer, its value will be transformed by both the layer weights $\mathbf{W}_h$ and its biases $\mathbf{b}_h$ and then activated by a $ReLU$ function.

$$
    \mathbf{Z}_h = \mathbf{W}_h \mathbf{X}_i + \mathbf{b}_h
$$

$$
    \mathbf{A}_h = ReLU(\mathbf{Z}_h)
$$

##### Dimensions

* $\mathbf{W}_h$ has dimensions $(10, 784)$
* $\mathbf{b}_h$ has dimensions $(10, 1)$
* $\mathbf{Z}_h$ has dimensions $(10, m)$
* $\mathbf{A}_h$ has dimensions $(10, m)$

##### Activation

The definition of the $ReLU$ function is:

```math
    ReLU(z) =

    \begin{cases}
        0 & \text{if } z <= 0 \\
        z & \text{if } z > 0
    \end{cases}
```

#### Output Layer

The activated value of the hidden layer is finally passed to the output layer, where its value will be transformed by both the layer weigths $\mathbf{W}_o$ and its biases $\mathbf{b}_o$ and then activated by a $Softmax$ function.

$$
    \mathbf{Z}_o = \mathbf{W}_o \mathbf{A}_h + \mathbf{b}_o
$$

$$
    \mathbf{A}_o = Softmax(\mathbf{Z}_o)
$$

##### Dimensions

* $\mathbf{W}_o$ has dimensions $(10, 10)$
* $\mathbf{b}_o$ has dimensions $(10, 1)$
* $\mathbf{Z}_o$ has dimensions $(10, m)$
* $\mathbf{A}_o$ has dimensions $(10, m)$

##### Activation

The definition of the $Softmax$ function is:

$$
    Softmax(\mathbf{z})_i = \frac{e^z_i}{\sum_{j} e^{z_j}}
$$

### Backward Propagation

Backward propagation computes gradients of the loss function with respect to all parameters and activations, enabling parameter updates via gradient descent. We assume a cross-entropy loss function $\mathcal{L}$ that measures the difference between predicted probabilities $\mathbf{A}_o$​ and ground-truth labels $\mathbf{Y}$.

#### Loss Function

We'll use a cross-entropy to compute our losses:

$$
\mathcal{L} = - \frac{1}{m} \sum_{i=1}^{m} \sum_{j=1}^{10} Y_{ij} log(A_{o,ij})
$$

#### Output Layer Activation

The gradient of the loss with respect to the output layer pre-activation is:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{Z}_o} = \mathbf{A}_o - \mathbf{Y}
$$

Using the chain rule and the property that the derivative of cross-entropy loss combined with softmax activation simplifies to this difference. A [detailed proof](softmax-derivative.md) is available.

##### Dimensions

* $\frac{\partial \mathcal{L}}{\partial \mathbf{Z}_o}$ has dimensions $(10, m)$

#### Output Layer Parameters

The gradients of the weights and biases of the output layer follow:

$$
    \frac{\partial \mathcal{L}}{\partial \mathbf{W}_o} = \frac{1}{m} \frac{\partial \mathcal{L}}{\partial \mathbf{Z}_o} \mathbf{A}_h^T
$$

$$
    \frac{\partial \mathcal{L}}{\partial \mathbf{b}_o} = \frac{1}{m} \sum \frac{\partial \mathcal{L}}{\partial \mathbf{Z}_o}
$$

Which we obtained by applying the chain rule:

$$
    \frac{\partial \mathcal{L}}{\partial \mathbf{W}_o} = \frac{\partial \mathcal{L}}{\partial \mathbf{Z}_o} \frac{\partial \mathbf{Z}_o}{\partial \mathbf{W}_o}
$$

$$
    \frac{\partial \mathcal{L}}{\partial \mathbf{b}_o} = \frac{\partial \mathcal{L}}{\partial \mathbf{Z}_o} \frac{\partial \mathbf{Z}_o}{\partial \mathbf{b}_o}
$$

##### Dimensions

* $\frac{\partial \mathcal{L}}{\partial \mathbf{W}_o}$ has the same dimensions as $\mathbf{W}_o$ $(10, 10)$
* $\frac{\partial \mathcal{L}}{\partial \mathbf{b}_o}$ has the same dimensions as $\mathbf{b}_o$ $(10, 1)$

#### Hidden Layer Activation

To backpropagate the error through the hidden layer, we start by computing the gradient of the loss with respect to the hidden layer’s activation, which we already defined as:

$$
    \frac{\partial \mathcal{L}}{\partial \mathbf{A}_h} = \mathbf{W}_o^T \frac{\partial \mathcal{L}}{\partial \mathbf{Z}_o}
$$

Which we obtained by applying the chain rule:

$$
    \frac{\partial \mathcal{L}}{\partial \mathbf{A}_h} = \frac{\partial \mathcal{L}}{\partial \mathbf{Z}_o} \frac{\partial \mathbf{Z}_o}{\partial \mathbf{A}_h}
$$

Next, we need the gradient with respect to the hidden layer’s pre-activation values $\mathbf{Z}_h$. Since the hidden layer uses the ReLU activation function, we apply its derivative element-wise:

$$
    \frac{\partial \mathcal{L}}{\partial {Z}_h} = \frac{\partial \mathcal{L}}{\partial \mathbf{A}_h} \odot \frac{\partial}{\partial \mathbf{Z}_h} ReLU({\mathbf{Z}_h})
$$

Where $\odot$ represents the element-wise (Hadamard) product and the derivative of ReLU, which is given by:

```math
    \frac{\partial}{\partial z} ReLU(z) =

    \begin{cases}
        0 & \text{if } z <= 0 \\
        1 & \text{if } z > 0
    \end{cases}
```

##### Dimensions

* $\frac{\partial \mathcal{L}}{\partial {Z}_h}$ has the same dimensions as $\mathbf{Z}_h$ $(10, m)$

#### Hidden Layer Parameters

Once we have $d\mathbf{Z}_h$, we can compute the gradients of the weights and biases for the hidden layer:

$$
    \frac{\partial \mathcal{L}}{\partial {W}_h} = \frac{1}{m} \frac{\partial \mathcal{L}}{\partial {Z}_h} \mathbf{X}_i^T
$$

$$
    \frac{\partial \mathcal{L}}{\partial {b}_h} = \frac{1}{m} \sum \frac{\partial \mathcal{L}}{\partial {Z}_h}
$$

##### Dimensions

* $\frac{\partial \mathcal{L}}{\partial {W}_h}$ has the same dimensions as $\mathbf{W}_h$ $(10, 784)$
* $\frac{\partial \mathcal{L}}{\partial {b}_h}$ has the same dimensions as $\mathbf{b}_h$ $(10, 1)$

### Gradient Descent

Once all gradients are computed, the final step of backpropagation is to update the network’s parameters so that the loss decreases in the next iteration.

Each weight and bias matrix is adjusted in the opposite direction of its gradient, scaled by a small constant called the learning rate ($\eta$):

$$
\begin{aligned}
    & \mathbf{W}_o \leftarrow \mathbf{W}_o - \eta \cdot d\mathbf{W}_o \\
    & \mathbf{b}_o \leftarrow \mathbf{b}_o - \eta \cdot d\mathbf{b}_o \\
    & \mathbf{W}_h \leftarrow \mathbf{W}_h - \eta \cdot d\mathbf{W}_h \\
    & \mathbf{b}_h \leftarrow \mathbf{b}_h - \eta \cdot d\mathbf{b}_h
\end{aligned}
$$

At each training step (or epoch), the network computes predictions through forward propagation, measures the loss, computes all gradients through backward propagation and updates the parameters using these equations.

Over time, this process gradually reduces the loss and improves accuracy.