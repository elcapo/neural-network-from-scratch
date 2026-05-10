# Building a Neural Network from Scratch

## The Math

### Forward Propagation

#### Input Layer

If $m$ is the number of samples in a batch, then our input matrix $\mathbf{X}_i$ has one sample per column.

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
Softmax(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j} e^{z_j}}
$$

### Backward Propagation

Backward propagation computes gradients of the loss function with respect to all parameters and activations, enabling parameter updates via gradient descent.

In order to avoid having to carry a $\frac{1}{m}$ factor for all the derivatives below, we define the total loss as the average of the per-sample losses:

$$
J = \frac{1}{m} \sum_{k=1}^{m} \mathcal{L}_k
$$

#### Loss Function

For a **single sample** $k$, we use the cross-entropy function:

$$
\mathcal{L}_k = - \sum_{i} y_i^{(k)} \log(a_i^{(k)})
$$

where:
- $y_i^{(k)}$ is the ground truth label for sample $k$ (one-hot encoded)
- $a_i^{(k)}$ is the network's predicted probability for class $i$ (i.e., $\mathbf{A}_o[i, k]$)

**Why does this formula make sense?** If the true label for sample $k$ is class $c$, then $y_c^{(k)} = 1$ and $y_i^{(k)} = 0$ for all $i \neq c$. The loss simplifies to:

$$
\mathcal{L}_k = - log(a_c^{(k)})
$$

So the loss is simply the negative log-probability the network assigned to the correct class. When the network is confident and correct ($a_c^{(k)} \approx 1$), the loss is near zero. When the network is wrong or uncertain ($a_c^{(k)} \approx 0$), the loss explodes.

For a **batch** of $m$ samples, we sum over all samples (and divide by $m$ at the $J$ level):

$$
\mathcal{L} = \sum_{k=1}^{m} \mathcal{L}_k = - \sum_{k} \sum_{i} y_i^{(k)} log(a_i^{(k)})
$$

In matrix form, this can be written as:

$$
\mathcal{L} = - \sum_{i} \sum_{k} y_i^{(k)} log(a_i^{(k)}) = - tr(\mathbf{Y}^T log(\mathbf{A}_o))
$$

The trace arises because $tr(\mathbf{Y}^T \mathbf{M}) = \sum_{i,k} Y_{k,i} M_{k,i}$, which gives exactly $\sum_i \sum_k y_i^{(k)} m_i^{(k)}$.

#### Gradient of the Loss with Respect to $\mathbf{A}_o$

We need $\frac{\partial \mathcal{L}}{\partial \mathbf{A}_o}$, which tells us how sensitive the loss is to each element of the output activations. Since $\mathcal{L} = - \sum_i \sum_k y_i^{(k)} log(a_i^{(k)})$, and the logarithm is element-wise, each $a_i^{(k)}$ only appears in the term $- y_i^{(k)} log(a_i^{(k)})$.

The derivative of a single term $-y \log(a)$ with respect to $a$ is:

$$
\frac{\partial}{\partial a} [-y \log(a)] = -y \cdot \frac{1}{a}
$$

Therefore, the gradient with respect to the entire matrix is:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{A}_o} = - \mathbf{Y} \oslash \mathbf{A}_o
$$

where $\oslash$ is element-wise division.

> **Note:** This derivation assumes **one-hot encoded** labels. If $\mathbf{Y}$ contained soft probability labels instead, the formula would still technically hold, but the intuition would change: each row $i$ would contribute a penalty proportional to how far the prediction $a_i$ is from the target $y_i$.

#### Derivative of the Softmax Function

Before we can compute $\frac{\partial \mathcal{L}}{\partial \mathbf{Z}_o}$, we need $\frac{\partial \mathbf{A}_o}{\partial \mathbf{Z}_o}$. Recall from [the softmax derivative document](softmax-derivative.md) that for a single sample:

$$
\frac{\partial a_j}{\partial z_i} = a_j (\delta_{ij} - a_i)
$$

where $\delta_{ij}$ is the Kronecker delta ($\delta_{ij} = 1$ if $i=j$, else $0$).

This result is key. There are two cases:
- **Diagonal ($i = j$):** $\frac{\partial a_i}{\partial z_i} = a_i(1 - a_i)$
- **Off-diagonal ($i \neq j$):** $\frac{\partial a_j}{\partial z_i} = -a_j a_i$

#### Gradient of the Loss with Respect to $\mathbf{Z}_o$

We now want $\frac{\partial \mathcal{L}}{\partial z_i}$, where $z_i$ is the pre-activation for class $i$ for a single sample. The chain rule tells us to sum over all output classes:

$$
\frac{\partial \mathcal{L}}{\partial z_i} = \sum_{j} \frac{\partial \mathcal{L}}{\partial a_j} \frac{\partial a_j}{\partial z_i}
$$

We already know both factors:
- $\frac{\partial \mathcal{L}}{\partial a_j} = - \frac{y_j}{a_j}$
- $\frac{\partial a_j}{\partial z_i} = a_j (\delta_{ij} - a_i)$

Substituting:

$$
\frac{\partial \mathcal{L}}{\partial z_i} = \sum_{j} \left( -\frac{y_j}{a_j} \right) \left( a_j (\delta_{ij} - a_i) \right)
$$

The $a_j$ terms cancel:

$$
\frac{\partial \mathcal{L}}{\partial z_i} = \sum_{j} -y_j (\delta_{ij} - a_i)
$$

Distributing the sum:

$$
\frac{\partial \mathcal{L}}{\partial z_i} = -\left( \sum_{j} y_j \delta_{ij} - \sum_{j} y_j a_i \right)
$$

Now we analyze each term:

**First term:** $\sum_j y_j \delta_{ij}$ — this picks out only the term where $j = i$, so it equals $y_i$.

**Second term:** $\sum_j y_j a_i = a_i \sum_j y_j$ — the factor $a_i$ is constant with respect to $j$. Since $\mathbf{Y}$ is one-hot encoded, $\sum_j y_j = 1$. Therefore this term simplifies to $a_i$.

Putting it together:

$$
\frac{\partial \mathcal{L}}{\partial z_i} = -(y_i - a_i) = a_i - y_i
$$

In matrix form for the entire batch:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{Z}_o} = \mathbf{A}_o - \mathbf{Y}
$$

> **Key insight:** The gradient $\mathbf{A}_o - \mathbf{Y}$ has a beautiful intuition: for each sample and each class, the gradient tells us exactly how much **too high** ($a_i > y_i$) or **too low** ($a_i < y_i$) our prediction was.

#### Output Layer Parameters

We apply the chain rule to relate $\frac{\partial \mathcal{L}}{\partial \mathbf{Z}_o}$ to the parameters.

Recall: $\mathbf{Z}_o = \mathbf{W}_o \mathbf{A}_h + \mathbf{b}_o$

For the **weights**:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}_o} = \frac{\partial \mathcal{L}}{\partial \mathbf{Z}_o} \frac{\partial \mathbf{Z}_o}{\partial \mathbf{W}_o}
$$

We need $\frac{\partial \mathbf{Z}_o}{\partial \mathbf{W}_o}$. The element $(\mathbf{Z}_o)_{ij}$ depends on $(\mathbf{W}_o)_{ik}$ through $(\mathbf{W}_o)_{ik} \cdot (\mathbf{A}_h)_{kj}$, so the partial derivative is simply $(\mathbf{A}_h)^T_{jk} = (\mathbf{A}_h^T)_{kj}$.

Using the standard matrix derivative result:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}_o} = \frac{\partial \mathcal{L}}{\partial \mathbf{Z}_o} \mathbf{A}_h^T
$$

This gives a matrix of shape $(10, 10)$, matching $\mathbf{W}_o$.

For the **biases**:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{b}_o} = \frac{\partial \mathcal{L}}{\partial \mathbf{Z}_o} \frac{\partial \mathbf{Z}_o}{\partial \mathbf{b}_o}
$$

Since $\mathbf{Z}_o = \mathbf{W}_o \mathbf{A}_h + \mathbf{b}_o$ and $\mathbf{b}_o$ is broadcast to every column, the derivative of $\mathbf{Z}_o$ with respect to $\mathbf{b}_o$ is a matrix of all ones. Summing over samples (columns) to get the gradient for the bias:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{b}_o} = \frac{\partial \mathcal{L}}{\partial \mathbf{Z}_o} \mathbf{1}
$$

where $\mathbf{1}$ is a column vector of ones. In simpler terms, this is the **sum of each row** of $\frac{\partial \mathcal{L}}{\partial \mathbf{Z}_o}$ (summing over all samples $m$).

#### Hidden Layer Activation

To backpropagate to the hidden layer, we apply the chain rule:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{A}_h} = \frac{\partial \mathcal{L}}{\partial \mathbf{Z}_o} \frac{\partial \mathbf{Z}_o}{\partial \mathbf{A}_h}
$$

Since $\mathbf{Z}_o = \mathbf{W}_o \mathbf{A}_h + \mathbf{b}_o$, the derivative $\frac{\partial \mathbf{Z}_o}{\partial \mathbf{A}_h} = \mathbf{W}_o^T$.

Therefore:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{A}_h} = \mathbf{W}_o^T \frac{\partial \mathcal{L}}{\partial \mathbf{Z}_o}
$$

#### Hidden Layer Pre-Activation (ReLU Derivative)

We now need $\frac{\partial \mathcal{L}}{\partial \mathbf{Z}_h}$. Since the hidden layer uses ReLU, we apply the chain rule element-wise:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{Z}_h} = \frac{\partial \mathcal{L}}{\partial \mathbf{A}_h} \odot \frac{\partial}{\partial \mathbf{Z}_h} ReLU(\mathbf{Z}_h)
$$

where $\odot$ is the element-wise (Hadamard) product.

The derivative of ReLU is:

```math
\frac{\partial}{\partial z} ReLU(z) =
\begin{cases}
    0 & \text{if } z <= 0 \\
    1 & \text{if } z > 0
\end{cases}
```

In matrix form, this is often written as $\mathbf{1}_{\mathbf{Z}_h > 0}$ — a mask matrix that is 1 where the pre-activation was positive and 0 elsewhere.

The key insight: **only neurons that were active during forward propagation** (where $z > 0$) can contribute to the loss gradient during backpropagation. Neurons that were "dead" ($z \leq 0$) receive zero gradient.

#### Hidden Layer Parameters

With $\frac{\partial \mathcal{L}}{\partial \mathbf{Z}_h}$ computed, we apply the chain rule once more.

Recall: $\mathbf{Z}_h = \mathbf{W}_h \mathbf{X}_i + \mathbf{b}_h$

For the **weights**:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}_h} = \frac{\partial \mathcal{L}}{\partial \mathbf{Z}_h} \frac{\partial \mathbf{Z}_h}{\partial \mathbf{W}_h}
$$

Since $\mathbf{Z}_h = \mathbf{W}_h \mathbf{X}_i$, the derivative $\frac{\partial \mathbf{Z}_h}{\partial \mathbf{W}_h} = \mathbf{X}_i^T$. Therefore:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}_h} = \frac{\partial \mathcal{L}}{\partial \mathbf{Z}_h} \mathbf{X}_i^T
$$

However, recall that our total loss $J$ includes a $\frac{1}{m}$ factor: $J = \frac{\mathcal{L}}{m}$. The gradient with respect to $J$ is:

$$
\frac{\partial J}{\partial \mathbf{W}_h} = \frac{1}{m} \frac{\partial \mathcal{L}}{\partial \mathbf{W}_h} = \frac{1}{m} \frac{\partial \mathcal{L}}{\partial \mathbf{Z}_h} \mathbf{X}_i^T
$$

For the **biases**:

$$
\frac{\partial J}{\partial \mathbf{b}_h} = \frac{1}{m} \sum \frac{\partial \mathcal{L}}{\partial \mathbf{Z}_h}
$$

Here, the sum is taken **over all samples** (columns), i.e., summing each row across the $m$ columns, then dividing by $m$. This gives a vector of shape $(10, 1)$ matching $\mathbf{b}_h$.

### Gradient Descent

Once all gradients are computed, the final step of backpropagation is to update the network's parameters so that the loss decreases in the next iteration.

Each weight and bias matrix is adjusted in the opposite direction of its gradient, scaled by a small constant called the learning rate ($\eta$):

$$
\begin{aligned}
& \mathbf{W}_o \leftarrow \mathbf{W}_o - \eta \cdot \frac{\partial J}{\partial \mathbf{W}_o} \\
& \mathbf{b}_o \leftarrow \mathbf{b}_o - \eta \cdot \frac{\partial J}{\partial \mathbf{b}_o} \\
& \mathbf{W}_h \leftarrow \mathbf{W}_h - \eta \cdot \frac{\partial J}{\partial \mathbf{W}_h} \\
& \mathbf{b}_h \leftarrow \mathbf{b}_h - \eta \cdot \frac{\partial J}{\partial \mathbf{b}_h}
\end{aligned}
$$

At each training step (or epoch), the network computes predictions through forward propagation, measures the loss, computes all gradients through backward propagation and updates the parameters using these equations.

Over time, this process gradually reduces the loss and improves accuracy.

### References

To get a refresh about how these things are calculated, check:

* [Matrix Calculus (for Machine Learning and Beyond)](https://ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2023/mit18_s096iap23_lec_full.pdf) by Alan Edelman and Steven G. Johnson
* [The Matrix Calculus You Need For Deep Learning](https://arxiv.org/pdf/1802.01528) by Terence Parr and Jeremy Howard
