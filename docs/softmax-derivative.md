# Building a Neural Network from Scratch

## The Math

### Derivative of the Softmax Function

To understand how to [derivate the softmax function](https://www.youtube.com/watch?v=09c7bkxpv9I), we'll consider a case were its input vector has only 3 elements: $\mathbf{Z} = (z_1, z_2, z_3)$. Our simplified softmax, for one of those elements would look like this:

$$
    s(z_1) = \frac{e^{z_1}}{\Sigma}
$$

Where we defined $\Sigma$ to make the expression shorter:

$$
    \Sigma = e^{z_1} + e^{z_2} + e^{z_3}
$$

To get the derivative of this expresion with regard to $z_1$ we'll apply the rule of the derivative of fractions. Given a function $h(x) = \frac{f(x)}{g(x)}$, its derivative with respect to $x$ is:

$$
    \frac{\partial h(x)}{\partial x} = \frac{f'(x)g(x) - g'(x)f(x)}{g(x)^2}
$$

If we apply this to the derivative of the softmax, we get:

```math
\begin{aligned}

    & \frac{\partial s(z_1)}{\partial z_1} = \frac{e^{z_1}\Sigma - e^{2 z_1}}{\Sigma^2} = s(z_1) (1 - s(z_1)) \\

    \\

    & \frac{\partial s(z_2)}{\partial z_1} = \frac{- e^{z_1} e^{z_2}}{\Sigma^2} = - s(z_1) s(z_2)

\end{aligned}
```

Now, we won't be as surprised when we see that this generalizes to:

```math
    \frac{\partial s(z_i)}{\partial z_j} =

    \begin{cases}
        s(z_i)(1 - s(z_i)) & \text{if } i = j \\
        - s(z_i) s(z_j) & \text{if } i \neq j
    \end{cases}
```

Which is usually simplified by introducing a [Kronecker's delta function](https://en.wikipedia.org/wiki/Kronecker_delta):

$$
    \frac{\partial s(z_i)}{\partial z_j} = s(z_i) (\delta_{ij} - s(z_j))
$$