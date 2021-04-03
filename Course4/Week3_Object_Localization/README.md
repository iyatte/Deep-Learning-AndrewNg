Course 4-Week 2
=========

## 1 Classic networks

### 1.1 LeNet-5
It's for grayscale images, so the dimension of the picuture is $32*32*1$
![LeNet-5](./pic/LeNet-5.jpg)

### 1.2 AlexNet
![AlexNet](./pic/alexnet.jpg)

### 1.3 VGGNet
VGG-16, l6 means covn+FC=16
![VGG-16](./pic/vgg16.jpg)

## 2 ResNets

### 2.1 Basic idea
![Resnets](./pic/resnet1.png)

The red line is skip connection, establish a connection between $a^{[l+2]}$ and $a^{[l]}$.
$$z^{[l+1]} = W^{[l+1]}*a^{[l]}+b^{[l+1]}$$
$$a^{[l+1]} = g(z^{[l+1]})$$
$$z^{[l+2]} = W^{[l+2]}*a^{[l+1]}+b^{[l+2]}$$
$$a^{[l+2]} = g(z^{[l+2]} + a^{[l]})$$

### 2.2 Why works
Compared to the Plain Network, the Residual Network is able to train a deeper layer of the neural network, effectively avoiding gradient disappearance and gradient explosion.   

Because:  
$$a^{[l+2]} = g(z^{[l+2]} + a^{[l]}) = g(W^{[l+2]}*a^{[l+1]}+b^{[l+2]} + a^{[l]})$$

When:
$W^{[l+2]}\approx0$ and $b^{[l+2]}\approx0$, 
Then:
$a^{[l+2]} = a^{[l]}$

When the dim of $z^{[l+2]}$ and  $a^{[l]}$ are different, we can add a parameter $W_s$:

$$a^{[l+2]} = g(z^{[l+2]} + W_s*a^{[l]})$$


It's a ResNets in CNN as following:
![ResNets](./pic/resnets_cnn.png)

* Usually use same mode

## 3  Inception Network Motivation
* Single:
![Inception](./pic/Inception_single.png)

* Inception Network
![Inception](./pic/Inception.png)

