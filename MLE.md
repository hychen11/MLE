# 面试准备计划

1.13

练手一些代码

练手一些基本的transformer,self attention的代码

1.14

梳理一边ml+dl的cheat sheet

把cs231n的内容回顾！

然后手撕transformer+diffusion model的代码

1.15

回顾nerf+latent nerf+diffusion的项目代码

练手一些ml的代码

1.16

看看jd，然后rag这一块

搜广推算法了解一下

1.17

# Conception

### Weights网格参数

网格参数指的是 **模型的权重参数**，也就是在神经网络中需要通过训练数据进行优化的参数。W and B

### Hyperparameters超参数

超参数是模型中**不通过训练数据学习**，而是由用户提前指定的参数

学习率、网络层数、Dropout 比例、batch size, 激活函数, 正则化参数

### Initialization

#### Random Initialization

生成标准正态分布（均值为 0，标准差为 1）

```python
random.randn(inputs,outputs)
```



```python
import numpy as np
W = np.random.rand(n_inputs, n_outputs)  # 生成 [0, 1) 的均匀分布
```

避免所有权重为 0，导致神经元输出相同，梯度消失

#### Uniform Initialization

```python
limit = np.sqrt(6 / (n_inputs + n_outputs))  # 经典初始化范围
W = np.random.uniform(-limit, limit, (n_inputs, n_outputs))
```

#### Gaussian Initialization

```python
W = np.random.normal(0, 0.01, (n_inputs, n_outputs))  # 均值 0，标准差 0.01
```

#### He Initialization

ReLU

$Var(W)=\frac{2}{n_{input}}$

```python
W = np.random.normal(0, np.sqrt(2 / n_inputs), (n_inputs, n_outputs))
```

#### Glorot Initialization

Xavier ->Sigmoid 或 Tanh 

$Var(W)=\frac{2}{n_{input}+n_{output}}$

```python
limit = np.sqrt(6 / (n_inputs + n_outputs))
W = np.random.uniform(-limit, limit, (n_inputs, n_outputs))
```

### Activation Function

Sigmoid $\frac{1}{1+e^{-x}}$ 二分类问题

```python
f=lambfa x:1.0/(1.0+np.exp(-x))
x=np.random.randn(3,1)  # create a vector of 3*1
h1=f(np.dot(W1,x)+b1)
h2=f(np.dot(W2,h1)+b2)
out=np.dot(W3,h2)+b3
```

Leaky Relu max(0.1x,x)

Relu max(0,x)

tanh $\frac{e^x-e^{-x}}{e^x+e^{-x}}$

softmax $f(x_i)=\frac{e^{x_i}}{\Sigma_{j=1}^n e^{x_i}}$ 多分类问题



# 

# Supervised Learning

Regression continuous, like predict the house price (linear regression)

naive bayes 假设特征之间是条件独立的 (logistic regression)

P(y∣x1,x2,…,xn)=P(y)⋅P(x1,x2,…,xn∣y)/P(x1,x2,…,xn)

P(x1,x2,…,xn∣y)=P(x1∣y)⋅P(x2∣y)⋯P(xn∣y)

### Generative Model

Estimate P(x|y) then deduce P(y|x)  , P(y|x)=P(x|y)*P(y)/P(x)

学习的是**数据的概率分布**

### Loss function

#### MLE

MLE极大似然**Maximum Likelihood Estimation**

The likelihood of a model L(θ) given parameters θ is used to find the optimal parameters θ through likelihood maximization. We have: argmax L(θ)

#### MSE

MSE mean squared err **Linear Regression**

MSE=1/n*∑(y^i−yi)2

Linear Regression 它假设输入特征与目标变量之间的关系是线性的

#### MAE

(Mean Absolute Error) 

MAE=N1i=1∑N∣yi−y^i∣

#### Logistic loss

log(1+exp(−*y**z*))

#### Huber loss | Hinge loss

max(0,1−yz)

#### Cross-entropy

classification 多分类里用`nn.CrossEntropy`

有n个分类,然后每个分类有一个概率,Cross-Entropy 衡量了两个概率分布（预测分布和真实分布）之间的差异，最小化 Cross-Entropy 损失等价于最大化对真实类别的预测概率。

$H(p,q)=-\Sigma P(i)logQ(i)$

$H(p)=-\Sigma P(i)logP(i)$

$KL=\Sigma P(i)log\frac{P(i)}{Q(i)}=H(P,Q)-H(P)$

KL散度反映了两个分布之间的差异程度

$D_{KL}(P||Q)!=D_{KL}(Q||P)$

### Gradient descent

1. 计算损失函数对参数 www 的梯度（偏导数），即： ∇L(w)=∂L/∂w 
2. 根据梯度的方向更新参数，使得损失函数减小： w:=w−η∇L(w)

在训练数据较大时，计算整个数据集的梯度（称为 **Batch Gradient Descent**）会非常耗时，因为每次迭代都需要遍历所有数据。

### SGD (Stochastic Gradient Descent)

不使用整个数据集来计算梯度，而是每次只随机选取一个样本或一个小批量（Mini-batch）数据来更新参数

w:=w−η∇L(w;xi,yi)

(xi,yi)(x_i, y_i)(xi,yi) 是一个样本（或一个小批量的样本）

#### Momentum

引入“动量”项，使参数更新时考虑之前的更新方向，减少震荡。

v=bv-η∇L(w)

v=w+v

#### RMSProp

动态调整学习率，根据每个参数的梯度历史进行自适应缩放。

#### Adam

结合了 Momentum 和 RMSProp 的优点

#### Mini-batch SGD



# Linear model

### Linear Regression

Linear Regression use **MSE** mean squared error!

**Normal equations** $theta=(X^TX)^{-1}X^Ty$

### Classification and logistic regression



# Gradient Descent

### Batch Gradient Descent

批量梯度下降,使用整个训练集计算梯度。

### Stochastic Gradient Descent

每次仅使用一个样本计算梯度

# SoftMax

是一种激活函数

```python
def softmax(z):
#overfloat
	exp_z=np.exp(z-np.max(z))
	return  exp_z/np.sum(exp_z)
```

# Norm

正则化

L1、L2 范数用于权重的正则化，限制模型复杂度

```pyhon
np.linalg.norm(x,ord=1)
np.linalg.norm(x,ord=2)
```

标准化

将向量归一化为单位长度

#### Batch Normalization

**BN** 将整个小批量数据归一化,需要考虑batch

#### Group Normalization

**GN** 将特征分为多个组，分别归一化

也就是不分数据,而分features

```python
import torch
import torch.nn as nn

# 输入张量：[batch_size, channels, height, width]
x = torch.rand(8, 32, 64, 64)

# 使用 GroupNorm
group_norm = nn.GroupNorm(num_groups=8, num_channels=32)  # 分为 8 组
output = group_norm(x)
print(output.shape)  # 输出形状与输入相同
```



# Backpropagation

### sigmoid

$σ′(z)=σ(z)⋅(1−σ(z))$



# Self-Attention

https://lilianweng.github.io/posts/2018-06-24-attention/

https://www.bilibili.com/video/BV1cV411X7XN?spm_id_from=333.788.videopod.episodes&vd_source=dff3ad76ed17ff6ff5fee17de98f73c5

word Embedding use vector to present word 

frame 25ms voice into vector, each time move 10ms, so 1s will have 100 frames

One-hot Embedding 就是N个大小集合用N大小的vector表示

#### sentiment analysis

多个word vector输入，单个输出 positive or negative

#### seq2seq

#### sequence Labeling

需要一个window size大小的input输入FC，的到label

#### self-attention

![](./MLE/1.png)

一系列vector a1,a2,a3,a4 -> self-attention -> b1,b2,b3,b4

step 1: find the relevant vectors in a sequence (a1,a4 relevant $\alpha$)

计算$\alpha$ 相关性的数值

![](./MLE/2.png)

![](./MLE/3.png)

softmax或者用group norm都可以

![](./MLE/4.png)

$\alpha$分数乘上v1 得到b1,这里b1,b2,b3,b4不是in sequence的,可以同时产生

![](./MLE/5.png)

把输入都拼起来,合成I一个矩阵,Wq,Wk,Wv都是网格参数,比如初始化为均值是0正态分布的矩阵。

```python
np.random.norm(0,0.01,(intput,output))
```

![](./MLE/6.png)

![](./MLE/7.png)

![](./MLE/8.png)

#### multi-head self-attention

![](./MLE/9.png)

#### positional Encoding

 这里不包括位置信息

可以给每个word vector ai 前加上ei (unique positional vector ei)

ei+ai

### truncated self-attention

Attention Matrix A' 大小是$L^2$ L就是input Length

只用一定大小的相邻的word vector

#### vs CNN

CNN is simplified self attention 只考虑receptive field里的, self attention考虑全部,或者说机器自己学需要的范围 

#### vs RNN Recurrent Neural Network

可以用self-attention取代RNN

![](./MLE/10.png)



#### GNN Graph Neural Network

![](./MLE/11.png)

# Transformer

#### seq2seq

 Input->Encoder->Decoder->output

#### Encoder

can use RNN or CNN 

In transform, it use self-attention

![](./MLE/12.png)

![](./MLE/13.png)

![](./MLE/14.png)

**BERT（Bidirectional Encoder Representations from Transformers）** 的架构主要基于 **Transformer 的 Encoder 部分**

#### Decoder

#### Cross Attention

![](./MLE/18.png)

#### Autoregressive

masked self-attention

![](./MLE/15.png)16

考虑bi只考虑bi及以前的,不考虑后面的

所以这里会有一个mask为上三角的矩阵

为什么要masked呢?就是一个一个产生,只考虑左边不考虑右边

但是我们不知道输出的长度

#### Non autoregressive model

![](./MLE/16.png)

#### cross attention

**Self-Attention**：序列内元素相互关注。

**Cross Attention**：一个序列（查询）关注另一个序列（键值对）

这里输出后就是minimize cross entropy

# Beam search

![](./MLE/17.png)

局部最优和全局最优



# ML Coding

### n次硬币向上概率

$T_n=T_{n−1}+1+0.5∗0+0.5∗T_n$

$T_n=2^{n+1}-2$

### Linear regression Norm

```python
x=np.array(x)
x_transpose=x.T
x_inv=np.linalg.inv(x_transpose.dot(x))
```

### Shuffle 

```python
import numpy as np

def shuffle_data(X, y, seed=None):
	if seed:
		np.random.seed(seed)
	
	idx=np.arange(X.shape[0])
	np.random.shuffle(idx)
	return X[idx],y[idx]
```

### KL Divergence Between Two Normal Distributions

#### KL Divergence

常用于**概率分布的对比分析**

$D_{KL}(P∥Q)=\Sigma _i P(i)log\frac{Q(i)}{P(i)}$ =H(P,Q)-H(P)

衡量分布 Q 近似分布 P 的信息损失

$ P \sim N(\mu_P, \sigma_P^2) $

$Q \sim N(\mu_Q, \sigma_Q^2)$
$$
D_{\text{KL}}(P \parallel Q) = \log \frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac{1}{2}
$$

#### Cross Entropy

常用作分类任务中的**损失函数，直接优化模型预测Q**

 $H(P,Q)=-\Sigma_i P(i)logQ(i)$

量分布 Q（模型预测分布）对分布 P（真实分布）的描述效率

```python
import numpy as np

def kl_divergence_normal(mu_p, sigma_p, mu_q, sigma_q):
	term1=np.log(sigma_q/sigma_p)
	term2=(sigma_p**2+(mu_p-mu_q)**2)/(2*sigma_q**2)
	return term1+term2-0.5

```

####  Single Neuron

```python
import math
import numpy as np
def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
	x=np.array(features)
	w=np.array(weights)
	b=np.array(bias)
	label=np.array(labels)
	y=x.dot(w)+b
	y=1/(1+np.exp(-y))
	mse=np.mean((y-label)**2)
	return np.round(y,4), np.round(mse,4)
```

### self-attention

```python
import numpy as np

def softmax(scores):
	exp_z=np.exp(scores-np.max(scores))
	return exp_z/np.sum(exp_z,axis=1,keepdims=True) #这个也容易错!只求和第二个dim!

def compute_qkv(X, W_q, W_k, W_v):
    Q = np.dot(X, W_q)
    K = np.dot(X, W_k)
    V = np.dot(X, W_v)
    return Q, K, V

def self_attention(Q, K, V):
	d_k=Q.shape[1]
	A=Q.dot(K.T)/np.sqrt(d_k) #这个遗漏了我!! np.sqrt(d_k)
	A=softmax(A)
	return A.dot(V)

```

### RNN forward

```python
import numpy as np

def rnn_forward(
    input_sequence: list[list[float]],
    initial_hidden_state: list[float],
    Wx: list[list[float]],
    Wh: list[list[float]],
    b: list[float]
) -> list[float]:
    # Initialize parameters and convert to numpy arrays
    h = np.array(initial_hidden_state)
    Wx = np.array(Wx)
    Wh = np.array(Wh)
    b = np.array(b)

    # Iterate through the input sequence
    for x in input_sequence:
        x = np.array(x)
        # Compute the hidden state using tanh activation
        h = np.tanh(np.dot(Wx, x) + np.dot(Wh, h) + b)

    # Final hidden state, rounded to 4 decimal places
    final_hidden_state = np.round(h, 4)
    return final_hidden_state.tolist()

```

### Single Neuron with Backpropagation

```python
import numpy as np
def sigmoid(x):
	return 1/(1+np.exp(-x))

def MSE(y,label):
	return np.mean((y-label)**2)

def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):  	# Your code here	 
	mse_values=[]	
	for epoch in range(epochs):
		y=features.dot(initial_weights)+initial_bias
		y_hat=sigmoid(y)
		mse=MSE(y_hat,labels)
		mse_values.append(np.round(mse,4))


		errors=y_hat-labels
		theta1=(2/len(labels))*np.dot(features.T,errors*y_hat*(1-y_hat))
		theta2=(2/len(labels))*np.sum(errors*y_hat*(1-y_hat))
		
		initial_weights-=learning_rate*theta1
		initial_bias-=learning_rate*theta2

	return np.round(initial_weights,4).tolist(),np.round(initial_bias,4), mse_values
```

### use torch

```python
import torch
import torch.nn as nn
features=torch.tensor([[],[]],detype=torch.float32)

def train_neuron_torch(
    features: torch.Tensor,
    labels: torch.Tensor,
    initial_weights: torch.Tensor,
    initial_bias: torch.Tensor,
    learning_rate: float,
    epochs: int
) -> (torch.Tensor, float, list[float]):	 
    weight=initial_weights.clone().requires_grad_(True)
    bias = initial_bias.clone().requires_grad_(True)
    mse_loss=nn.MSELoss()
    
	mse_values=[]	
	for epoch in range(epochs):
     	y = torch.matmul(features, weights) + bias
        y_hat = torch.sigmoid(y)
        loss = mse_loss(y_hat, labels)
		
        # Store the rounded loss value
        mse_values.append(round(loss.item(), 4))

        # Backward pass
        loss.backward()

        # Gradient descent step
        with torch.no_grad():
            weights -= learning_rate * weights.grad
            bias -= learning_rate * bias.grad

        # Zero the gradients for the next iteration
        weights.grad.zero_()
        bias.grad.zero_()

    return weights.detach().round(4), bias.item(), mse_values
```

# Interview

### DNN 过拟合怎么解决

增加data

正则化L1,L2和dropout来减少对个别Neural的依赖

简化模型

交叉验证Cross validation

引入噪声

### Regularization 

Prevent the model  from doing too well on training data

Why regularize?

- Express preferences over weights  
- Make the model simple so it works on test data 
- Improve optimization by adding curvature

### Dropout