Course 4-Week 1
=========

## 1 Convolutional 
when filter is f*f, padding is p, then
the dim after convolutional is:
$$[\frac{n+2p-f}{s} +1] * [\frac{n+2p-f}{s} +1]$$


$[...]$ means taking down the whole.  


## 2 One dim 
### 1.1 full mode
padding  $ p= f-1$

### 1.2 valid mode
p = 0, means no padding

### 1.3 same mode
The dim doesn't change, when s=1, the p should match condition:
$$p = \frac{f-1}{2}$$

