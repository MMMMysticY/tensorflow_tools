# tensorflow tools
tf.keras相关的工具  
- [tensorflow tools](#tensorflow-tools)
  - [keras](#keras)
    - [Layer类](#layer类)
      - [weights属性](#weights属性)
      - [output属性](#output属性)
      - [get_weights / set_weights](#get_weights--set_weights)
      - [trainable_weights / non_trainable_weights / trainable_variables / non_trainable_variables](#trainable_weights--non_trainable_weights--trainable_variables--non_trainable_variables)
    - [Model类](#model类)
      - [layers / get_layer()](#layers--get_layer)
      - [Model对象构建模型](#model对象构建模型)
      - [build / summary](#build--summary)
      - [trainable / training](#trainable--training)
    - [callbacks](#callbacks)
      - [ModelCheckpoint](#modelcheckpoint)
      - [TensorBoard](#tensorboard)
## keras


### Layer类
layer对象是基类tf.keras.layers.Layer的实例对象  
tensorflow的经典预设layer可以从tf.keras.layers.xxx获取  

#### weights属性
Layer对象的weights属性可以获取所有层的变量和权重值  
通过name和shape属性可以获得每个层具体的名称和维度  
**获取Layer对象的所有参数名称和维度：**  
```python
for weight in layer.weights:
    print(weight.name, weight.shape)
```

#### output属性
output属性可以得到Layer对象的output属性 在取模型子层时十分有用  

#### get_weights / set_weights
layer对象的get_weights方法可以得到Layer对象的所有参数 参数使用list结构存储 每个位置都是一个numpy矩阵 代表每个变量的具体取值  
layer对象的set_weights方法可以对Layer对象的参数进行赋值 使用list存储需要赋值的参数 所有的维度必须和load_weights的大小相同  
**模型导入其他模型参数样例：**  
```python
attention_layer = model.get_layer(index=0)
attention_layer_weights = attention_layer.get_weights()
# 获取model的第一层 并且获得所有参数
bert_attention_layer = bert_model.get_layer(name='Transformer-1-MultiHeadSelfAttention')
bert_attention_layer_weights = bert_attention_layer.get_weights()
# 获取bert的attention层 并获得所有参数
assert len(attention_layer_weights) == len(bert_attention_layer_weights)
# 二者同为list对象 len()相同代表变量个数相同
for i in range(len(attention_layer_weights)):
    print(attention_layer_weights[i].shape == bert_attention_layer_weights[i].shape)
# 每个位置上的shape都相同
attention_layer.set_weights(bert_attention_layer_weights)
# 使用bert参数进行初始化
```
*小tips：如果model1的一层对应model2分开的两层 直接把model2的两层的list进行相加即可set_weights*  

#### trainable_weights / non_trainable_weights / trainable_variables / non_trainable_variables
trainable_weights和trainable_variables都返回Layer对象的可训练可更新的参数情况  
non_trainable_weights和non_trainable_variables返回Layer对象不可更新(Frozen)的参数的情况  



### Model类
#### layers / get_layer()
layers可以获取Model对象的所有层对象 get_layer可以按照名称或序号进行获得特定层对象 注意layer对象为(tf.keras.layer.Layer)  

#### Model对象构建模型
使用Model对象可以直接构建keras模型 参数为inputs和outputs 其中inputs为一个初始的tensor outputs为inputs经过一些layer或者计算后的结果tensor  
**使用Model对象直接构建keras模型：**  
```python
inputs = tf.keras.Input(shape=(3,))
x = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs)
outputs = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
```

**那么结合Model构建模型、get_layer方法和Layer对象的output属性 就可以获得模型的顺序子模型**  
```python
model = MyModel()
res1 = model.predict(input_)
sub_model = Model(inputs=model.input, outputs=model.get_layer(2).output)
# 使用input和output对象进行Model构建
res2 = model.predict(input_)
```

#### build / summary
summary方法可以得到模型的结构和参数等信息 **但是模型必须进行一次输入计算后或者build后 才可以输出summary**  
build方法相当于告诉模型input的维度 从而可以构建模型图  
build方法的参数是input_shape 在不确定的维度可以使用None进行表示 如[None, None, 768]可以代表[batch, seq_len, dim]维度的输入  

#### trainable / training
trainable是Model对象的属性 training是Model对象call方法的参数  
**trainable属性确定了Model对象是否进行更新 用其确定模型参数是否进行frozen**  
**training参数确定了模型是训练状态还是推理状态 直接影响如dropout等方法是否起效**  



### callbacks
callbacks是在keras模型fit(训练)的过程中，在某个特定位置进行操作的一系列方法。  
一般放置在model.fit()方法的callbacks的位置上 (注意，即便只有一个callback，在callbacks的位置上也需要用list)  
#### ModelCheckpoint
ModelCheckpoint对象进行模型的保存，需要为模型指定filepath即保存的文件夹位置，monitor参数选择要监控的变量，save_best_only和save_weights_only顾名思义  
#### TensorBoard
TensorBoard是对训练过程进行可视化的callback，使用tensorboard命令可以进行训练过程可视化。  
需要对TensorBoard对象指定log_path即log保存位置，histogram_freq直方图频率。  
