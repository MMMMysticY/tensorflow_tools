- [tensorflow bugs](#tensorflow-bugs)
  - [tf.float32_ref tf.float32](#tffloat32_ref-tffloat32)
    - [报错信息](#报错信息)
    - [问题描述](#问题描述)
    - [解决方法](#解决方法)
  - [使用未初始化变量](#使用未初始化变量)
    - [报错信息](#报错信息-1)
    - [问题描述](#问题描述-1)
    - [解决方法](#解决方法-1)
# tensorflow bugs
## tf.float32_ref tf.float32
### 报错信息
```
Tensor conversion requested dtype float32_ref for Tensor with dtype float32  
```
### 问题描述  
tf1.15版本 keras风格模型  
keras模型可以推理，但是在训练的时候，优化器部分报错，使用的是tf.keras.optimizers.xxx。  
由于模型可以推理，问题出在优化器上。  
### 解决方法
tf.keras.optimizers类的优化器可能有些问题，使用tf.train.xxx进行解决。  

## 使用未初始化变量
### 报错信息
```
tensorflow.python.framework.errors_impl.FailedPreconditionError: Attempting to use uninitialized value var 
```
### 问题描述
tf1.15版本 keras风格模型
keras模型可以推理，在model.fit函数中，报错使用未初始化变量  
### 解决方法
```python
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    history = model.fit(x=[all_input_ids, all_token_type_ids, all_image_embeddings, all_masked_lm_positions],
                        y=all_masked_lm_ids,
                        batch_size=32, epochs=1, validation_split=0.2, validation_freq=1, callbacks=[cp_callback])
```
使用sess.run(tf.global_variables_initializer())和sess.run(tf.local_variables_initializer())  
