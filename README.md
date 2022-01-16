# ConvNeXt-Tensorflow

Unofficial TF2 Keras implementation of [ConvNeXt](https://github.com/facebookresearch/ConvNeXt).

<div align="center">

** Unofficial TF2/Keras implementation of [ConvNeXt](https://github.com/facebookresearch/ConvNeXt). Serializable. **

</div>

* ⚡ Easy to deploy: Model is serializable & deployable to tensorflow-serving.

****

Functional approach to make i

```python
import tensorflow as tf
model = MobileV3Large.from_pretrained().cuda().eval()
model.predict(images)
```

## Examples
```python
import tensorflow as tf
from models.convnext_tf import create_model

randx = tf.random.uniform((10, 32, 32, 3))

model = build_model_functional(
    name="convnext_tiny",
    shape=(32, 32, 3),
    num_classes = 100
)
print(model.summary())

model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001),
    loss      = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False),
    metrics   = ['accuracy']
)

print(model(randx)) # (1, 100)
```

## Thanks
- TF2 with converted weights: https://github.com/bamps53/convnext-tf
- https://github.com/facebookresearch/ConvNeXt  
 
## Reference

```BibTeX
@Article{liu2021convnet,
  author  = {Zhuang Liu and Hanzi Mao and Chao-Yuan Wu and Christoph Feichtenhofer and Trevor Darrell and Saining Xie},
  title   = {A ConvNet for the 2020s},
  journal = {arXiv preprint arXiv:2201.03545},
  year    = {2022},
}
```

