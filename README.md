Repo for COSE474(01) - Deep Learning @KoreaUniversity

## Dependencies
- PyTorch
- Tensorflow@1.13.1
- matplotlib
- numpy@1.16.6
- SciPy@1.1.0

```python
# 에러 무시
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
```

## Assignments
 - [x] Perceptron
 - [x] Multi-Layer Perceptron
   - 4 Layer
   - Weight initializer(He, Xavier)
   - Dropout
   - Weight decay
   - Early stopping epoch
   - Analyzing models with TensorBoard
 - [x] LeNet
    - Convolution, pooling, FC layers
    - Filter initialization
    - Learning rate decay
    - Test accuracy(final) for CIFAR-10 : 73.01%
 - [x] ResNet
    - Residual learning
    - Batch normalization
    - Test accuracy(final) for CIFAR-10 : 91.22%
    - Inference on my cat : correct
