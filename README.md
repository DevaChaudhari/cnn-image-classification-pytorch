# CNN Image Classification with PyTorch

Hands-on practice with convolutional neural networks in PyTorch: how images are represented as tensors, how `Conv2d` / `MaxPool2d` layers change tensor shapes, and how a CNN compares with a plain fully-connected network on a real image-classification task.

## Results

Six-class image classification (65×65 RGB images, 480-image held-out test set):

| Model | Architecture | Test accuracy |
|---|---|---|
| MLP baseline | Flatten → 3 linear layers (ReLU), SGD | **46.3%** |
| CNN | 2+ conv layers + max pooling + linear head, Adam | **62.5%** |

The CNN improves accuracy by about 16 percentage points over the MLP while using fewer parameters (530K vs 814K), because it learns local spatial features instead of treating every pixel independently.

## What's inside

| Notebook | Topics |
|---|---|
| [`cnn_fundamentals.ipynb`](cnn_fundamentals.ipynb) | Image structure (grayscale vs RGB channels), `torchvision` transforms, normalization, convolution, padding, stride, pooling, building CNNs, MNIST, `ImageFolder` datasets, weighted sampling for class imbalance |
| [`conv2d_shape_practice.ipynb`](conv2d_shape_practice.ipynb) | Step-by-step shape tracing through `Conv2d → MaxPool2d → Conv2d → ReLU → Flatten → Linear` on a toy tensor |
| [`image_classification_mlp_vs_cnn.ipynb`](image_classification_mlp_vs_cnn.ipynb) | End-to-end pipeline: `ImageFolder` loading, train/val/test split, DataLoaders, MLP baseline, CNN, loss curves, confusion matrices, classification reports |

Supporting code:

- [`handy_code/ModelBuilder.py`](handy_code/ModelBuilder.py): reusable training loop (train/validation steps, loss tracking and plotting, prediction, model checkpointing)
- [`handy_code/weighted_sampler.py`](handy_code/weighted_sampler.py): `WeightedRandomSampler` helper for imbalanced datasets
- [`cnn_utils.py`](cnn_utils.py): synthetic line-image generator and plotting helpers

## Skills practised

- Representing images as PyTorch tensors (`C × H × W`) and applying `torchvision` transforms
- Building models with `nn.Sequential`, `nn.Conv2d`, `nn.MaxPool2d`, `nn.Flatten`, `nn.Linear`
- Predicting output shapes for convolution, padding, stride and pooling
- Training with SGD and Adam, monitoring training vs validation loss for under/over-fitting
- Evaluating with accuracy, normalized confusion matrices and `classification_report`
- Reproducible experiments with seeded `random_split` and `torch.manual_seed`

## Data

- `train/`, `val/`: 6 classes from the Imagenette subset of ImageNet (English springer, chainsaw, church, garbage truck, golf ball, parachute); 2,400 training and 1,200 validation images. The validation set is split 60/40 into validation and test.
- `MNIST/`: handwritten digits dataset in raw format, used in the fundamentals notebook.
- `images/`: figures used inside the notebooks.

## Running it

```bash
pip install -r requirements.txt
jupyter notebook
```

Open any notebook from the repository root so the relative data paths resolve. Training runs on CPU; a CUDA GPU is used automatically if available.

## Credits

The fundamentals notebook and helper code are adapted from teaching material by Mariusz Bajger.
