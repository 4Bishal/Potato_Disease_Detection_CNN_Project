# Potato Disease Detection Using CNN

A prototype deep learning application for classifying potato leaf images into three disease categories using a Convolutional Neural Network (CNN).

## Project Overview

This project explores image-based potato disease classification with TensorFlow and Keras. It loads leaf photographs, preprocesses and augments the images, trains CNN architectures, evaluates predictions on a held-out test set, and saves trained `.keras` and TensorFlow SavedModel artifacts.

The prototype currently covers the data pipeline and model-development stage. Full-stack integration, including a production backend and frontend, is planned for a future phase.

## Project Status

**Initial deep learning prototype**

- CNN model development and experimentation are implemented.
- Training, validation, test evaluation, visualization, model export, and single-image inference are present in the Jupyter notebook.
- Backend API development and frontend integration are not implemented.

## Objectives

- Build a CNN-based image classifier for potato leaf diseases.
- Experiment with convolutional architectures and data augmentation.
- Apply training callbacks to reduce overfitting and preserve the best validation model.
- Evaluate model performance using test accuracy, classification reports, and confusion matrices.
- Explore test-time prediction and confidence-based inference.
- Develop practical experience with TensorFlow/Keras data pipelines and model deployment artifacts.

## Dataset

The dataset is organized by class in a directory structure supported by `tf.keras.utils.image_dataset_from_directory`:

```text
training/dataset/
├── Potato___Early_blight/
├── Potato___Late_blight/
└── Potato___healthy/
```

The implementation discovers exactly three classes:

1. `Potato___Early_blight`
2. `Potato___Late_blight`
3. `Potato___healthy`

The local directory contains 2,152 JPEG images across the three classes:

| Class | Image count |
|---|---:|
| Potato___Early_blight | 1,000 |
| Potato___Late_blight | 1,000 |
| Potato___healthy | 152 |

The code does not identify a dataset name or source, and no dataset license or provenance metadata is included. Obtain the dataset separately and place it at `training/dataset` with the three class directories above. Preserve the original license and usage terms for the dataset you use.

The original dataset directory is excluded by `.gitignore` because the repository does not need to track the image corpus.

## Dataset Pipeline

- Images are resized to `256 x 256` pixels.
- RGB input is represented by 3 channels.
- A batch size of `32` is used.
- Keras loads labels by directory name and maps class names in alphabetical directory order.
- Pixel values are rescaled from `[0, 255]` to `[0, 1]` by a preprocessing layer inside the model.
- The dataset is shuffled with seed `42`.
- Batches are split into:
  - 80% training
  - 10% validation
  - 10% test
- Training, validation, and test datasets are cached and prefetched.
- Training uses geometric augmentation:
  - Random horizontal and vertical flips
  - Random rotation up to 20%
  - Random zoom from -20% to +20% in the primary architecture experiment

The split is performed at the batch level by the custom `get_dataset_partitions_tf` function.

## CNN Architecture

The notebook contains two CNN experiments.

### Primary anti-overfitting CNN

This model is named `CNN_Anti_Overfit_Detector`:

1. Input: `256 x 256 x 3`
2. Resizing to `256 x 256`
3. Rescaling from `[0, 255]` to `[0, 1]`
4. Random flip, rotation, and zoom augmentation
5. `Conv2D(32, 3 x 3, ReLU)` → `MaxPooling2D(2 x 2)`
6. `Conv2D(64, 3 x 3, ReLU)` → `MaxPooling2D(2 x 2)`
7. `Conv2D(128, 3 x 3, ReLU)` → `MaxPooling2D(2 x 2)`
8. `Conv2D(128, 3 x 3, ReLU)` → `MaxPooling2D(2 x 2)`
9. `Conv2D(128, 3 x 3, padding="same", ReLU)` → `MaxPooling2D(2 x 2)`
10. Flatten
11. Dense layer with 64 units, ReLU, and L2 regularization (`0.01`)
12. Dropout (`0.2`)
13. Dense output with 3 units and Softmax

### Six-block experimental CNN

This model is named `CNN_6Block_Potato_Detector` and is the architecture saved as `model_2.keras` and exported under `models/model_2`:

1. Input: `256 x 256 x 3`
2. Resizing and rescaling
3. Random horizontal/vertical flip and 20% random rotation
4. Six convolution blocks using `Conv2D(64, 3 x 3, ReLU)` with `MaxPooling2D(2 x 2)` after each block
5. Flatten
6. Dense layer with 64 ReLU units
7. Dense output with 3 units and Softmax

The first convolution block uses 32 filters in this architecture; the subsequent five convolution blocks use 64 filters.

### Training Configuration

- Optimizer: Adam
- Learning rate: `0.001`
- Gradient clipping: `clipnorm=1.0`
- Loss: Sparse categorical cross-entropy
- Metric: Accuracy
- Maximum epochs: `30`
- ReduceLROnPlateau: validation-loss monitor, factor `0.5`, patience `3`, minimum learning rate `1e-6`
- EarlyStopping: validation-loss monitor, patience `7`, restore best weights
- ModelCheckpoint: saves the best model to `training/best_potato_model.keras`

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- Jupyter Notebook

## Project Structure

```text
potato_disease_project/
├── README.md
├── .gitignore
├── models/
│   ├── model_1/
│   │   ├── saved_model.pb
│   │   ├── fingerprint.pb
│   │   └── variables/
│   ├── model_2/
│   │   ├── saved_model.pb
│   │   ├── fingerprint.pb
│   │   └── variables/
│   └── model_2.keras
└── training/
    ├── train_model.ipynb
    ├── best_potato_model.keras
    ├── class_names.json
    ├── .ipynb_checkpoints/
    └── dataset/
        ├── Potato___Early_blight/
        ├── Potato___Late_blight/
        └── Potato___healthy/
```

## Installation and Setup

The notebook is run from the `training` directory because its relative paths use `./dataset` and `../models`.

```bash
git clone <repository-url>
cd potato_disease_project

python -m venv .venv
```

Activate the environment:

**Windows**

```powershell
.venv\Scripts\Activate.ps1
```

**macOS/Linux**

```bash
source .venv/bin/activate
```

Install the libraries imported by the notebook:

```bash
python -m pip install tensorflow matplotlib numpy seaborn scikit-learn jupyter
```

Place the dataset at `training/dataset` with its three class subdirectories before starting training.

## Model Training

Start Jupyter from the project root:

```bash
jupyter notebook
```

Open `training/train_model.ipynb`, then run the cells from the beginning. The notebook:

1. Loads images from `training/dataset`.
2. Creates training, validation, and test partitions.
3. Builds and trains the CNN experiments.
4. Plots accuracy and loss curves.
5. Evaluates the test set.
6. Prints a classification report and confusion matrix.
7. Exports the SavedModel directories and Keras models under `models/`.

The notebook saves the best checkpoint as `training/best_potato_model.keras`. The final saved full Keras model is `models/model_2.keras`.

## Prediction / Inference

Single-image inference is implemented in the notebook. The function loads `models/model_2.keras`, reads `training/class_names.json`, resizes an image to `256 x 256`, and runs the model.

Example use from the `training` directory:

```python
result = predict_potato_leaf(
    "./dataset/Potato___Early_blight/<image-file>.JPG"
)
print(result)
```

The code uses an 80% confidence threshold:

- Confidence below `80%` returns `status: "uncertain"` and recommends capturing a clearer, closer image.
- Confidence at or above `80%` returns `status: "confident"` and the predicted class.
- It also reports probabilities for all three classes.

Preprocessing and normalization are already included in the saved model, so inference does not manually divide RGB values by `255`.

The notebook also contains test-time augmentation code that compares horizontally flipped, vertically flipped, and rotated views, but it does not provide a separate deployment application or command-line interface.

## Current Progress

### Implemented

- Dataset discovery and image loading
- 256 x 256 image resizing
- `[0, 255]` to `[0, 1]` rescaling
- 80/10/10 train/validation/test batch split
- Dataset caching and prefetching
- Random image augmentation
- Two CNN architecture experiments
- Adam optimization and sparse categorical cross-entropy
- Learning-rate reduction and early stopping
- Best-model checkpointing
- Test accuracy evaluation
- Classification report and confusion matrix
- Accuracy/loss visualization
- Test-time augmentation experimentation
- SavedModel and `.keras` model exports
- Class-name mapping
- Confidence-thresholded single-image inference within the notebook

### Planned

- Backend API
- Frontend interface
- Real-time image upload
- Production prediction service

## Future Improvements

- Optimize model performance through hyperparameter tuning and additional experimentation.
- Experiment with deeper and more efficient CNN architectures.
- Apply transfer learning using suitable pretrained image models.
- Add stronger evaluation with per-class metrics, ROC or PR analysis, and held-out validation.
- Visualize training and model interpretability results.
- Improve augmentation using broader and domain-relevant transformations.
- Develop a backend API for model loading and inference.
- Integrate a frontend for image selection and prediction display.
- Support real-time image upload and end-to-end disease prediction.
- Package dependencies with a pinned `requirements.txt` or environment file.

## Learning Outcomes

This project explores:

- CNN construction with convolution, pooling, flattening, dense, dropout, and softmax layers
- Image resizing and normalization
- Training-set augmentation
- Train/validation/test splitting
- Sparse categorical classification
- Adam optimization and gradient clipping
- Overfitting control using regularization and callbacks
- Model evaluation and visualization
- TensorFlow SavedModel and Keras serialization
- Softmax probabilities and confidence-based inference

## Author

**Bishal Bhandari**
