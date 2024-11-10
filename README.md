This repository is part of a group project for the module IE4483 - Artificial Intelligence & Data Mining at Nanyang Technological University (NTU). The objective of the project is to develop a machine learning-based classifier designed to analyze and classify the sentiment of product reviews. Through this project, we aim to explore various data mining techniques and AI models to accurately predict whether product reviews are positive, negative, or neutral, ultimately contributing to the broader field of sentiment analysis and natural language processing.

### Required Libraries:

1. **`pandas`**: For data manipulation and analysis.
2. **`scikit-learn`**: For machine learning tools such as `train_test_split`.
3. **`transformers`**: For using pre-trained transformer models like BERT and tools such as `BertTokenizer` and `AdamW`.
4. **`torch`**: The core library for building and training deep learning models, specifically with PyTorch.
5. **`numpy`**: A foundational library for numerical computing (often required by PyTorch and scikit-learn).
6. **`tqdm`**: For progress bars (optional but commonly used with training loops).

### Installation Steps for Windows:

To install the necessary libraries, open the command prompt (CMD) and use `pip`, Python's package manager. Here's how you can install each module:

#### 1. **Install `pandas`**:
```bash
pip install pandas
```

#### 2. **Install `scikit-learn`**:
```bash
pip install scikit-learn
```

#### 3. **Install `transformers` (for BERT)**:
```bash
pip install transformers
```

#### 4. **Install `torch` (PyTorch)**:
- The installation for PyTorch may depend on whether you're using a CPU or a GPU. To install the correct version of PyTorch, you can follow the instructions on the official website [PyTorch Installation](https://pytorch.org/get-started/locally/).
- Here's a basic command for installing the CPU version:
```bash
pip install torch
```
- For GPU support (CUDA), you would use:
```bash
pip install torch torchvision torchaudio
```

Ensure that you install the version of PyTorch that matches your system's CUDA (GPU) version, if applicable. You can check the system specifications and choose the right version on the PyTorch website.

#### 5. **Install `numpy`**:
```bash
pip install numpy
```

#### 6. **(Optional) Install `tqdm`** (progress bar for loops, commonly used in training loops):
```bash
pip install tqdm
```

### Summary of Commands to Install All Dependencies:

You can run these commands one after the other to install everything you need:

```bash
pip install pandas
pip install scikit-learn
pip install transformers
pip install torch
pip install numpy
pip install tqdm
```

### Verifying Installation:
Once you've installed these libraries, you can verify the installation by opening a Python interpreter or running the code. For instance:

```python
import pandas as pd
import torch
from transformers import BertTokenizer
print(torch.__version__)  # Should print the installed PyTorch version
```

If the libraries are correctly installed, there should be no errors when you run this.

Let me know if you encounter any issues!