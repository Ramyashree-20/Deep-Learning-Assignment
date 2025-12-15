👩‍🎓 Student Details

| Field            | Information                                  |
| ---------------- | -------------------------------------------- |
| **Name**         | Ramya Shree R                                |
| **Semester**     | 7th                                          |
| **USN**          | 1CD22AI046                                   |
| **Department**   | Artificial Intelligence and Machine Learning |
| **Subject**      | Deep Learning and Reinforcement Learning     |
| **Subject Code** | BAI701                                       |


📌 Assignment Summary

This repository contains the Deep Learning assignment programs implemented as part of the BAI701 course. Initially, basic versions of each program were provided. I have modified and enhanced each program to improve performance, readability, and learning outcomes.

Through these changes, I gained hands-on experience in:

Neural network architecture design

CNNs, RNNs, LSTMs, and Reinforcement Learning

Dataset preprocessing and normalization

Model evaluation and visualization

Understanding why architectural and training choices matter in deep learning

All programs were executed successfully after modification.

📂 Project Structure

DL_ASSIGNMENT/
│
├── airline-passengers.csv
├── AlexNet.py
├── Rnn.py
├── LSTM.py
├── Deepreinforcementlearning.py
├── TicTacToe.py
├── policy_p1
├── original_data.png
└── README.md

🧠 Program-wise Explanation of Changes

1️⃣ AlexNet.py – Convolutional Neural Network

🔹 Original Implementation

Basic AlexNet CNN architecture

Convolution + Pooling + Fully Connected layers

No normalization layers

Simple classification model

🔹 Changes Made

Added Batch Normalization after convolution layers

Changed padding to preserve spatial features

Improved training stability

Maintained original AlexNet structure

🔹 Why These Changes?

Batch normalization speeds up convergence

Reduces internal covariate shift

Improves accuracy and stability for deep CNNs

2️⃣ Rnn.py – Character-Level RNN Text Generation

🔹 Original Implementation

SimpleRNN with ReLU activation

High number of epochs

No regularization

Greedy character prediction

🔹 Changes Made

Changed activation from ReLU → tanh

Reduced number of RNN units

Added Dropout

Reduced epochs for small dataset

Improved text generation clarity

🔹 Why These Changes?

tanh is better suited for sequential data

Dropout reduces overfitting

Small dataset requires controlled training

Improves learning stability

3️⃣ LSTM.py – Airline Passenger Time Series Prediction

🔹 Original Implementation

Single LSTM layer (10 units)

Hardcoded dataset path

No validation during training

Basic evaluation using RMSE

🔹 Changes Made

Implemented Bidirectional LSTM

Used 3 stacked BiLSTM layers (64–64–32)

Added Dropout (0.2) after each layer

Automatic dataset download from GitHub

Increased time window to 12 months

Added EarlyStopping & ReduceLROnPlateau

Evaluated using RMSE and MAE

Added multiple high-quality plots

🔹 Why These Changes?

Bidirectional LSTM captures past and future context

Deeper architecture models seasonal patterns

Adaptive learning improves convergence

MAE gives more interpretable error values

4️⃣ Deepreinforcementlearning.py – Q-Learning on Graph

🔹 Original Implementation

Q-learning to find shortest path

Reward matrix and Q-matrix

Basic environment awareness

🔹 Changes Made

Simplified reward and Q-matrix initialization

Improved function naming and structure

Added consistent graph visualization

Modularized environment (police & drug traces)

Improved action selection logic

🔹 Why These Changes?

Cleaner structure improves understanding

Better modularization helps explain RL concepts

Makes the code easier for academic explanation

5️⃣ TicTacToe.py – Reinforcement Learning Game Agent

🔹 Original Implementation

Q-learning based Tic-Tac-Toe

Inconsistent player symbol initialization

Uneven reward handling

Long training time

🔹 Changes Made

Fixed player symbols (1 and -1)

Corrected reward distribution

Ensured Player 1 always starts

Reduced training rounds for faster execution

Improved code readability

🔹 Why These Changes?

Ensures correct state representation

Fair reward propagation improves learning

Faster execution while maintaining accuracy

✅ Result Summary

| Program Name                 | Execution Status        |
| ---------------------------- | ----------------------- |
| AlexNet.py                   | ✅ Executed Successfully |
| Rnn.py                       | ✅ Executed Successfully |
| LSTM.py                      | ✅ Executed Successfully |
| Deepreinforcementlearning.py | ✅ Executed Successfully |
| TicTacToe.py                 | ✅ Executed Successfully |


▶️ How to Run the Programs

🔹 Step 1: Clone or Download the Repository
git clone https://github.com/Ramyashree-20/Deep-Learning-Assignment.git
cd DL_ASSIGNMENT

Or download and extract the ZIP file.

🔹 Step 2: Create / Activate Python Environment (Recommended)
Using Conda:

conda create -n dl_env python=3.9
conda activate dl_env

🔹 Step 3: Install Required Libraries
pip install numpy pandas matplotlib tensorflow keras scikit-learn networkx

(Optional for plotting model architecture)

pip install pydot graphviz

🔹 Step 4: Run Individual Programs
Run any program using:

python filename.py

Examples:

python AlexNet.py
python Rnn.py
python LSTM.py
python Deepreinforcementlearning.py
python TicTacToe.py

📦 Libraries / Tools Used

| Library                | Purpose                            |
| ---------------------- | ---------------------------------- |
| **NumPy**              | Numerical computations             |
| **Pandas**             | Dataset handling                   |
| **Matplotlib**         | Data visualization                 |
| **TensorFlow / Keras** | Deep learning models               |
| **Scikit-learn**       | Scaling and evaluation metrics     |
| **NetworkX**           | Graph visualization (RL)           |
| **Math**               | Error calculations                 |
| **OS / urllib**        | Dataset download and file handling |

💻 System Requirements

Python 3.8 – 3.11

Minimum 4 GB RAM

GPU (optional, for faster training)

Windows / Linux / macOS

🏁 Conclusion

All programs were successfully executed after modification.
This assignment helped me gain strong practical understanding of Deep Learning architectures and Reinforcement Learning techniques. The process of modifying existing code improved my ability to analyze, 
optimize, and explain deep learning models effectively.
