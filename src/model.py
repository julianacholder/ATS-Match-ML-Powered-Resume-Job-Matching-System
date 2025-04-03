import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import regularizers
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_keras_model(model_path="models/best_model.keras"):
    """
    Load a pre-trained Keras model
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Loaded Keras model
    """
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Creating a placeholder model instead.")
        return create_placeholder_model()

def create_placeholder_model(input_shape=1000):
    """
    Create a placeholder model when the real model cannot be loaded
    
    Args:
        input_shape: Input dimension for the model
        
    Returns:
        A simple Keras model
    """
    model = Sequential([
        Input(shape=(input_shape,)),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dropout(0.4),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=SGD(learning_rate=0.01, momentum=0.9),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def define_model(input_shape, 
                 optimization='sgd', 
                 regularization_rate=0.005, 
                 early_stopping=True, 
                 dropout_rate=0.4, 
                 learning_rate=0.01):
    """
    Define a neural network model with specified hyperparameters
    
    Args:
        input_shape: Input dimension for the model
        optimization: Optimizer to use ('adam', 'sgd', or 'rmsprop')
        regularization_rate: L2 regularization rate
        early_stopping: Whether to use early stopping
        dropout_rate: Dropout rate for regularization
        learning_rate: Learning rate for optimization
        
    Returns:
        Compiled Keras model
    """
    model = Sequential()
    reg = regularizers.l2(regularization_rate) if regularization_rate else None
    
    # Input layer
    model.add(Input(shape=(input_shape,)))
    
    # First hidden layer
    model.add(Dense(128, activation='relu', kernel_regularizer=reg))
    
    # Optional dropout
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))
    
    # Second hidden layer
    model.add(Dense(64, activation='relu', kernel_regularizer=reg))
    
    # Third hidden layer
    model.add(Dense(32, activation='relu', kernel_regularizer=reg))
    
    # Optional dropout
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))
    
    # Output layer
    model.add(Dense(1, activation='sigmoid'))
    
    # Optimizer selection
    if optimization is None or optimization.lower() == 'adam':
        opt = Adam(learning_rate=learning_rate)
    elif optimization.lower() == 'rmsprop':
        opt = RMSprop(learning_rate=learning_rate)
    elif optimization.lower() == 'sgd':
        opt = SGD(learning_rate=learning_rate, momentum=0.9)
    else:
        opt = Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=opt,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    """
    Train the model with early stopping and learning rate reduction
    
    Args:
        model: Compiled Keras model
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        epochs: Maximum number of epochs
        batch_size: Batch size for training
        
    Returns:
        History object from training
    """
    # Define callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.7,
        patience=3,
        min_lr=1e-4
    )
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, lr_scheduler],
        verbose=1
    )
    
    return history

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model and return performance metrics

    Args:
        model: Keras model to evaluate
        X_test: Test features
        y_test: Test labels

    Returns:
        Dictionary containing evaluation metrics
    """
    # Get predictions
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype("int32")

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Model evaluation
    loss, keras_accuracy = model.evaluate(X_test, y_test, verbose=0)

    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'loss': loss,
        'keras_accuracy': keras_accuracy 
    }

def save_model(model, path="models/retrained_model.keras"):
    """
    Save the model to a file
    
    Args:
        model: Keras model to save
        path: Path where the model will be saved
        
    Returns:
        True if successful
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save the model
    model.save(path)
    
    return True

def plot_training_history(history, model_name='Model'):
    """
    Plot training and validation loss and accuracy curves
    
    Args:
        history: History object from model training
        model_name: Name to display in plot titles
    """
    plt.figure(figsize=(12, 5))
    
    # Loss Subplot
    plt.subplot(1, 2, 1)
    epochs_range = range(1, len(history.history['loss']) + 1)
    plt.plot(epochs_range, history.history['loss'], 'bo-', label='Training Loss')
    plt.plot(epochs_range, history.history['val_loss'], 'ro-', label='Validation Loss')
    plt.title(f'{model_name} - Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy Subplot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history.history['accuracy'], 'bo-', label='Training Accuracy')
    plt.plot(epochs_range, history.history['val_accuracy'], 'ro-', label='Validation Accuracy')
    plt.title(f'{model_name} - Accuracy Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix'):
    """
    Plot a confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        title: Title for the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    labels = ['Not Relevant', 'Relevant']
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title(title)
    plt.show()
    
    return cm