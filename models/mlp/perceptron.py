"""
Kompletny projekt Deep MLP (Multilayer Perceptron) 
Implementacja własnej sieci neuronowej bez użycia TensorFlow/PyTorch
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Union, Any
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time
import random
from abc import ABC, abstractmethod
import copy
from dataclasses import dataclass
import json
import random
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback


# FUNKCJE AKTYWACJI 
class ActivationFunction(ABC):
    """Abstrakcyjna klasa bazowa dla funkcji aktywacji"""
    
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Przejście w przód przez funkcję aktywacji"""
        pass
    
    @abstractmethod
    def backward(self, x: np.ndarray) -> np.ndarray:
        """Pochodna funkcji aktywacji"""
        pass
    
    @abstractmethod
    def __str__(self) -> str:
        """Reprezentacja tekstowa"""
        pass

class Softmax(ActivationFunction):
    """Funkcja aktywacji Softmax dla klasyfikacji wieloklasowej"""
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)
    
    def __str__(self) -> str:
        return "Softmax"

class Sigmoid(ActivationFunction):
    """Funkcja aktywacji Sigmoid"""
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        sig = self.forward(x)
        return sig * (1 - sig)
    
    def __str__(self) -> str:
        return "Sigmoid"


class ReLU(ActivationFunction):
    """Funkcja aktywacji ReLU"""
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)
    
    def __str__(self) -> str:
        return "ReLU"


class Tanh(ActivationFunction):
    """Funkcja aktywacji Tanh"""
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.clip(x, -500, 500)
        return np.tanh(x)
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        tanh_x = self.forward(x)
        return 1 - tanh_x ** 2
    
    def __str__(self) -> str:
        return "Tanh"


class LeakyReLU(ActivationFunction):
    """Funkcja aktywacji Leaky ReLU"""
    
    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, x, x * self.alpha)
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1, self.alpha)
    
    def __str__(self) -> str:
        return f"LeakyReLU(alpha={self.alpha})"


class ELU(ActivationFunction):
    """Funkcja aktywacji ELU (Exponential Linear Unit)"""
    
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, x, self.alpha * (np.exp(x) - 1))
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1, self.alpha * np.exp(x))
    
    def __str__(self) -> str:
        return f"ELU(alpha={self.alpha})"


class Linear(ActivationFunction):
    """Funkcja aktywacji liniowa (brak aktywacji)"""
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x
    
    def backward(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)
    
    def __str__(self) -> str:
        return "Linear"


# OPTYMALIZATORY 

class Optimizer(ABC):
    """Abstrakcyjna klasa bazowa dla optymalizatorów"""
    
    @abstractmethod
    def update(self, weights: np.ndarray, gradients: np.ndarray, 
               layer_index: int) -> np.ndarray:
        """Aktualizacja wag na podstawie gradientów"""
        pass
    
    @abstractmethod
    def __str__(self) -> str:
        """Reprezentacja tekstowa"""
        pass


class SGD(Optimizer):
    """Stochastic Gradient Descent z opcjonalnym momentum"""
    
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocities = {}
    
    def update(self, weights: np.ndarray, gradients: np.ndarray, 
               layer_index: int) -> np.ndarray:
        if layer_index not in self.velocities:
            self.velocities[layer_index] = np.zeros_like(weights)
        
        self.velocities[layer_index] = (self.momentum * self.velocities[layer_index] - 
                                        self.learning_rate * gradients)
        return weights + self.velocities[layer_index]
    
    def __str__(self) -> str:
        return f"SGD(lr={self.learning_rate}, momentum={self.momentum})"


class RMSProp(Optimizer):
    """RMSProp optimizer"""
    
    def __init__(self, learning_rate: float = 0.001, beta: float = 0.9, 
                 epsilon: float = 1e-8):
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.squared_gradients = {}
    
    def update(self, weights: np.ndarray, gradients: np.ndarray, 
               layer_index: int) -> np.ndarray:
        if layer_index not in self.squared_gradients:
            self.squared_gradients[layer_index] = np.zeros_like(weights)
        
        self.squared_gradients[layer_index] = (self.beta * self.squared_gradients[layer_index] + 
                                               (1 - self.beta) * gradients ** 2)
        return weights - self.learning_rate * gradients / (np.sqrt(self.squared_gradients[layer_index]) + self.epsilon)
    
    def __str__(self) -> str:
        return f"RMSProp(lr={self.learning_rate}, beta={self.beta})"


class Adam(Optimizer):
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, 
                 beta2: float = 0.999, epsilon: float = 1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = {}
    
    def update(self, weights: np.ndarray, gradients: np.ndarray, 
               layer_index: int) -> np.ndarray:
        if layer_index not in self.m:
            self.m[layer_index] = np.zeros_like(weights)
            self.v[layer_index] = np.zeros_like(weights)
            self.t[layer_index] = 0
        
        self.t[layer_index] += 1
        
        # Aktualizacja momentów
        self.m[layer_index] = self.beta1 * self.m[layer_index] + (1 - self.beta1) * gradients
        self.v[layer_index] = self.beta2 * self.v[layer_index] + (1 - self.beta2) * gradients ** 2
        
        # Korekcja bias
        m_hat = self.m[layer_index] / (1 - self.beta1 ** self.t[layer_index])
        v_hat = self.v[layer_index] / (1 - self.beta2 ** self.t[layer_index])
        
        return weights - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
    
    def __str__(self) -> str:
        return f"Adam(lr={self.learning_rate}, beta1={self.beta1}, beta2={self.beta2})"


# WARSTWA SIECI 

class Layer:
    """Pojedyncza warstwa sieci neuronowej"""
    
    def __init__(self, input_size: int, output_size: int, 
                 activation: ActivationFunction = None,
                 weight_init: str = 'xavier'):
        """
        Inicjalizacja warstwy
        
        Args:
            input_size: Liczba wejść
            output_size: Liczba wyjść (neuronów)
            activation: Funkcja aktywacji
            weight_init: Metoda inicjalizacji wag ('xavier', 'he', 'uniform')
        """
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation if activation else ReLU()
        
        # Inicjalizacja wag
        if weight_init == 'xavier':
            # Xavier/Glorot initialization
            limit = np.sqrt(6 / (input_size + output_size))
            self.weights = np.random.uniform(-limit, limit, (input_size, output_size))
        elif weight_init == 'he':
            # He initialization (dobra dla ReLU)
            self.weights = np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)
        else:
            # Uniform initialization
            self.weights = np.random.uniform(-0.5, 0.5, (input_size, output_size))
        
        self.bias = np.zeros((1, output_size))
        
        # Cache dla backpropagation
        self.input_cache = None
        self.z_cache = None
        self.output_cache = None
        
        # Gradienty
        self.weight_gradient = None
        self.bias_gradient = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Przejście w przód przez warstwę"""
        self.input_cache = x
        self.z_cache = x @ self.weights + self.bias
        self.output_cache = self.activation.forward(self.z_cache)
        return self.output_cache
    
    def backward(self, gradient: np.ndarray) -> np.ndarray:
        """Propagacja wsteczna przez warstwę"""
        # Gradient względem aktywacji
        activation_gradient = gradient * self.activation.backward(self.z_cache)
        
        # Gradient względem wag i biasu
        self.weight_gradient = self.input_cache.T @ activation_gradient
        self.bias_gradient = np.sum(activation_gradient, axis=0, keepdims=True)
        
        # Gradient względem wejścia
        input_gradient = activation_gradient @ self.weights.T
        
        return input_gradient
    
    def update_weights(self, optimizer: Optimizer, layer_index: int):
        """Aktualizacja wag używając optymalizatora"""
        self.weights = optimizer.update(self.weights, self.weight_gradient, 
                                       layer_index * 2)
        self.bias = optimizer.update(self.bias, self.bias_gradient, 
                                   layer_index * 2 + 1)


# GŁĘBOKI MLP 

class DeepMLP:
    """Głęboki perceptron wielowarstwowy"""
    
    def __init__(self, 
                 input_size: int,
                 hidden_sizes: List[int],
                 output_size: int,
                 hidden_activations: Union[List[ActivationFunction], ActivationFunction] = None,
                 output_activation: ActivationFunction = None,
                 optimizer: Optimizer = None,
                 weight_init: str = 'xavier',
                 task_type: str = 'classification',
                 random_seed: Optional[int] = None):
        """
        Inicjalizacja głębokiego MLP
        
        Args:
            input_size: Wymiar wejścia
            hidden_sizes: Lista z liczbą neuronów w każdej warstwie ukrytej
            output_size: Wymiar wyjścia
            hidden_activations: Funkcje aktywacji dla warstw ukrytych
            output_activation: Funkcja aktywacji dla warstwy wyjściowej
            optimizer: Optymalizator
            weight_init: Metoda inicjalizacji wag
            task_type: 'classification' lub 'regression'
            random_seed: Ziarno dla generatora losowego
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.task_type = task_type
        
        # Domyślne funkcje aktywacji
        if hidden_activations is None:
            hidden_activations = ReLU()
        
        if isinstance(hidden_activations, ActivationFunction):
            hidden_activations = [copy.deepcopy(hidden_activations) 
                                 for _ in range(len(hidden_sizes))]
        
        if output_activation is None:
            if task_type == 'classification':
                # Sprawdź, czy klasyfikacja binarna czy wieloklasowa
                output_activation = Sigmoid() if output_size == 1 else Softmax()
            else:
                output_activation = Linear()
        
        # Domyślny optymalizator
        if optimizer is None:
            optimizer = Adam(learning_rate=0.001)
        self.optimizer = optimizer
        
        # Tworzenie warstw
        self.layers = []
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        
        for i in range(len(layer_sizes) - 1):
            if i < len(hidden_sizes):
                activation = hidden_activations[i]
            else:
                activation = output_activation
            
            layer = Layer(layer_sizes[i], layer_sizes[i+1], 
                         activation, weight_init)
            self.layers.append(layer)
        
        # Historia treningu
        self.train_history = {
            'loss': [],
            'val_loss': [],
            'metrics': []
        }
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Przejście w przód przez całą sieć"""
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Propagacja wsteczna przez całą sieć"""
        batch_size = y_true.shape[0]
    
        if self.task_type == 'classification':
            epsilon = 1e-15
            y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
            
            # Sprawdź czy binarna czy wieloklasowa
            if self.output_size == 1 or y_true.shape[1] == 1:
                # Binary cross-entropy
                loss = -np.mean(y_true * np.log(y_pred) + 
                            (1 - y_true) * np.log(1 - y_pred))
            else:
                # Categorical cross-entropy
                loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
            
            gradient = (y_pred - y_true) / batch_size
        else:
            # MSE loss dla regresji
            loss = np.mean((y_pred - y_true) ** 2)
            gradient = 2 * (y_pred - y_true) / batch_size
        
        # Propagacja wsteczna przez warstwy
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)
        
        return loss
    
    def update_weights(self):
        """Aktualizacja wag wszystkich warstw"""
        for i, layer in enumerate(self.layers):
            layer.update_weights(self.optimizer, i)
    
    def train(self, 
              X_train: np.ndarray, 
              y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              epochs: int = 100,
              batch_size: int = 32,
              verbose: bool = True,
              early_stopping: bool = False,
              patience: int = 10) -> Dict[str, List[float]]:
        """
        Trenowanie sieci
        
        Args:
            X_train: Dane treningowe
            y_train: Etykiety treningowe
            X_val: Dane walidacyjne
            y_val: Etykiety walidacyjne
            epochs: Liczba epok
            batch_size: Rozmiar batcha
            verbose: Czy wyświetlać postęp
            early_stopping: Czy używać early stopping
            patience: Cierpliwość dla early stopping
            
        Returns:
            Historia treningu
        """
        n_samples = X_train.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_weights = None
        
        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            epoch_losses = []
            
            # Mini-batch training
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Forward pass
                y_pred = self.forward(X_batch)
                
                # Backward pass
                loss = self.backward(y_batch, y_pred)
                epoch_losses.append(loss)
                
                # Update weights
                self.update_weights()
            
            # Średnia strata epoki
            avg_loss = np.mean(epoch_losses)
            self.train_history['loss'].append(avg_loss)
            
            # Walidacja
            if X_val is not None:
                val_pred = self.predict_proba(X_val) if self.task_type == 'classification' else self.predict(X_val)
                if self.task_type == 'classification':
                    epsilon = 1e-15
                    val_pred = np.clip(val_pred, epsilon, 1 - epsilon)
                    val_loss = -np.mean(y_val * np.log(val_pred) + 
                                      (1 - y_val) * np.log(1 - val_pred))
                else:
                    val_loss = np.mean((val_pred - y_val) ** 2)
                
                self.train_history['val_loss'].append(val_loss)
                
                # Early stopping
                if early_stopping:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_weights = self._get_weights_copy()
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            if verbose:
                                print(f"Early stopping at epoch {epoch+1}")
                            if best_weights is not None:
                                self._set_weights(best_weights)
                            break
            
            # Wyświetlanie postępu
            if verbose and (epoch + 1) % 10 == 0:
                msg = f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}"
                if X_val is not None:
                    msg += f", Val Loss: {val_loss:.6f}"
                print(msg)
        
        return self.train_history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predykcja dla nowych danych"""
        if self.task_type == 'classification':
            proba = self.predict_proba(X)
            if self.output_size == 1:
                return (proba > 0.5).astype(int).flatten()
            else:
                return np.argmax(proba, axis=1)
        else:
            return self.forward(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predykcja prawdopodobieństw dla klasyfikacji"""
        return self.forward(X)
    
    def _get_weights_copy(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Kopia wag sieci"""
        return [(layer.weights.copy(), layer.bias.copy()) 
                for layer in self.layers]
    
    def _set_weights(self, weights: List[Tuple[np.ndarray, np.ndarray]]):
        """Ustawienie wag sieci"""
        for layer, (w, b) in zip(self.layers, weights):
            layer.weights = w.copy()
            layer.bias = b.copy()


# EWALUACJA I METRYKI 

class Evaluator:
    """Klasa do ewaluacji modeli"""
    
    @staticmethod
    def classification_metrics(y_true: np.ndarray, 
                              y_pred: np.ndarray) -> Dict[str, float]:
        """Obliczanie metryk dla klasyfikacji"""
        # Konwersja do 1D jeśli potrzeba
        if y_true.ndim > 1:
            y_true = y_true.flatten()
        if y_pred.ndim > 1:
            y_pred = y_pred.flatten()
        
        y_true = y_true.astype(int)
        y_pred = y_pred.astype(int)
        
        # Obliczanie metryk
        accuracy = accuracy_score(y_true, y_pred)
        
        # Dla problemów wieloklasowych używamy average='weighted'
        unique_classes = np.unique(y_true)
        if len(unique_classes) > 2:
            precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        else:
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    @staticmethod
    def regression_metrics(y_true: np.ndarray, 
                          y_pred: np.ndarray) -> Dict[str, float]:
        """Obliczanie metryk dla regresji"""
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mse)
        
        # Obliczanie "accuracy" dla regresji (procent predykcji w zakresie 10% prawdziwej wartości)
        threshold = 0.1  # 10%
        relative_error = np.abs(y_pred - y_true) / (np.abs(y_true) + 1e-10)
        accuracy = np.mean(relative_error <= threshold)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'accuracy_10pct': accuracy
        }


# PRZYGOTOWANIE DANYCH 

class DataPreparator:
    """Klasa do przygotowania i podziału danych"""
    
    @staticmethod
    def prepare_classification_data(X: np.ndarray, y: np.ndarray, 
                                   split_variant: str = 'A',
                                   random_state: int = 42) -> Dict[str, np.ndarray]:
        """
        Przygotowanie danych do klasyfikacji z zachowaniem proporcji klas
        
        Args:
            X: Cechy
            y: Etykiety
            split_variant: 'A' (80/20) lub 'B' (70/15/15)
            random_state: Ziarno losowe
            
        Returns:
            Słownik z podzielonymi danymi
        """
        if split_variant == 'A':
            # 80% train, 20% test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=random_state
            )
            return {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test
            }
        else:  # variant B
            # 70% train, 15% val, 15% test
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=0.15, stratify=y, random_state=random_state
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=0.176, stratify=y_temp, 
                random_state=random_state  # 0.176 * 0.85 ≈ 0.15
            )
            return {
                'X_train': X_train,
                'X_val': X_val,
                'X_test': X_test,
                'y_train': y_train,
                'y_val': y_val,
                'y_test': y_test
            }
    
    @staticmethod
    def prepare_regression_data(X: np.ndarray, y: np.ndarray,
                               split_variant: str = 'A') -> Dict[str, np.ndarray]:
        """
        Przygotowanie danych do regresji (szereg czasowy)
        
        Args:
            X: Cechy
            y: Wartości docelowe
            split_variant: 'A' (80/20) lub 'B' (70/15/15)
            
        Returns:
            Słownik z podzielonymi danymi
        """
        n_samples = len(X)
        
        if split_variant == 'A':
            # 80% train, 20% test
            train_size = int(0.8 * n_samples)
            X_train = X[:train_size]
            X_test = X[train_size:]
            y_train = y[:train_size]
            y_test = y[train_size:]
            
            return {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test
            }
        else:  # variant B
            # 70% train, 15% val, 15% test
            train_size = int(0.7 * n_samples)
            val_size = int(0.85 * n_samples)
            
            X_train = X[:train_size]
            X_val = X[train_size:val_size]
            X_test = X[val_size:]
            y_train = y[:train_size]
            y_val = y[train_size:val_size]
            y_test = y[val_size:]
            
            return {
                'X_train': X_train,
                'X_val': X_val,
                'X_test': X_test,
                'y_train': y_train,
                'y_val': y_val,
                'y_test': y_test
            }


# OPTYMALIZACJA HIPERPARAMETRÓW 

@dataclass
class HyperparameterConfig:
    """Konfiguracja hiperparametrów"""
    n_hidden_layers: int
    hidden_sizes: List[int]
    learning_rate: float
    momentum: float
    optimizer_type: str
    activation_functions: List[str]
    weight_init: str = 'xavier'



class HyperparameterOptimizer:
    """Klasa do optymalizacji hiperparametrów"""
    
    def __init__(self, task_type: str = 'classification', n_runs: int = 5, n_jobs: int = -1):
            """
            Args:
                task_type: 'classification' lub 'regression'
                n_runs: Liczba powtórzeń dla każdej konfiguracji
                n_jobs: Liczba procesów (-1 = wszystkie rdzenie, 1 = bez równoległości)
            """
            self.task_type = task_type
            self.n_runs = n_runs
            self.n_jobs = cpu_count() if n_jobs == -1 else max(1, n_jobs)
            self.results_df = pd.DataFrame()
    
    def _create_optimizer(self, optimizer_type: str, 
                          learning_rate: float, 
                          momentum: float) -> Optimizer:
        """Tworzenie optymalizatora na podstawie typu"""
        if optimizer_type == 'SGD':
            return SGD(learning_rate=learning_rate, momentum=momentum)
        elif optimizer_type == 'RMSProp':
            return RMSProp(learning_rate=learning_rate)
        elif optimizer_type == 'Adam':
            return Adam(learning_rate=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    def _create_activation(self, activation_name: str) -> ActivationFunction:
        """Tworzenie funkcji aktywacji na podstawie nazwy"""
        activations = {
            'sigmoid': Sigmoid(),
            'relu': ReLU(),
            'tanh': Tanh(),
            'leaky_relu': LeakyReLU(),
            'elu': ELU()
        }
        return activations.get(activation_name.lower(), ReLU())
    
    def grid_search(self, 
                   X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray,
                   param_grid: Dict[str, List]) -> pd.DataFrame:
        """
        Grid Search po hiperparametrach
        
        Args:
            X_train: Dane treningowe
            y_train: Etykiety treningowe
            X_val: Dane walidacyjne
            y_val: Etykiety walidacyjne
            param_grid: Siatka parametrów
            
        Returns:
            DataFrame z wynikami
        """
        results = []
        total_combinations = 1
        for values in param_grid.values():
            total_combinations *= len(values)
        
        print(f"Grid Search: testowanie {total_combinations} kombinacji...")
        combination_idx = 0
        
        # Iteracja po wszystkich kombinacjach
        for n_layers in param_grid.get('n_hidden_layers', [2]):
            for n_neurons in param_grid.get('n_neurons', [64]):
                for lr in param_grid.get('learning_rate', [0.001]):
                    for momentum in param_grid.get('momentum', [0.0]):
                        for opt_type in param_grid.get('optimizer', ['Adam']):
                            for activation in param_grid.get('activation', ['relu']):
                                combination_idx += 1
                                print(f"  Kombinacja {combination_idx}/{total_combinations}: "
                                     f"layers={n_layers}, neurons={n_neurons}, lr={lr}, "
                                     f"opt={opt_type}, act={activation}")
                                
                                config = HyperparameterConfig(
                                    n_hidden_layers=n_layers,
                                    hidden_sizes=[n_neurons] * n_layers,
                                    learning_rate=lr,
                                    momentum=momentum,
                                    optimizer_type=opt_type,
                                    activation_functions=[activation] * n_layers
                                )
                                
                                run_results = self._evaluate_config(
                                    config, X_train, y_train, X_val, y_val
                                )
                                
                                result = {
                                    'n_hidden_layers': n_layers,
                                    'n_neurons': n_neurons,
                                    'learning_rate': lr,
                                    'momentum': momentum,
                                    'optimizer': opt_type,
                                    'activation': activation,
                                    **run_results
                                }
                                results.append(result)
        
        self.results_df = pd.DataFrame(results)
        return self.results_df
    
    def random_search(self, 
                     X_train: np.ndarray, y_train: np.ndarray,
                     X_val: np.ndarray, y_val: np.ndarray,
                     param_distributions: Dict[str, Any],
                     n_iter: int = 20,
                     use_parallel: bool = True) -> pd.DataFrame:
        """
        Random Search po hiperparametrach z opcjonalną równoległością
        
        Args:
            X_train: Dane treningowe
            y_train: Etykiety treningowe  
            X_val: Dane walidacyjne
            y_val: Etykiety walidacyjne
            param_distributions: Rozkłady parametrów
            n_iter: Liczba iteracji
            use_parallel: Czy użyć przetwarzania równoległego
            
        Returns:
            DataFrame z wynikami
        """
        print(f"Random Search: testowanie {n_iter} konfiguracji...")
        
        configs = []
        for i in range(n_iter):
            config = self._generate_random_config(param_distributions, i)
            configs.append(config)
        
        if use_parallel and self.n_jobs > 1:
            print(f"  Używam {self.n_jobs} procesów równoległych")
            results = self._parallel_evaluate_configs(
                configs, X_train, y_train, X_val, y_val
            )
        else:
            print("  Wykonanie sekwencyjne")
            results = self._sequential_evaluate_configs(
                configs, X_train, y_train, X_val, y_val
            )
        
        self.results_df = pd.DataFrame(results)
        return self.results_df
    
    def _generate_random_config(self, param_distributions: Dict[str, Any], 
                               iteration: int) -> Dict[str, Any]:
        """Generowanie losowej konfiguracji"""
        n_layers = random.choice(param_distributions.get('n_hidden_layers', [2, 4, 6]))
        n_neurons = random.choice(param_distributions.get('n_neurons', [32, 64, 128, 256]))
        lr = 10 ** random.uniform(
            param_distributions.get('learning_rate_log', [-4, -1])[0],
            param_distributions.get('learning_rate_log', [-4, -1])[1]
        )
        momentum = random.uniform(
            param_distributions.get('momentum', [0.0, 0.9])[0],
            param_distributions.get('momentum', [0.0, 0.9])[1]
        )
        opt_type = random.choice(param_distributions.get('optimizer', ['Adam', 'SGD', 'RMSProp']))
        activation = random.choice(param_distributions.get('activation', 
                                                          ['relu', 'tanh', 'sigmoid', 'leaky_relu']))
        
        return {
            'iteration': iteration,
            'n_layers': n_layers,
            'n_neurons': n_neurons,
            'lr': lr,
            'momentum': momentum,
            'opt_type': opt_type,
            'activation': activation
        }
    
    def _parallel_evaluate_configs(self, configs: List[Dict[str, Any]],
                              X_train: np.ndarray, y_train: np.ndarray,
                              X_val: np.ndarray, y_val: np.ndarray) -> List[Dict[str, Any]]:
        """Równoległa ewaluacja konfiguracji"""
        results = []
        
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            # Przygotowanie zadań
            futures = {}
            for config in configs:
                future = executor.submit(
                    _evaluate_single_config_wrapper,
                    config, X_train, y_train, X_val, y_val,
                    self.task_type, self.n_runs
                )
                futures[future] = config
            
            # Zbieranie wyników w miarę ukończenia
            for i, future in enumerate(as_completed(futures), 1):
                config = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    print(f"  ✓ Ukończono {i}/{len(configs)}: "
                        f"layers={config['n_layers']}, neurons={config['n_neurons']}, "
                        f"lr={config['lr']:.6f}, momentum={config['momentum']:.3f}, "
                        f"opt={config['opt_type']}, act={config['activation']}")
                except Exception as e:
                    print(f"  ✗ Błąd w konfiguracji {i}: {e}")
                    traceback.print_exc()
        
        return results
    
    def _sequential_evaluate_configs(self, configs: List[Dict[str, Any]],
                                    X_train: np.ndarray, y_train: np.ndarray,
                                    X_val: np.ndarray, y_val: np.ndarray) -> List[Dict[str, Any]]:
        """Sekwencyjna ewaluacja konfiguracji"""
        results = []
        
        for i, config in enumerate(configs, 1):
            print(f"  Konfiguracja {i}/{len(configs)}: "
                 f"layers={config['n_layers']}, neurons={config['n_neurons']}, "
                 f"lr={config['lr']:.6f}, opt={config['opt_type']}, act={config['activation']}")
            
            hp_config = HyperparameterConfig(
                n_hidden_layers=config['n_layers'],
                hidden_sizes=[config['n_neurons']] * config['n_layers'],
                learning_rate=config['lr'],
                momentum=config['momentum'],
                optimizer_type=config['opt_type'],
                activation_functions=[config['activation']] * config['n_layers']
            )
            
            run_results = self._evaluate_config(
                hp_config, X_train, y_train, X_val, y_val
            )
            
            result = {
                'n_hidden_layers': config['n_layers'],
                'n_neurons': config['n_neurons'],
                'learning_rate': config['lr'],
                'momentum': config['momentum'],
                'optimizer': config['opt_type'],
                'activation': config['activation'],
                **run_results
            }
            results.append(result)
        
        return results
    
    def _evaluate_config(self, 
                        config: HyperparameterConfig,
                        X_train: np.ndarray, y_train: np.ndarray,
                        X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
        """Ewaluacja pojedynczej konfiguracji"""
        metrics_list = []
        
        for run in range(self.n_runs):
            input_size = X_train.shape[1]
            output_size = y_train.shape[1] if len(y_train.shape) > 1 else 1
            
            activations = [self._create_activation(act) 
                          for act in config.activation_functions]
            
            optimizer = self._create_optimizer(config.optimizer_type, 
                                              config.learning_rate, 
                                              config.momentum)
            
            model = DeepMLP(
                input_size=input_size,
                hidden_sizes=config.hidden_sizes,
                output_size=output_size,
                hidden_activations=activations,
                optimizer=optimizer,
                weight_init=config.weight_init,
                task_type=self.task_type,
                random_seed=42 + run  
            )
            
            # Trenowanie
            model.train(X_train, y_train, X_val, y_val, 
                       epochs=150, batch_size=32, verbose=False, early_stopping=True, patience=20)
            
            # Ewaluacja
            y_pred = model.predict(X_val)
            
            if self.task_type == 'classification':
                metrics = Evaluator.classification_metrics(y_val, y_pred)
            else:
                metrics = Evaluator.regression_metrics(y_val, y_pred)
            
            metrics_list.append(metrics)
        
        # Agregacja wyników
        aggregated_results = {}
        metric_names = metrics_list[0].keys()
        
        for metric in metric_names:
            values = [m[metric] for m in metrics_list]
            aggregated_results[f'{metric}_mean'] = np.mean(values)
            aggregated_results[f'{metric}_std'] = np.std(values)
            aggregated_results[f'{metric}_max'] = np.max(values)
        
        return aggregated_results
    
    def get_best_config(self, metric: str = 'accuracy_mean') -> Dict[str, Any]:
        """Zwraca najlepszą konfigurację"""
        if self.results_df.empty:
            raise ValueError("Brak wyników. Najpierw uruchom grid_search lub random_search.")
        
        best_idx = self.results_df[metric].idxmax()
        return self.results_df.iloc[best_idx].to_dict()

def _evaluate_single_config_wrapper(config: Dict[str, Any],
                                   X_train: np.ndarray, y_train: np.ndarray,
                                   X_val: np.ndarray, y_val: np.ndarray,
                                   task_type: str, n_runs: int) -> Dict[str, Any]:
    """
    Wrapper do ewaluacji pojedynczej konfiguracji (dla multiprocessing)
    """
    # Funkcje pomocnicze
    def _create_optimizer(optimizer_type: str, learning_rate: float, momentum: float) -> Optimizer:
        if optimizer_type == 'SGD':
            return SGD(learning_rate=learning_rate, momentum=momentum)
        elif optimizer_type == 'RMSProp':
            return RMSProp(learning_rate=learning_rate)
        elif optimizer_type == 'Adam':
            return Adam(learning_rate=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    def _create_activation(activation_name: str) -> ActivationFunction:
        activations = {
            'sigmoid': Sigmoid(),
            'relu': ReLU(),
            'tanh': Tanh(),
            'leaky_relu': LeakyReLU(),
            'elu': ELU()
        }
        return activations.get(activation_name.lower(), ReLU())
    
    def _prepare_labels_for_metrics(y_true: np.ndarray, y_pred: np.ndarray):
        """Konwersja do 1D dla metryk"""
        if y_true.ndim > 1 and y_true.shape[1] > 1:
            y_true_1d = np.argmax(y_true, axis=1)
        else:
            y_true_1d = y_true.flatten()
        
        if y_pred.ndim > 1:
            y_pred_1d = y_pred.flatten()
        else:
            y_pred_1d = y_pred
        
        return y_true_1d.astype(int), y_pred_1d.astype(int)
    
    # Tworzenie konfiguracji
    hp_config = HyperparameterConfig(
        n_hidden_layers=config['n_layers'],
        hidden_sizes=[config['n_neurons']] * config['n_layers'],
        learning_rate=config['lr'],
        momentum=config['momentum'],
        optimizer_type=config['opt_type'],
        activation_functions=[config['activation']] * config['n_layers'],
        weight_init='xavier'
    )
    
    # Ewaluacja wielokrotna
    metrics_list = []
    
    for run in range(n_runs):
        # Tworzenie modelu
        input_size = X_train.shape[1]
        output_size = y_train.shape[1] if len(y_train.shape) > 1 else 1
        
        activations = [_create_activation(act) 
                      for act in hp_config.activation_functions]
        
        optimizer = _create_optimizer(hp_config.optimizer_type, 
                                     hp_config.learning_rate, 
                                     hp_config.momentum)
        
        model = DeepMLP(
            input_size=input_size,
            hidden_sizes=hp_config.hidden_sizes,
            output_size=output_size,
            hidden_activations=activations,
            optimizer=optimizer,
            weight_init=hp_config.weight_init,
            task_type=task_type,
            random_seed=42 + run
        )
        
        # Trenowanie
        model.train(X_train, y_train, X_val, y_val, 
                   epochs=150, batch_size=128, verbose=False,  # batch_size=128
                   early_stopping=True, patience=20)
        
        # Ewaluacja
        y_pred = model.predict(X_val)
        
        if task_type == 'classification':
            y_true_val, y_pred_val = _prepare_labels_for_metrics(y_val, y_pred)
            metrics = Evaluator.classification_metrics(y_true_val, y_pred_val)
        else:
            metrics = Evaluator.regression_metrics(y_val, y_pred)
        
        metrics_list.append(metrics)
    
    # Agregacja wyników
    aggregated_results = {}
    metric_names = metrics_list[0].keys()
    
    for metric in metric_names:
        values = [m[metric] for m in metrics_list]
        aggregated_results[f'{metric}_mean'] = np.mean(values)
        aggregated_results[f'{metric}_std'] = np.std(values)
        aggregated_results[f'{metric}_max'] = np.max(values)
    
    return {
        'n_hidden_layers': config['n_layers'],
        'n_neurons': config['n_neurons'],
        'learning_rate': config['lr'],
        'momentum': config['momentum'],
        'optimizer': config['opt_type'],
        'activation': config['activation'],
        **aggregated_results
    }

# PORÓWNANIE Z BIBLIOTEKĄ 

class LibraryComparison:
    """Klasa do porównania z implementacją sklearn"""
    
    @staticmethod
    def compare_classification(X_train: np.ndarray, y_train: np.ndarray,
                              X_test: np.ndarray, y_test: np.ndarray,
                              custom_model: DeepMLP,
                              hidden_layer_sizes: tuple = (100, 50)) -> pd.DataFrame:
        """Porównanie dla problemu klasyfikacji"""
        results = []
        
        # POMOCNICZA FUNKCJA DO KONWERSJI ETYKIET
        def prepare_labels(y_true, y_pred):
            """Konwersja etykiet do formatu 1D"""
            if y_true.ndim > 1 and y_true.shape[1] > 1:
                y_true_1d = np.argmax(y_true, axis=1)
            else:
                y_true_1d = y_true.flatten()
            
            if y_pred.ndim > 1:
                y_pred_1d = y_pred.flatten()
            else:
                y_pred_1d = y_pred
            
            return y_true_1d.astype(int), y_pred_1d.astype(int)
        
        # Trenowanie własnego modelu
        start_time = time.time()
        custom_model.train(X_train, y_train, epochs=100, batch_size=128, verbose=False)
        custom_train_time = time.time() - start_time
        
        # Predykcja własnym modelem
        y_pred_custom = custom_model.predict(X_test)
        
        # KONWERSJA DO 1D
        y_test_1d, y_pred_custom_1d = prepare_labels(y_test, y_pred_custom)
        custom_metrics = Evaluator.classification_metrics(y_test_1d, y_pred_custom_1d)
        
        results.append({
            'model': 'Custom MLP',
            'train_time': custom_train_time,
            **custom_metrics
        })
        
        y_train_sklearn = np.argmax(y_train, axis=1) if y_train.ndim > 1 and y_train.shape[1] > 1 else y_train.ravel()
        
        sklearn_model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation='relu',
            solver='adam',
            learning_rate_init=0.001,
            max_iter=100,
            batch_size=128,
            random_state=42
        )
        
        start_time = time.time()
        sklearn_model.fit(X_train, y_train_sklearn)
        sklearn_train_time = time.time() - start_time
        
        y_pred_sklearn = sklearn_model.predict(X_test)
        
        y_test_sklearn, y_pred_sklearn = prepare_labels(y_test, y_pred_sklearn)
        sklearn_metrics = Evaluator.classification_metrics(y_test_sklearn, y_pred_sklearn)
        
        results.append({
            'model': 'Sklearn MLP',
            'train_time': sklearn_train_time,
            **sklearn_metrics
        })
        
        return pd.DataFrame(results)
    
    @staticmethod
    def compare_regression(X_train: np.ndarray, y_train: np.ndarray,
                          X_test: np.ndarray, y_test: np.ndarray,
                          custom_model: DeepMLP,
                          hidden_layer_sizes: tuple = (100, 50)) -> pd.DataFrame:
        """Porównanie dla problemu regresji"""
        results = []
        
        # Trenowanie własnego modelu
        start_time = time.time()
        custom_model.train(X_train, y_train, epochs=100, batch_size=32, verbose=False)
        custom_train_time = time.time() - start_time
        
        # Predykcja własnym modelem
        y_pred_custom = custom_model.predict(X_test)
        custom_metrics = Evaluator.regression_metrics(y_test, y_pred_custom)
        
        results.append({
            'model': 'Custom MLP',
            'train_time': custom_train_time,
            **custom_metrics
        })
        
        # Model sklearn
        sklearn_model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation='relu',
            solver='adam',
            learning_rate_init=0.001,
            max_iter=100,
            random_state=42
        )
        
        start_time = time.time()
        sklearn_model.fit(X_train, y_train.ravel() if y_train.ndim > 1 else y_train)
        sklearn_train_time = time.time() - start_time
        
        y_pred_sklearn = sklearn_model.predict(X_test)
        sklearn_metrics = Evaluator.regression_metrics(y_test, y_pred_sklearn)
        
        results.append({
            'model': 'Sklearn MLP',
            'train_time': sklearn_train_time,
            **sklearn_metrics
        })
        
        return pd.DataFrame(results)


# GŁÓWNA KLASA EKSPERYMENTU 

class MLPExperiment:
    """Główna klasa do przeprowadzania eksperymentów"""
    
    def __init__(self, task_type: str = 'classification', 
                 split_variant: str = 'B',
                 n_repeats: int = 5,
                 n_jobs: int = -1):
        """
        Args:
            task_type: 'classification' lub 'regression'
            split_variant: 'A' lub 'B'
            n_repeats: Liczba powtórzeń eksperymentu
            n_jobs: Liczba procesów dla równoległości (-1 = wszystkie rdzenie)
        """
        self.task_type = task_type
        self.split_variant = split_variant
        self.n_repeats = n_repeats
        self.n_jobs = n_jobs
        self.results = {}

    def _prepare_labels_for_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Przygotowanie etykiet do obliczeń metryk
        
        Args:
            y_true: Prawdziwe etykiety (mogą być one-hot)
            y_pred: Predykcje (klasy)
            
        Returns:
            Tuple z etykietami w formacie 1D
        """
        # Konwersja y_true z one-hot do klas
        if y_true.ndim > 1 and y_true.shape[1] > 1:
            y_true_1d = np.argmax(y_true, axis=1)
        else:
            y_true_1d = y_true.flatten()
        
        # y_pred już powinno być w formacie klas (z model.predict())
        if y_pred.ndim > 1:
            y_pred_1d = y_pred.flatten()
        else:
            y_pred_1d = y_pred
        
        return y_true_1d.astype(int), y_pred_1d.astype(int)

    def _perform_random_search(self, data: Dict[str, np.ndarray]) -> pd.DataFrame:
        """Wykonanie Random Search z równoległością"""
        optimizer = HyperparameterOptimizer(
            self.task_type, 
            n_runs=3, 
            n_jobs=self.n_jobs  # PRZEKAZANIE n_jobs
        )
        
        param_distributions = {
            'n_hidden_layers': [2, 3, 4, 5, 6, 7, 8],
            'n_neurons': list(range(16, 257, 16)),
            'learning_rate_log': [-4, -1],
            'momentum': [0.0, 0.9],
            'optimizer': ['SGD', 'Adam', 'RMSProp'],
            'activation': ['relu', 'tanh', 'sigmoid', 'leaky_relu', 'elu']
        }
        
        X_val = data.get('X_val', data['X_test'])
        y_val = data.get('y_val', data['y_test'])
        
        results = optimizer.random_search(
            data['X_train'], data['y_train'],
            X_val, y_val,
            param_distributions,
            n_iter=20,
            use_parallel=True
        )
        
        if self.task_type == 'classification':
            results = results.sort_values('accuracy_mean', ascending=False)
        else:
            results = results.sort_values('r2_mean', ascending=False)
        
        return results
    
    def run_full_experiment(self, X: np.ndarray = None, y: np.ndarray = None,
                           data_dict: dict = None,
                           use_parallel_random_search: bool = True) -> Dict[str, Any]:
        """
        Przeprowadza pełny eksperyment
        
        Args:
            X: Dane wejściowe (opcjonalne jeśli data_dict podane)
            y: Etykiety/wartości docelowe (opcjonalne jeśli data_dict podane)
            data_dict: Gotowy słownik z podziałem danych
            use_parallel_random_search: Czy użyć równoległości w Random Search
            
        Returns:
            Słownik z wszystkimi wynikami
        """
        print("=" * 80)
        print(f"EKSPERYMENT MLP - {self.task_type.upper()}")
        print("=" * 80)
        
        # 1. Przygotowanie danych
        print("\n1. Przygotowanie danych...")
        
        if data_dict is not None:
            # Użyj gotowego podziału
            data = data_dict
            print("✓ Używam gotowego podziału danych")
        else:
            # Standardowy przepływ
            if y.ndim == 1:
                y = y.reshape(-1, 1)

            if self.task_type == 'classification':
                data = DataPreparator.prepare_classification_data(
                    X, y, self.split_variant)
            else:
                data = DataPreparator.prepare_regression_data(
                    X, y, self.split_variant)
            
            # Normalizacja danych
            data['X_train'] = data['X_train'].to_numpy() if hasattr(data['X_train'], 'to_numpy') else data['X_train']
            data['X_test'] = data['X_test'].to_numpy() if hasattr(data['X_test'], 'to_numpy') else data['X_test']
            if 'X_val' in data:
                data['X_val'] = data['X_val'].to_numpy() if hasattr(data['X_val'], 'to_numpy') else data['X_val']

        print(f"  Rozmiar zbioru treningowego: {data['X_train'].shape}")
        if 'X_val' in data:
            print(f"  Rozmiar zbioru walidacyjnego: {data['X_val'].shape}")
        print(f"  Rozmiar zbioru testowego: {data['X_test'].shape}")
        
        # 2. Trenowanie modelu podstawowego
        print("\n2. Trenowanie modelu podstawowego...")
        basic_results = self._train_basic_model(data)
        self.results['basic_model'] = basic_results
        
        # 3. Random Search
        print("\n3. Random Search...")
        random_results = self._perform_random_search(data)
        self.results['random_search'] = random_results
        
        # 4. Porównanie z biblioteką sklearn
        print("\n4. Porównanie z biblioteką sklearn...")
        comparison_results = self._compare_with_sklearn(data)
        self.results['sklearn_comparison'] = comparison_results
        
        # 5. Trenowanie najlepszego modelu
        print("\n5. Trenowanie najlepszego modelu...")
        best_results = self._train_best_model(data)
        self.results['best_model'] = best_results
        
        # 6. Podsumowanie
        print("\n" + "=" * 80)
        print("PODSUMOWANIE WYNIKÓW")
        print("=" * 80)
        self._print_summary()
        
        return self.results
    
    def _train_basic_model(self, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Trenowanie podstawowego modelu"""
        input_size = data['X_train'].shape[1]
        output_size = data['y_train'].shape[1] if len(data['y_train'].shape) > 1 else 1
        
        # Konfiguracja podstawowa
        hidden_sizes = [128, 64, 32]  # 3 warstwy ukryte
        
        all_results = []
        
        for i in range(self.n_repeats):
            print(f"  Powtórzenie {i+1}/{self.n_repeats}")
            
            model = DeepMLP(
                input_size=input_size,
                hidden_sizes=hidden_sizes,
                output_size=output_size,
                hidden_activations=ReLU(),
                optimizer=Adam(learning_rate=0.001),
                task_type=self.task_type,
                random_seed=42 + i
            )
            
            # Trenowanie
            X_val = data.get('X_val', None)
            y_val = data.get('y_val', None)
            
            history = model.train(
                data['X_train'], data['y_train'],
                X_val, y_val,
                epochs=100,
                batch_size=128,  # Większy batch dla obrazów
                verbose=False,
                early_stopping=True,
                patience=10
            )
            
            # Ewaluacja na wszystkich zbiorach
            results = {}
            
            # Zbiór treningowy
            y_pred_train = model.predict(data['X_train'])
            if self.task_type == 'classification':
                y_true_train, y_pred_train = self._prepare_labels_for_metrics(data['y_train'], y_pred_train)
                results['train'] = Evaluator.classification_metrics(y_true_train, y_pred_train)
            else:
                results['train'] = Evaluator.regression_metrics(data['y_train'], y_pred_train)
            
            # Zbiór walidacyjny
            if X_val is not None:
                y_pred_val = model.predict(X_val)
                if self.task_type == 'classification':
                    y_true_val, y_pred_val = self._prepare_labels_for_metrics(y_val, y_pred_val)
                    results['val'] = Evaluator.classification_metrics(y_true_val, y_pred_val)
                else:
                    results['val'] = Evaluator.regression_metrics(y_val, y_pred_val)
            
            # Zbiór testowy
            y_pred_test = model.predict(data['X_test'])
            if self.task_type == 'classification':
                y_true_test, y_pred_test = self._prepare_labels_for_metrics(data['y_test'], y_pred_test)
                results['test'] = Evaluator.classification_metrics(y_true_test, y_pred_test)
            else:
                results['test'] = Evaluator.regression_metrics(data['y_test'], y_pred_test)
            
            all_results.append(results)
        
        # Agregacja wyników
        aggregated = self._aggregate_results(all_results)
        return aggregated
    
    def _perform_grid_search(self, data: Dict[str, np.ndarray]) -> pd.DataFrame:
        """Wykonanie Grid Search"""
        optimizer = HyperparameterOptimizer(self.task_type, n_runs=3)
        
        param_grid = {
            'n_hidden_layers': [2, 4, 6],
            'n_neurons': [32, 64, 128],
            'learning_rate': [0.01, 0.001, 0.0001],
            'momentum': [0.0, 0.5, 0.9],
            'optimizer': ['SGD', 'Adam'],
            'activation': ['relu', 'tanh', 'sigmoid']
        }
        
        X_val = data.get('X_val', data['X_test'])
        y_val = data.get('y_val', data['y_test'])
        
        results = optimizer.grid_search(
            data['X_train'], data['y_train'],
            X_val, y_val,
            param_grid
        )
        
        # Sortowanie po najlepszej metryce
        if self.task_type == 'classification':
            results = results.sort_values('accuracy_mean', ascending=False)
        else:
            results = results.sort_values('r2_mean', ascending=False)
        
        return results
    
    def _compare_with_sklearn(self, data: Dict[str, np.ndarray]) -> pd.DataFrame:
        """Porównanie z implementacją sklearn"""
        input_size = data['X_train'].shape[1]
        output_size = data['y_train'].shape[1] if len(data['y_train'].shape) > 1 else 1
        
        # Model własny
        custom_model = DeepMLP(
            input_size=input_size,
            hidden_sizes=[100, 50],
            output_size=output_size,
            optimizer=Adam(learning_rate=0.001),
            task_type=self.task_type,
            random_seed=42
        )
        
        if self.task_type == 'classification':
            return LibraryComparison.compare_classification(
                data['X_train'], data['y_train'],
                data['X_test'], data['y_test'],
                custom_model,
                hidden_layer_sizes=(100, 50)
            )
        else:
            return LibraryComparison.compare_regression(
                data['X_train'], data['y_train'],
                data['X_test'], data['y_test'],
                custom_model,
                hidden_layer_sizes=(100, 50)
            )
    
    def _train_best_model(self, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Trenowanie najlepszego modelu znalezionego podczas optymalizacji"""
        # Wybór najlepszej konfiguracji
        if self.task_type == 'classification':
            metric = 'accuracy_mean'
        else:
            metric = 'r2_mean'
        
        random_best = self.results['random_search'].iloc[0][metric]
        best_config = self.results['random_search'].iloc[0]
        
        print(f"  Parametry: layers={best_config['n_hidden_layers']}, "
            f"neurons={best_config['n_neurons']}, "
            f"lr={best_config['learning_rate']:.6f}, "
            f"opt={best_config['optimizer']}, "
            f"act={best_config['activation']}")
        
        # Trenowanie z najlepszymi parametrami
        input_size = data['X_train'].shape[1]
        output_size = data['y_train'].shape[1] if len(data['y_train'].shape) > 1 else 1

        hidden_sizes = [int(best_config['n_neurons'])] * int(best_config['n_hidden_layers'])

        # Tworzenie optymalizatora
        if best_config['optimizer'] == 'SGD':
            optimizer = SGD(best_config['learning_rate'], best_config['momentum'])
        elif best_config['optimizer'] == 'RMSProp':
            optimizer = RMSProp(best_config['learning_rate'])
        else:  # Adam
            optimizer = Adam(best_config['learning_rate'])
        
        # Tworzenie funkcji aktywacji
        activation_map = {
            'sigmoid': Sigmoid(),
            'relu': ReLU(),
            'tanh': Tanh(),
            'leaky_relu': LeakyReLU(),
            'elu': ELU()
        }
        activation = activation_map.get(best_config['activation'], ReLU())
        
        all_results = []
        
        for i in range(self.n_repeats):
            model = DeepMLP(
                input_size=input_size,
                hidden_sizes=hidden_sizes,
                output_size=output_size,
                hidden_activations=activation,
                optimizer=optimizer,
                task_type=self.task_type,
                random_seed=42 + i
            )
            
            # Trenowanie
            X_val = data.get('X_val', None)
            y_val = data.get('y_val', None)
            
            model.train(
                data['X_train'], data['y_train'],
                X_val, y_val,
                epochs=150,
                batch_size=128,  # Większy batch
                verbose=False,
                early_stopping=True,
                patience=15
            )
            
            # Ewaluacja
            y_pred_test = model.predict(data['X_test'])
            
            if self.task_type == 'classification':
                y_true_test, y_pred_test = self._prepare_labels_for_metrics(data['y_test'], y_pred_test)
                metrics = Evaluator.classification_metrics(y_true_test, y_pred_test)
            else:
                metrics = Evaluator.regression_metrics(data['y_test'], y_pred_test)
            
            all_results.append(metrics)
        
        # Agregacja
        final_results = {}
        for metric in all_results[0].keys():
            values = [r[metric] for r in all_results]
            final_results[f'{metric}_mean'] = np.mean(values)
            final_results[f'{metric}_std'] = np.std(values)
            final_results[f'{metric}_max'] = np.max(values)
        
        final_results['config'] = best_config.to_dict()
        
        return final_results
    
    def _aggregate_results(self, results_list: List[Dict]) -> Dict[str, Any]:
        """Agregacja wyników z wielu uruchomień"""
        aggregated = {}
        
        for dataset in results_list[0].keys():
            aggregated[dataset] = {}
            
            for metric in results_list[0][dataset].keys():
                values = [r[dataset][metric] for r in results_list]
                aggregated[dataset][f'{metric}_mean'] = np.mean(values)
                aggregated[dataset][f'{metric}_std'] = np.std(values)
                aggregated[dataset][f'{metric}_max'] = np.max(values)
                aggregated[dataset][f'{metric}_min'] = np.min(values)
        
        return aggregated
    
    def _print_summary(self):
        """Wyświetlenie podsumowania wyników"""
        print("\n--- Model podstawowy ---")
        if self.task_type == 'classification':
            metric_name = 'accuracy'
        else:
            metric_name = 'r2'
        
        for dataset in ['train', 'val', 'test']:
            if dataset in self.results['basic_model']:
                mean_val = self.results['basic_model'][dataset][f'{metric_name}_mean']
                std_val = self.results['basic_model'][dataset][f'{metric_name}_std']
                print(f"{dataset.capitalize():5s}: {metric_name}={mean_val:.4f} (±{std_val:.4f})")
        
        print("\n--- Najlepszy model (po optymalizacji) ---")
        best = self.results['best_model']
        print(f"Test: {metric_name}={best[f'{metric_name}_mean']:.4f} (±{best[f'{metric_name}_std']:.4f})")
        
        print("\n--- Porównanie z sklearn ---")
        print(self.results['sklearn_comparison'].to_string())
        
        print("\n--- Top 3 konfiguracje (Random Search) ---")
        print(self.results['random_search'].head(3)[
            ['n_hidden_layers', 'n_neurons', 'learning_rate', 'optimizer', 'activation',
             f'{metric_name}_mean']
        ].to_string())
    
    def save_results(self, filepath: str = 'mlp_results.json'):
        """Zapisanie wyników do pliku"""
        # Konwersja DataFrame do dict dla serializacji JSON
        results_to_save = {}
        for key, value in self.results.items():
            if isinstance(value, pd.DataFrame):
                results_to_save[key] = value.to_dict()
            else:
                results_to_save[key] = value
        
        with open(filepath, 'w') as f:
            json.dump(results_to_save, f, indent=2, default=str)
        
        print(f"\nWyniki zapisane do: {filepath}")


# def main():
#    """Główna funkcja uruchamiająca eksperymenty"""
    
#    print("\n" + "="*80)
#    print("DEEP MLP - KOMPLETNY SYSTEM EKSPERYMENTÓW")
#    print("="*80)
    
#    # ========== EKSPERYMENTY KLASYFIKACYJNE ==========
#    print("\n\n*** PROBLEM KLASYFIKACJI ***\n")
    
#    # Generowanie lub wczytanie danych klasyfikacyjnych
#    # X_class, y_class = generate_sample_data('classification', n_samples=1000, n_features=20)
    
#    # Eksperyment z wariantem B (70/15/15)
#    experiment_class = MLPExperiment(
#        task_type='classification',
#        split_variant='B',
#        n_repeats=5
#    )
    
#    results_class = experiment_class.run_full_experiment(X_class, y_class)
#    experiment_class.save_results('classification_results.json')
    
#    # ========== EKSPERYMENTY REGRESYJNE ==========
#    print("\n\n*** PROBLEM REGRESJI ***\n")
    
#    # Generowanie lub wczytanie danych regresyjnych
#    # X_reg, y_reg = generate_sample_data('regression', n_samples=1000, n_features=15)
    
#    # Eksperyment z wariantem B (70/15/15)
#    experiment_reg = MLPExperiment(
#        task_type='regression',
#        split_variant='B',
#        n_repeats=5
#    )
    
#    results_reg = experiment_reg.run_full_experiment(X_reg, y_reg)
#    experiment_reg.save_results('regression_results.json')
    
#    print("\n" + "="*80)
#    print("EKSPERYMENTY ZAKOŃCZONE")
#    print("="*80)


# if __name__ == "__main__":
#    main()