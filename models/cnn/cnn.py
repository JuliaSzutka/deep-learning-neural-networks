import warnings
warnings.filterwarnings('ignore')

# Podstawowe biblioteki
import pandas as pd
import numpy as np
import json
from datetime import datetime
import random
from math import exp
from itertools import product
from typing import List, Dict, Any
from multiprocessing import Pool, cpu_count

# Wizualizacja danych
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

# Przetwarzanie danych
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Sieci neuronowe i deep learning
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam, SGD, RMSprop
import tensorflow as tf
from tensorflow import keras
from keras import layers

class NumpyEncoder(json.JSONEncoder):
    """Encoder dla typów NumPy do serializacji JSON"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

class TimeSeriesCNN:
    """
    Sieć konwolucyjna do analizy szeregów czasowych.
    Zgodna z wytycznymi projektu: random search parametrów, wielokrotne powtórzenia, zapis do JSON.
    POPRAWIONA WERSJA: X jest normalizowane OSOBNO dla każdego zbioru (bez data leakage), y pozostaje w oryginalnej skali.
    DODANO: Multiprocessing support
    """
    
    def __init__(self, sequence_length=30):
        """
        Args:
            sequence_length: długość sekwencji wejściowej
        """
        self.sequence_length = sequence_length
        self.n_features = None
        self.scaler_X = StandardScaler()  # Tylko dla X
        self.final_model = None
        self.results = {
            'model_type': 'CNN',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'search_method': 'random_search',
            'experiments': [],
            'normalization_info': 'X normalized with StandardScaler (fit on train only), y kept in original scale'
        }
    
    def prepare_sequences(self, X_data, y_data):
        """
        Przygotowanie sekwencji dla sieci CNN.
        
        Args:
            X_data: array z cechami (już znormalizowane)
            y_data: array z targetem (nieznormalizowany!)
        
        Returns:
            X: sekwencje cech, shape (n_samples, sequence_length, n_features)
            y: wartości docelowe, shape (n_samples,) - w oryginalnej skali
        """
        X, y = [], []
        
        for i in range(len(X_data) - self.sequence_length):
            # Bierzemy sekwencję wszystkich cech
            X.append(X_data[i:i + self.sequence_length])
            # Target z następnego kroku - NIEZNORMALIZOWANY
            y.append(y_data[i + self.sequence_length])
        
        return np.array(X), np.array(y)
    
    def split_data(self, X, y, train_ratio=0.7, val_ratio=0.15):
        """Podział danych zgodnie z wytycznymi: 70% train, 15% val, 15% test"""
        n = len(X)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        X_train = X[:train_end]
        X_val = X[train_end:val_end]
        X_test = X[val_end:]
        
        y_train = y[:train_end]
        y_val = y[train_end:val_end]
        y_test = y[val_end:]
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def build_model(self, n_filters=64, kernel_size=3, n_conv_layers=2, 
                    dense_neurons=50, learning_rate=0.001, optimizer_type='adam',
                    use_momentum=False, momentum=0.9):
        """
        Budowa modelu CNN z parametryzacją zgodną z wymaganiami projektu.
        """
        model = keras.Sequential()
        
        kernel_size = int(kernel_size)
        
        # Obliczenie maksymalnej liczby warstw dla danego kernel_size
        current_size = self.sequence_length
        max_possible_layers = 0
        
        for _ in range(n_conv_layers):
            if current_size > kernel_size:
                current_size = (current_size - kernel_size + 1) // 2
                max_possible_layers += 1
            else:
                break
        
        actual_layers = min(n_conv_layers, max_possible_layers)
        
        if actual_layers == 0:
            raise ValueError(
                f"Niemożliwe zbudowanie modelu: sequence_length={self.sequence_length} "
                f"jest za małe dla kernel_size={kernel_size}. "
                f"Zwiększ sequence_length lub zmniejsz kernel_size."
            )
        
        if actual_layers < n_conv_layers:
            print(f"    Ostrzeżenie: Zmniejszono liczbę warstw z {n_conv_layers} do {actual_layers} (kernel_size={kernel_size})")
        
        # Warstwy konwolucyjne
        for i in range(actual_layers):
            if i == 0:
                model.add(layers.Conv1D(
                    filters=n_filters,
                    kernel_size=kernel_size,
                    activation='relu',
                    input_shape=(self.sequence_length, self.n_features)
                ))
            else:
                model.add(layers.Conv1D(
                    filters=n_filters * (i + 1),
                    kernel_size=kernel_size,
                    activation='relu'
                ))
            model.add(layers.MaxPooling1D(pool_size=2))
            model.add(layers.Dropout(0.2))
        
        # Spłaszczenie
        model.add(layers.Flatten())
        
        # Warstwa Dense
        model.add(layers.Dense(dense_neurons, activation='relu'))
        model.add(layers.Dropout(0.2))
        
        # Warstwa wyjściowa
        model.add(layers.Dense(1))
        
        # Optymalizator
        if optimizer_type == 'adam':
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_type == 'sgd':
            if use_momentum:
                optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
            else:
                optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
        elif optimizer_type == 'rmsprop':
            optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model
    
    def calculate_metrics(self, y_true, y_pred):
        """
        Obliczanie miar jakości: MSE, MAE, R², RMSE
        y_true i y_pred są w ORYGINALNEJ SKALI (nieznormalizowane)
        """
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        return {
            'MSE': float(mse),
            'MAE': float(mae),
            'R2': float(r2),
            'RMSE': float(np.sqrt(mse))
        }
    
    def train_single_repeat(self, args):
        """
        Funkcja do trenowania pojedynczego powtórzenia (dla multiprocessingu).
        
        Args:
            args: tuple (repeat_num, X_train, y_train, X_val, y_val, X_test, y_test, params, epochs, batch_size)
        
        Returns:
            dict z wynikami powtórzenia
        """
        repeat, X_train, y_train, X_val, y_val, X_test, y_test, params, epochs, batch_size = args
        
        try:
            # Budowa modelu
            model = self.build_model(**params)
            
            # Early stopping
            early_stop = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            # Trening - y jest w oryginalnej skali
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stop],
                verbose=0
            )
            
            # Predykcje - będą w oryginalnej skali
            train_pred = model.predict(X_train, verbose=0).flatten()
            val_pred = model.predict(X_val, verbose=0).flatten()
            test_pred = model.predict(X_test, verbose=0).flatten()
            
            # Metryki - wszystko w oryginalnej skali
            repeat_result = {
                'repeat': repeat + 1,
                'train_metrics': self.calculate_metrics(y_train, train_pred),
                'val_metrics': self.calculate_metrics(y_val, val_pred),
                'test_metrics': self.calculate_metrics(y_test, test_pred),
                'epochs_trained': len(history.history['loss'])
            }
            
            return repeat_result
            
        except ValueError as e:
            print(f"BŁĄD w powtórzeniu {repeat + 1}: {e}")
            return None
    
    def train_and_evaluate(self, X_train, y_train, X_val, y_val, X_test, y_test,
                          params, n_repeats=5, epochs=50, batch_size=32,
                          use_multiprocessing=False, n_jobs=-1):
        """
        Trening i ewaluacja z wielokrotnym powtórzeniem (min. 5 powtórzeń zgodnie z wytycznymi)
        y_train, y_val, y_test są w ORYGINALNEJ SKALI
        
        Args:
            use_multiprocessing: czy używać multiprocessingu
            n_jobs: liczba procesów (-1 = wszystkie CPU, 1 = sekwencyjnie, >1 = konkretna liczba)
        """
        repeat_results = []
        
        if use_multiprocessing and n_repeats > 1:
            # Przygotowanie argumentów dla multiprocessingu
            args_list = [
                (i, X_train, y_train, X_val, y_val, X_test, y_test, params, epochs, batch_size)
                for i in range(n_repeats)
            ]
            
            # Ustalenie liczby procesów
            if n_jobs == -1:
                n_processes = min(cpu_count(), n_repeats)
            else:
                n_processes = min(max(1, n_jobs), cpu_count(), n_repeats)
            
            print(f"    Używam {n_processes} procesów dla {n_repeats} powtórzeń (dostępne CPU: {cpu_count()})")
            
            with Pool(processes=n_processes) as pool:
                repeat_results = pool.map(self.train_single_repeat, args_list)
            
            # Filtruj None (błędne powtórzenia)
            repeat_results = [r for r in repeat_results if r is not None]
            
            if not repeat_results:
                print("    ⚠️  Wszystkie powtórzenia zakończyły się błędem")
                return None
            
            # Wyświetl wyniki
            for result in repeat_results:
                print(f"    Powtórzenie {result['repeat']}/{n_repeats} - Test MSE: {result['test_metrics']['MSE']:.8f}")
        else:
            # Sekwencyjny trening (standardowy)
            for repeat in range(n_repeats):
                print(f"    Powtórzenie {repeat + 1}/{n_repeats}", end=" ")
                
                try:
                    # Budowa modelu
                    model = self.build_model(**params)
                    
                    # Early stopping
                    early_stop = keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=10,
                        restore_best_weights=True
                    )
                    
                    # Trening - y jest w oryginalnej skali
                    history = model.fit(
                        X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[early_stop],
                        verbose=0
                    )
                    
                    # Predykcje - będą w oryginalnej skali
                    train_pred = model.predict(X_train, verbose=0).flatten()
                    val_pred = model.predict(X_val, verbose=0).flatten()
                    test_pred = model.predict(X_test, verbose=0).flatten()
                    
                    # Metryki - wszystko w oryginalnej skali
                    repeat_result = {
                        'repeat': repeat + 1,
                        'train_metrics': self.calculate_metrics(y_train, train_pred),
                        'val_metrics': self.calculate_metrics(y_val, val_pred),
                        'test_metrics': self.calculate_metrics(y_test, test_pred),
                        'epochs_trained': len(history.history['loss'])
                    }
                    
                    repeat_results.append(repeat_result)
                    print(f"Test MSE: {repeat_result['test_metrics']['MSE']:.8f}")
                    
                except ValueError as e:
                    print(f"BŁĄD: {e}")
                    continue
        
        if not repeat_results:
            return None
        
        # Obliczanie średnich wyników
        avg_result = self._calculate_average_results(repeat_results)
        
        return {
            'parameters': params,
            'repeats': repeat_results,
            'average': avg_result
        }
    
    def _calculate_average_results(self, repeat_results):
        """Obliczanie średnich wyników z powtórzeń"""
        metrics = ['MSE', 'MAE', 'R2', 'RMSE']
        sets = ['train_metrics', 'val_metrics', 'test_metrics']
        
        avg = {}
        for set_name in sets:
            avg[set_name] = {}
            for metric in metrics:
                values = [r[set_name][metric] for r in repeat_results]
                avg[set_name][metric] = float(np.mean(values))
                avg[set_name][f'{metric}_std'] = float(np.std(values))
        
        return avg
    
    def _find_best_result(self, repeat_results):
        """Znajdowanie najlepszego wyniku (najniższe MSE na zbiorze walidacyjnym)"""
        best_idx = np.argmin([r['val_metrics']['MSE'] for r in repeat_results])
        return repeat_results[best_idx]
    
    def train_best_model(self, X_train, y_train, X_val, y_val, X_test, y_test, 
                        best_params, epochs=100, batch_size=32):
        """
        Trening najlepszego modelu na połączonych danych train+val, 
        ewaluacja na test.
        """
        print(f"\n{'='*70}")
        print("TRENING FINALNEGO MODELU NA TRAIN+VAL")
        print(f"{'='*70}")
        print(f"Parametry: {best_params}")
        
        # Połączenie train i val
        X_train_full = np.concatenate([X_train, X_val], axis=0)
        y_train_full = np.concatenate([y_train, y_val], axis=0)
        
        print(f"\nRozmiar danych treningowych: {X_train_full.shape}")
        print(f"Rozmiar danych testowych: {X_test.shape}")
        
        # Budowa modelu
        model = self.build_model(**best_params)
        
        # Early stopping na zbiorze testowym (tylko monitoring, nie używamy do wyboru wag)
        early_stop = keras.callbacks.EarlyStopping(
            monitor='loss',  # Monitorujemy loss treningowy
            patience=15,
            restore_best_weights=True
        )
        
        # Trening
        print("\nTrening w toku...")
        history = model.fit(
            X_train_full, y_train_full,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=1
        )
        
        # Predykcje
        print("\nObliczanie predykcji...")
        train_pred = model.predict(X_train_full, verbose=0).flatten()
        test_pred = model.predict(X_test, verbose=0).flatten()
        
        # Metryki
        train_metrics = self.calculate_metrics(y_train_full, train_pred)
        test_metrics = self.calculate_metrics(y_test, test_pred)
        
        final_model_results = {
            'parameters': best_params,
            'training_info': {
                'train_samples': int(len(X_train_full)),
                'test_samples': int(len(X_test)),
                'epochs_trained': len(history.history['loss']),
                'final_train_loss': float(history.history['loss'][-1])
            },
            'train_metrics': train_metrics,
            'test_metrics': test_metrics
        }
        
        print(f"\n{'='*70}")
        print("WYNIKI FINALNEGO MODELU:")
        print(f"{'='*70}")
        print(f"Epochs trained: {len(history.history['loss'])}")
        print(f"\nTrain metrics:")
        print(f"  MSE:  {train_metrics['MSE']:.8f}")
        print(f"  MAE:  {train_metrics['MAE']:.8f}")
        print(f"  RMSE: {train_metrics['RMSE']:.8f}")
        print(f"  R2:   {train_metrics['R2']:.6f}")
        print(f"\nTest metrics:")
        print(f"  MSE:  {test_metrics['MSE']:.8f}")
        print(f"  MAE:  {test_metrics['MAE']:.8f}")
        print(f"  RMSE: {test_metrics['RMSE']:.8f}")
        print(f"  R2:   {test_metrics['R2']:.6f}")
        
        return model, final_model_results
    
    def generate_random_search_configs(self, n_configs=20):
        """
        Generowanie konfiguracji parametrów za pomocą Random Search.
        """
        
        param_space = {
            'n_conv_layers': [1, 2, 3, 4],
            'n_filters': [5, 10, 15, 30],
            'kernel_size': [3, 5, 7],
            'dense_neurons': [20, 40, 60, 80],
            'learning_rate': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05],
            'optimizer_type': ['adam', 'sgd', 'rmsprop'],
            'momentum': [0.5, 0.7, 0.9, 0.95, 0.99]
        }
        
        configs = []
        
        for i in range(n_configs):
            config = {
                'n_conv_layers': int(np.random.choice(param_space['n_conv_layers'])),
                'n_filters': int(np.random.choice(param_space['n_filters'])),
                'kernel_size': int(np.random.choice(param_space['kernel_size'])),
                'dense_neurons': int(np.random.choice(param_space['dense_neurons'])),
                'learning_rate': float(np.random.choice(param_space['learning_rate'])),
                'optimizer_type': str(np.random.choice(param_space['optimizer_type']))
            }
            
            if config['optimizer_type'] == 'sgd':
                config['use_momentum'] = bool(np.random.choice([True, False]))
                if config['use_momentum']:
                    config['momentum'] = float(np.random.choice(param_space['momentum']))
                else:
                    config['use_momentum'] = False
            else:
                config['use_momentum'] = False
            
            configs.append(config)
        
        return configs
    
    def run_random_search(self, X, y, n_configs=20, n_repeats=5, epochs=30, 
                         batch_size=32, use_multiprocessing=False, n_jobs=-1):
        """
        Przeprowadzenie Random Search zgodnie z wytycznymi projektu.
        X - features (będą znormalizowane)
        y - target (pozostanie w oryginalnej skali)
        
        Args:
            use_multiprocessing: czy używać multiprocessingu
            n_jobs: liczba procesów (-1 = wszystkie CPU, 1 = sekwencyjnie, >1 = konkretna liczba)
        """
        # Konwersja do numpy arrays
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        if y.ndim > 1:
            y = y.flatten()
        
        # Walidacja danych
        min_samples = self.sequence_length + 100
        if X.shape[0] < min_samples:
            raise ValueError(
                f"Za mało danych! Potrzeba minimum {min_samples} próbek, "
                f"masz {X.shape[0]}. Zmniejsz sequence_length lub dodaj więcej danych."
            )
        
        if X.shape[0] != len(y):
            raise ValueError(f"X ma {X.shape[0]} próbek, y ma {len(y)} próbek - muszą się zgadzać!")
        
        # Sprawdzenie NaN/Inf
        if np.isnan(X).any() or np.isinf(X).any():
            raise ValueError("X zawiera wartości NaN lub Inf! Wyczyść dane przed treningiem.")
        if np.isnan(y).any() or np.isinf(y).any():
            raise ValueError("y zawiera wartości NaN lub Inf! Wyczyść dane przed treningiem.")
        
        print(f"\n{'='*70}")
        print("CNN - RANDOM SEARCH")
        print(f"{'='*70}")
        if use_multiprocessing:
            if n_jobs == -1:
                print(f"Multiprocessing: TAK (wszystkie dostępne CPU: {cpu_count()})")
            else:
                print(f"Multiprocessing: TAK ({n_jobs} procesów)")
        else:
            print(f"Multiprocessing: NIE")
        
        print(f"\n{'='*70}")
        print("INFORMACJE O DANYCH:")
        print(f"{'='*70}")
        print(f"Całkowita liczba próbek: {X.shape[0]}")
        print(f"Liczba features w X: {X.shape[1]}")
        print(f"Długość sekwencji: {self.sequence_length}")
        print(f"Statystyki y (oryginalnego):")
        print(f"  Min: {y.min():.6f}")
        print(f"  Max: {y.max():.6f}")
        print(f"  Mean: {y.mean():.6f}")
        print(f"  Std: {y.std():.6f}")
        
        # POPRAWKA: Najpierw podział danych, POTEM normalizacja (aby uniknąć data leakage)
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
        
        print(f"\nRozmiary zbiorów po podziale (PRZED normalizacją):")
        print(f"  Train: X={X_train.shape}, y={y_train.shape}")
        print(f"  Val:   X={X_val.shape}, y={y_val.shape}")
        print(f"  Test:  X={X_test.shape}, y={y_test.shape}")
        
        # Normalizacja TYLKO X (nie y!) - fit TYLKO na train, transform na wszystkie
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        X_val_scaled = self.scaler_X.transform(X_val)
        X_test_scaled = self.scaler_X.transform(X_test)
        
        print(f"\nNormalizacja wykonana (fit na train, transform na val i test)")
        
        # Przygotowanie sekwencji - y pozostaje nieznormalizowane
        X_train_seq, y_train_seq = self.prepare_sequences(X_train_scaled, y_train)
        X_val_seq, y_val_seq = self.prepare_sequences(X_val_scaled, y_val)
        X_test_seq, y_test_seq = self.prepare_sequences(X_test_scaled, y_test)
        
        # Ustawienie liczby cech
        self.n_features = X_train_seq.shape[2]
        
        print(f"\nRozmiary zbiorów po utworzeniu sekwencji:")
        print(f"  Train: X={X_train_seq.shape}, y={y_train_seq.shape}")
        print(f"  Val:   X={X_val_seq.shape}, y={y_val_seq.shape}")
        print(f"  Test:  X={X_test_seq.shape}, y={y_test_seq.shape}")
        print(f"\nLiczba cech w sekwencjach: {self.n_features}")
        print(f"\nStatystyki y_test (do predykcji - ORYGINALNA SKALA):")
        print(f"  Min: {y_test_seq.min():.6f}")
        print(f"  Max: {y_test_seq.max():.6f}")
        print(f"  Mean: {y_test_seq.mean():.6f}")
        print(f"  Std: {y_test_seq.std():.6f}")
        
        print(f"\nGenerowanie {n_configs} losowych konfiguracji...")
        
        # Generowanie konfiguracji
        configs = self.generate_random_search_configs(n_configs)
        
        # Zapisanie informacji
        self.results['search_space'] = {
            'n_conv_layers': [1, 2, 3, 4],
            'n_filters': [5, 10, 15, 30],
            'kernel_size': [3, 5, 7],
            'dense_neurons': [20, 40, 60, 80],
            'learning_rate': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05],
            'optimizer_type': ['adam', 'sgd', 'rmsprop'],
            'momentum': [0.5, 0.7, 0.9, 0.95, 0.99]
        }
        self.results['data_info'] = {
            'sequence_length': self.sequence_length,
            'n_features': self.n_features,
            'train_samples': int(len(X_train_seq)),
            'val_samples': int(len(X_val_seq)),
            'test_samples': int(len(X_test_seq)),
            'y_statistics': {
                'min': float(y.min()),
                'max': float(y.max()),
                'mean': float(y.mean()),
                'std': float(y.std())
            }
        }
        self.results['n_configurations'] = n_configs
        self.results['n_repeats_per_config'] = n_repeats
        
        # Przeprowadzenie eksperymentów
        for i, config in enumerate(configs):
            print(f"\n{'='*70}")
            print(f"Konfiguracja {i + 1}/{n_configs}")
            print(f"{'='*70}")
            print(f"  Parametry: {config}")
            
            result = self.train_and_evaluate(
                X_train_seq, y_train_seq, X_val_seq, y_val_seq, X_test_seq, y_test_seq,
                config,
                n_repeats=n_repeats,
                epochs=epochs,
                batch_size=batch_size,
                use_multiprocessing=use_multiprocessing,
                n_jobs=n_jobs
            )
            
            if result is None:
                print(f"  ⚠️  Pominięto konfigurację {i + 1} z powodu błędu")
                continue
            
            result['config_id'] = i + 1
            self.results['experiments'].append(result)
            
            print(f"\n  Wyniki (średnie - ORYGINALNA SKALA y):")
            print(f"    Train MSE: {result['average']['train_metrics']['MSE']:.8f} ± {result['average']['train_metrics']['MSE_std']:.8f}")
            print(f"    Val   MSE: {result['average']['val_metrics']['MSE']:.8f} ± {result['average']['val_metrics']['MSE_std']:.8f}")
            print(f"    Test  MSE: {result['average']['test_metrics']['MSE']:.8f} ± {result['average']['test_metrics']['MSE_std']:.8f}")
            print(f"    Test  MAE: {result['average']['test_metrics']['MAE']:.8f}")
            print(f"    Test  R2:  {result['average']['test_metrics']['R2']:.6f}")
        
        # Znajdź najlepszą konfigurację
        if len(self.results['experiments']) > 0:
            best_config = self._find_best_configuration()
            
            # Teraz wytrenuj finalny model na najlepszej konfiguracji
            print(f"\n{'='*70}")
            print("TRENING FINALNEGO MODELU")
            print(f"{'='*70}")
            
            final_model, final_results = self.train_best_model(
                X_train_seq, y_train_seq, 
                X_val_seq, y_val_seq, 
                X_test_seq, y_test_seq,
                best_config['parameters'],
                epochs=100,
                batch_size=batch_size
            )
            
            self.results['final_model'] = final_results
            self.final_model = final_model
            
        else:
            print("\n⚠️  UWAGA: Żadna konfiguracja nie zadziałała!")
    
    def _find_best_configuration(self):
        """Znajdowanie najlepszej konfiguracji na podstawie wyników walidacyjnych"""
        best_idx = np.argmin([exp['average']['val_metrics']['MSE'] 
                              for exp in self.results['experiments']])
        
        best_config = {
            'config_id': self.results['experiments'][best_idx]['config_id'],
            'parameters': self.results['experiments'][best_idx]['parameters'],
            'average_results': self.results['experiments'][best_idx]['average']
        }
        
        self.results['best_configuration'] = best_config
        
        print(f"\n{'='*70}")
        print("NAJLEPSZA KONFIGURACJA (ŚREDNIE WYNIKI Z RANDOM SEARCH):")
        print(f"{'='*70}")
        print(f"ID: {best_config['config_id']}")
        print(f"Parametry: {best_config['parameters']}")
        print(f"\nWyniki średnie (z {self.results['n_repeats_per_config']} powtórzeń):")
        print(f"  Val MSE:  {best_config['average_results']['val_metrics']['MSE']:.8f} ± {best_config['average_results']['val_metrics']['MSE_std']:.8f}")
        print(f"  Test MSE: {best_config['average_results']['test_metrics']['MSE']:.8f} ± {best_config['average_results']['test_metrics']['MSE_std']:.8f}")
        print(f"  Test MAE: {best_config['average_results']['test_metrics']['MAE']:.8f} ± {best_config['average_results']['test_metrics']['MAE_std']:.8f}")
        print(f"  Test R2:  {best_config['average_results']['test_metrics']['R2']:.6f} ± {best_config['average_results']['test_metrics']['R2_std']:.6f}")
        
        return best_config
    
    def save_results(self, filename='cnn_random_search_results.json'):
        """Zapis wyników do JSON z obsługą typów NumPy"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        print(f"\n✅ Wyniki zapisane do pliku: {filename}")


# ============================================================================
# PRZYKŁAD UŻYCIA
# ============================================================================

def prepare_and_run_cnn(X, y, sequence_length=30, n_configs=20, n_repeats=5,
                       use_multiprocessing=False, n_jobs=-1):
    """
    Pomocnicza funkcja do przygotowania danych i uruchomienia CNN.
    
    Args:
        X: DataFrame lub array z cechami (będą znormalizowane)
        y: Series lub array z targetem (pozostanie w oryginalnej skali)
        sequence_length: długość sekwencji (domyślnie 30)
        n_configs: liczba konfiguracji do przetestowania
        n_repeats: liczba powtórzeń dla każdej konfiguracji
        use_multiprocessing: czy używać multiprocessingu (domyślnie False)
        n_jobs: liczba procesów (-1 = wszystkie CPU, 1 = sekwencyjnie, >1 = konkretna liczba)
    
    Examples:
        # Bez multiprocessingu
        cnn = prepare_and_run_cnn(X, y, n_configs=10, n_repeats=5)
        
        # Wszystkie dostępne CPU
        cnn = prepare_and_run_cnn(X, y, n_configs=10, n_repeats=5, 
                                  use_multiprocessing=True, n_jobs=-1)
        
        # Konkretna liczba procesów (np. 4)
        cnn = prepare_and_run_cnn(X, y, n_configs=10, n_repeats=5, 
                                  use_multiprocessing=True, n_jobs=4)
    """
    
    print(f"\n{'='*70}")
    print("START ANALIZY CNN")
    print(f"{'='*70}")
    
    # Inicjalizacja modelu
    cnn_model = TimeSeriesCNN(sequence_length=sequence_length)
    
    # Uruchomienie Random Search
    cnn_model.run_random_search(
        X=X,
        y=y,
        n_configs=n_configs,
        n_repeats=n_repeats,
        epochs=40,
        batch_size=32,
        use_multiprocessing=use_multiprocessing,
        n_jobs=n_jobs
    )
    
    # Zapis wyników
    cnn_model.save_results('cnn_random_search_results2.json')
    
    print("\n✅ Zakończono!")
    
    return cnn_model