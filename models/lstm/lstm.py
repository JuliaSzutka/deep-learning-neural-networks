import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow import keras
from keras import layers
from datetime import datetime
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')


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


class LSTMTimeSeriesModel:
    """
    POPRAWIONA implementacja LSTM dla szeregów czasowych.
    Używa prawdziwych sekwencji czasowych (np. 30 kroków) zamiast 1 timestep.
    DODANO: trening finalnego modelu + multiprocessing
    """
    
    def __init__(self, sequence_length=30):
        """
        Args:
            sequence_length: długość sekwencji wejściowej (domyślnie 30, jak w CNN)
        """
        self.sequence_length = sequence_length
        self.n_features = None
        self.scaler_X = StandardScaler()
        self.final_model = None
        self.results = {
            'model_type': 'LSTM_CORRECTED',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'search_method': 'random_search',
            'sequence_length': sequence_length,
            'aggregated_results': [],
            'best_models': {},
            'normalization_info': 'X normalized with StandardScaler, y kept in original scale'
        }
    
    def prepare_sequences(self, X_data, y_data):
        """
        Przygotowanie sekwencji dla LSTM.
        
        Args:
            X_data: array z cechami (już znormalizowane)
            y_data: array z targetem (nieznormalizowany)
        
        Returns:
            X: sekwencje cech, shape (n_samples, sequence_length, n_features)
            y: wartości docelowe, shape (n_samples,)
        """
        X, y = [], []
        for i in range(len(X_data) - self.sequence_length):
            X.append(X_data[i:i + self.sequence_length])
            y.append(y_data[i + self.sequence_length])
        
        return np.array(X), np.array(y)
    
    def split_data(self, X, y, train_ratio=0.7, val_ratio=0.15):
        """Podział danych: 70% train, 15% val, 15% test"""
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
    
    def build_model(self, num_layers, lstm_units_layer1, lstm_units_layer2,
                    learning_rate, optimizer, dropout_rate, batch_size):
        """
        Budowa modelu LSTM.
        
        Args:
            num_layers: liczba warstw LSTM (1 lub 2)
            lstm_units_layer1: liczba jednostek w pierwszej warstwie LSTM
            lstm_units_layer2: liczba jednostek w drugiej warstwie LSTM (używane tylko gdy num_layers=2)
            learning_rate: learning rate
            optimizer: typ optymalizatora ('adam', 'rmsprop', 'sgd')
            dropout_rate: dropout rate
            batch_size: batch size (nie używane w build, ale potrzebne do zapisu parametrów)
        
        Returns:
            model: skompilowany model Keras
        """
        model = keras.Sequential()
        
        if num_layers == 1:
            # Jedna warstwa LSTM
            model.add(layers.LSTM(
                units=lstm_units_layer1,
                input_shape=(self.sequence_length, self.n_features)
            ))
        else:
            # Dwie warstwy LSTM
            model.add(layers.LSTM(
                units=lstm_units_layer1,
                return_sequences=True,  # Zwracamy sekwencje dla następnej warstwy LSTM
                input_shape=(self.sequence_length, self.n_features)
            ))
            model.add(layers.Dropout(dropout_rate))
            model.add(layers.LSTM(units=lstm_units_layer2))
        
        model.add(layers.Dropout(dropout_rate))
        model.add(layers.Dense(1))
        
        # Wybór optymalizatora
        if optimizer == 'adam':
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'rmsprop':
            opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            opt = keras.optimizers.SGD(learning_rate=learning_rate)
        else:
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
        
        model.compile(optimizer=opt, loss='mse', metrics=['mae'])
        
        return model
    
    def calculate_metrics(self, y_true, y_pred):
        """Obliczanie miar jakości"""
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mse)
        
        return {
            'MSE': float(mse),
            'RMSE': float(rmse),
            'MAE': float(mae),
            'R2': float(r2)
        }
    
    def train_single_repeat(self, args):
        """
        Funkcja do trenowania pojedynczego powtórzenia (dla multiprocessingu).
        
        Args:
            args: tuple (repeat_num, X_train, y_train, X_val, y_val, X_test, y_test, params, epochs)
        
        Returns:
            dict z wynikami powtórzenia
        """
        repeat, X_train, y_train, X_val, y_val, X_test, y_test, params, epochs = args
        
        # Budowa modelu
        model = self.build_model(**params)
        
        # Early stopping
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Trening
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=params['batch_size'],
            callbacks=[early_stop],
            verbose=0
        )
        
        # Predykcje
        train_pred = model.predict(X_train, verbose=0).flatten()
        val_pred = model.predict(X_val, verbose=0).flatten()
        test_pred = model.predict(X_test, verbose=0).flatten()
        
        # Metryki
        repeat_result = {
            'repeat': repeat + 1,
            'train_metrics': self.calculate_metrics(y_train, train_pred),
            'val_metrics': self.calculate_metrics(y_val, val_pred),
            'test_metrics': self.calculate_metrics(y_test, test_pred),
            'epochs_trained': len(history.history['loss'])
        }
        
        return repeat_result
    
    def train_and_evaluate(self, X_train, y_train, X_val, y_val, X_test, y_test,
                          params, n_repeats=5, epochs=50, use_multiprocessing=False, n_jobs=-1):
        """
        Trening i ewaluacja z wielokrotnym powtórzeniem.
        
        Args:
            params: słownik z parametrami modelu
            n_repeats: liczba powtórzeń
            epochs: liczba epok
            use_multiprocessing: czy używać multiprocessingu
            n_jobs: liczba procesów (-1 = wszystkie dostępne CPU, 1 = bez multiprocessingu)
        
        Returns:
            dict z wynikami
        """
        repeat_results = []
        
        if use_multiprocessing and n_repeats > 1:
            # Przygotowanie argumentów dla multiprocessingu
            args_list = [
                (i, X_train, y_train, X_val, y_val, X_test, y_test, params, epochs)
                for i in range(n_repeats)
            ]
            
            # Ustalenie liczby procesów
            if n_jobs == -1:
                n_processes = min(cpu_count(), n_repeats)
            else:
                n_processes = min(max(1, n_jobs), cpu_count(), n_repeats)
            
            print(f"  Używam {n_processes} procesów dla {n_repeats} powtórzeń (dostępne CPU: {cpu_count()})")
            
            with Pool(processes=n_processes) as pool:
                repeat_results = pool.map(self.train_single_repeat, args_list)
            
            # Wyświetl wyniki
            for result in repeat_results:
                print(f"  Powtórzenie {result['repeat']}/{n_repeats} - Test RMSE: {result['test_metrics']['RMSE']:.6f}")
        else:
            # Sekwencyjny trening (standardowy)
            for repeat in range(n_repeats):
                print(f"  Powtórzenie {repeat + 1}/{n_repeats}", end=" ")
                
                # Budowa modelu
                model = self.build_model(**params)
                
                # Early stopping
                early_stop = keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                )
                
                # Trening
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=epochs,
                    batch_size=params['batch_size'],
                    callbacks=[early_stop],
                    verbose=0
                )
                
                # Predykcje
                train_pred = model.predict(X_train, verbose=0).flatten()
                val_pred = model.predict(X_val, verbose=0).flatten()
                test_pred = model.predict(X_test, verbose=0).flatten()
                
                # Metryki
                repeat_result = {
                    'repeat': repeat + 1,
                    'train_metrics': self.calculate_metrics(y_train, train_pred),
                    'val_metrics': self.calculate_metrics(y_val, val_pred),
                    'test_metrics': self.calculate_metrics(y_test, test_pred),
                    'epochs_trained': len(history.history['loss'])
                }
                repeat_results.append(repeat_result)
                
                print(f"Test RMSE: {repeat_result['test_metrics']['RMSE']:.6f}")
        
        # Agregacja wyników
        aggregated = self._aggregate_results(repeat_results)
        
        return {
            'parameters': params,
            'aggregated_results': aggregated,
            'individual_repeats': repeat_results
        }
    
    def _aggregate_results(self, repeat_results):
        """Agregacja wyników z powtórzeń"""
        metrics = ['MSE', 'RMSE', 'MAE', 'R2']
        sets = ['train', 'val', 'test']
        
        aggregated = {}
        for set_name in sets:
            aggregated[set_name] = {}
            for metric in metrics:
                values = [r[f'{set_name}_metrics'][metric] for r in repeat_results]
                aggregated[set_name][f'{metric}_mean'] = float(np.mean(values))
                aggregated[set_name][f'{metric}_std'] = float(np.std(values))
        
        return aggregated
    
    def generate_random_configs(self, n_configs=20):
        """Generowanie losowych konfiguracji dla Random Search"""
        configs = []
        
        for _ in range(n_configs):
            config = {
                'num_layers': int(np.random.choice([1, 2])),
                'lstm_units_layer1': int(np.random.choice([16, 32, 64, 128])),
                'lstm_units_layer2': int(np.random.choice([16, 32, 64, 128])),
                'learning_rate': float(np.random.choice([0.0001, 0.001, 0.01, 0.1])),
                'optimizer': str(np.random.choice(['adam', 'rmsprop', 'sgd'])),
                'dropout_rate': float(np.random.choice([0.0, 0.1, 0.2, 0.3])),
                'batch_size': int(np.random.choice([8, 16, 32, 64]))
            }
            configs.append(config)
        
        return configs
    
    def train_best_model(self, X_train, y_train, X_val, y_val, X_test, y_test, 
                        best_params, epochs=100):
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
        
        # Early stopping na loss treningowym
        early_stop = keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=15,
            restore_best_weights=True
        )
        
        # Trening
        print("\nTrening w toku...")
        history = model.fit(
            X_train_full, y_train_full,
            epochs=epochs,
            batch_size=best_params['batch_size'],
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
    
    def run_random_search(self, X, y, n_configs=20, n_repeats=5, epochs=50, 
                         use_multiprocessing=False, n_jobs=-1):
        """
        Przeprowadzenie Random Search.
        
        Args:
            X: DataFrame lub array z cechami
            y: Series lub array z targetem
            n_configs: liczba konfiguracji do przetestowania
            n_repeats: liczba powtórzeń dla każdej konfiguracji
            epochs: liczba epok treningu
            use_multiprocessing: czy używać multiprocessingu (domyślnie False)
            n_jobs: liczba procesów (-1 = wszystkie CPU, 1 = sekwencyjnie, >1 = konkretna liczba)
        """
        print(f"\n{'='*70}")
        print("LSTM - RANDOM SEARCH Z PRAWDZIWYMI SEKWENCJAMI CZASOWYMI")
        print(f"{'='*70}")
        print(f"Długość sekwencji: {self.sequence_length} kroków czasowych")
        if use_multiprocessing:
            if n_jobs == -1:
                print(f"Multiprocessing: TAK (wszystkie dostępne CPU: {cpu_count()})")
            else:
                print(f"Multiprocessing: TAK ({n_jobs} procesów)")
        else:
            print(f"Multiprocessing: NIE")
        
        # Konwersja do numpy
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        if y.ndim > 1:
            y = y.flatten()
        
        # Walidacja
        min_samples = self.sequence_length + 100
        if X.shape[0] < min_samples:
            raise ValueError(f"Za mało danych! Potrzeba min {min_samples}, masz {X.shape[0]}")
        
        print(f"\nDane wejściowe:")
        print(f"  Liczba próbek: {X.shape[0]}")
        print(f"  Liczba cech: {X.shape[1]}")
        print(f"Statystyki y (oryginalnego):")
        print(f"  Min: {y.min():.6f}")
        print(f"  Max: {y.max():.6f}")
        print(f"  Mean: {y.mean():.6f}")
        print(f"  Std: {y.std():.6f}")
        
        # Podział danych PRZED normalizacją
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
        
        print(f"\nPodział danych:")
        print(f"  Train: {X_train.shape[0]} próbek")
        print(f"  Val: {X_val.shape[0]} próbek")
        print(f"  Test: {X_test.shape[0]} próbek")
        
        # Normalizacja TYLKO X
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        X_val_scaled = self.scaler_X.transform(X_val)
        X_test_scaled = self.scaler_X.transform(X_test)
        
        # Przygotowanie sekwencji
        X_train_seq, y_train_seq = self.prepare_sequences(X_train_scaled, y_train)
        X_val_seq, y_val_seq = self.prepare_sequences(X_val_scaled, y_val)
        X_test_seq, y_test_seq = self.prepare_sequences(X_test_scaled, y_test)
        
        self.n_features = X_train_seq.shape[2]
        
        print(f"\nPo utworzeniu sekwencji:")
        print(f"  Train: {X_train_seq.shape} -> {y_train_seq.shape}")
        print(f"  Val: {X_val_seq.shape} -> {y_val_seq.shape}")
        print(f"  Test: {X_test_seq.shape} -> {y_test_seq.shape}")
        print(f"  Każda próbka = {self.sequence_length} kroków × {self.n_features} cech")
        
        # Zapisanie informacji o danych
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
        
        # Generowanie konfiguracji
        configs = self.generate_random_configs(n_configs)
        
        print(f"\n{'='*70}")
        print(f"Testowanie {n_configs} konfiguracji (każda {n_repeats}× powtórzeń)")
        print(f"{'='*70}\n")
        
        # Testowanie konfiguracji
        for i, config in enumerate(configs):
            print(f"Konfiguracja {i + 1}/{n_configs}")
            print(f"  Parametry: {config}")
            
            result = self.train_and_evaluate(
                X_train_seq, y_train_seq,
                X_val_seq, y_val_seq,
                X_test_seq, y_test_seq,
                config,
                n_repeats=n_repeats,
                epochs=epochs,
                use_multiprocessing=use_multiprocessing,
                n_jobs=n_jobs
            )
            
            result['config_id'] = i + 1
            self.results['aggregated_results'].append(result)
            
            avg = result['aggregated_results']
            print(f"  Wyniki (średnie):")
            print(f"    Test RMSE: {avg['test']['RMSE_mean']:.6f} ± {avg['test']['RMSE_std']:.6f}")
            print(f"    Test R²: {avg['test']['R2_mean']:.4f} ± {avg['test']['R2_std']:.4f}\n")
        
        # Znajdź najlepszy model
        self._find_best_models()
        
        print(f"\n{'='*70}")
        print("NAJLEPSZY MODEL Z RANDOM SEARCH (wg RMSE)")
        print(f"{'='*70}")
        best = self.results['best_models']['best_rmse']
        print(f"Parametry: {best['parameters']}")
        print(f"Test RMSE: {best['aggregated_results']['test']['RMSE_mean']:.6f}")
        print(f"Test R²: {best['aggregated_results']['test']['R2_mean']:.4f}")
        
        # Trening finalnego modelu na najlepszych parametrach
        print(f"\n{'='*70}")
        print("TRENING FINALNEGO MODELU")
        print(f"{'='*70}")
        
        final_model, final_results = self.train_best_model(
            X_train_seq, y_train_seq,
            X_val_seq, y_val_seq,
            X_test_seq, y_test_seq,
            best['parameters'],
            epochs=100
        )
        
        self.results['final_model'] = final_results
        self.final_model = final_model
    
    def _find_best_models(self):
        """Znajdź najlepsze modele według różnych metryk"""
        results = self.results['aggregated_results']
        
        # Najlepszy wg RMSE
        best_rmse_idx = np.argmin([r['aggregated_results']['test']['RMSE_mean'] for r in results])
        self.results['best_models']['best_rmse'] = results[best_rmse_idx]
        
        # Najlepszy wg R²
        best_r2_idx = np.argmax([r['aggregated_results']['test']['R2_mean'] for r in results])
        self.results['best_models']['best_r2'] = results[best_r2_idx]
        
        # Zapisanie info o najlepszej konfiguracji
        self.results['best_configuration'] = {
            'config_id': results[best_rmse_idx]['config_id'],
            'parameters': results[best_rmse_idx]['parameters'],
            'average_results': results[best_rmse_idx]['aggregated_results']
        }
    
    def save_results(self, filename='lstm_corrected_results.json'):
        """Zapis wyników do JSON"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        print(f"\n✅ Wyniki zapisane do: {filename}")


# ============================================================================
# PRZYKŁAD UŻYCIA
# ============================================================================
def run_corrected_lstm(X, y, sequence_length=5, n_configs=20, n_repeats=5, 
                      use_multiprocessing=False, n_jobs=-1):
    """
    Uruchomienie poprawionego LSTM z sekwencjami czasowymi.
    
    Args:
        X: DataFrame lub array z cechami
        y: Series lub array z targetem
        sequence_length: długość sekwencji (domyślnie 5)
        n_configs: liczba konfiguracji
        n_repeats: liczba powtórzeń
        use_multiprocessing: czy używać multiprocessingu (domyślnie False)
        n_jobs: liczba procesów (-1 = wszystkie CPU, 1 = sekwencyjnie, >1 = konkretna liczba)
    
    Examples:
        # Bez multiprocessingu
        lstm = run_corrected_lstm(X, y, n_configs=10, n_repeats=5)
        
        # Wszystkie dostępne CPU
        lstm = run_corrected_lstm(X, y, n_configs=10, n_repeats=5, 
                                 use_multiprocessing=True, n_jobs=-1)
        
        # Konkretna liczba procesów (np. 4)
        lstm = run_corrected_lstm(X, y, n_configs=10, n_repeats=5, 
                                 use_multiprocessing=True, n_jobs=4)
    """
    lstm = LSTMTimeSeriesModel(sequence_length=sequence_length)
    lstm.run_random_search(
        X, y, 
        n_configs=n_configs, 
        n_repeats=n_repeats, 
        epochs=40,
        use_multiprocessing=use_multiprocessing,
        n_jobs=n_jobs
    )
    lstm.save_results('lstm_corrected_results1.json')
    return lstm