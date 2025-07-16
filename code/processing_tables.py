"""
Multi-Modal Medical Data Encoder
Encoders spécialisés pour données tabulaires, textuelles et time series
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.decomposition import PCA
import warnings
from pathlib import Path
import pickle
import json
import math

warnings.filterwarnings('ignore')

class FeatureSelectionNN(nn.Module):
    """
    Feed-forward Neural Network pour la sélection de features
    """
    def __init__(self, input_dim, output_dim, hidden_dims=[256, 128, 64], dropout=0.3):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Construire les couches
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Couche de sortie
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Couche d'attention pour feature importance
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, input_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Calcul de l'importance des features
        attention_weights = self.attention(x)
        
        # Application de l'attention
        x_attended = x * attention_weights
        
        # Passage dans le réseau principal
        output = self.network(x_attended)
        
        return output, attention_weights

class NeuralFeatureSelector:
    """
    Sélecteur de features basé sur un réseau de neurones
    """
    def __init__(self, target_features, hidden_dims=[256, 128, 64], epochs=100, 
                 learning_rate=0.001, batch_size=32, patience=10):
        self.target_features = target_features
        self.hidden_dims = hidden_dims
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.patience = patience
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_importance_ = None
        self.selected_features_ = None
        
    def fit(self, X, y=None):
        """
        Entraîne le réseau pour la sélection de features
        """
        print(f"    Entraînement NN pour sélection de features: {X.shape[1]} → {self.target_features}")
        
        # Préparation des données
        X_scaled = self.scaler.fit_transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        # Si on a des labels, on fait de la sélection supervisée
        if y is not None:
            # Déterminer le nombre de classes
            unique_labels = np.unique(y[y >= 0])  # Ignorer les -1 (inconnus)
            n_classes = len(unique_labels)
            
            # Créer le modèle
            self.model = FeatureSelectionNN(
                input_dim=X.shape[1],
                output_dim=n_classes,
                hidden_dims=self.hidden_dims
            ).to(self.device)
            
            # Préparer les labels
            y_valid = y[y >= 0]  # Garder seulement les labels valides
            X_valid = X_scaled[y >= 0]
            
            if len(y_valid) > 0:
                X_valid_tensor = torch.FloatTensor(X_valid).to(self.device)
                y_valid_tensor = torch.LongTensor(y_valid).to(self.device)
                
                # Entraînement supervisé
                self._train_supervised(X_valid_tensor, y_valid_tensor)
            else:
                print("    Pas de labels valides, passage en mode non-supervisé")
                self._train_unsupervised(X_tensor)
        else:
            # Sélection non-supervisée (autoencoder)
            print("    Mode non-supervisé (autoencoder)")
            self.model = FeatureSelectionNN(
                input_dim=X.shape[1],
                output_dim=X.shape[1],  # Reconstruction
                hidden_dims=self.hidden_dims
            ).to(self.device)
            
            self._train_unsupervised(X_tensor)
        
        # Calculer l'importance des features
        self._compute_feature_importance(X_tensor)
        
        # Sélectionner les meilleures features
        self.selected_features_ = np.argsort(self.feature_importance_)[-self.target_features:]
        
        self.is_fitted = True
        print(f"    Features sélectionnées par NN: {len(self.selected_features_)}")
        
        return self
    
    def _train_supervised(self, X, y):
        """Entraînement supervisé pour classification"""
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        
        # Dataset et DataLoader
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        best_loss = float('inf')
        patience_counter = 0
        
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0
            
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                
                outputs, attention_weights = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Regularisation pour encourager la sparsité
                sparsity_loss = 0.001 * torch.mean(attention_weights)
                total_loss = loss + sparsity_loss
                
                total_loss.backward()
                optimizer.step()
                
                epoch_loss += total_loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            scheduler.step(avg_loss)
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"    Early stopping à l'époque {epoch}")
                    break
            
            if (epoch + 1) % 20 == 0:
                print(f"    Époque {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")
    
    def _train_unsupervised(self, X):
        """Entraînement non-supervisé (autoencoder)"""
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Dataset et DataLoader
        dataset = torch.utils.data.TensorDataset(X)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0
            
            for (batch_X,) in dataloader:
                optimizer.zero_grad()
                
                reconstructed, attention_weights = self.model(batch_X)
                loss = criterion(reconstructed, batch_X)
                
                # Regularisation pour la sparsité
                sparsity_loss = 0.001 * torch.mean(attention_weights)
                total_loss = loss + sparsity_loss
                
                total_loss.backward()
                optimizer.step()
                
                epoch_loss += total_loss.item()
            
            if (epoch + 1) % 20 == 0:
                avg_loss = epoch_loss / len(dataloader)
                print(f"    Époque {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")
    
    def _compute_feature_importance(self, X):
        """Calcule l'importance des features via les poids d'attention"""
        self.model.eval()
        
        with torch.no_grad():
            _, attention_weights = self.model(X)
            # Moyenne des poids d'attention sur tous les échantillons
            self.feature_importance_ = attention_weights.mean(dim=0).cpu().numpy()
    
    def transform(self, X):
        """Applique la sélection de features"""
        if not self.is_fitted:
            raise ValueError("Le sélecteur doit être fitté avant transform")
        
        # Normaliser
        X_scaled = self.scaler.transform(X)
        
        # Sélectionner les features importantes
        return X_scaled[:, self.selected_features_]
    
    def fit_transform(self, X, y=None):
        """Fit et transform en une seule étape"""
        return self.fit(X, y).transform(X)


class TabularEncoder:
    """
    Encodeur classique pour données tabulaires
    Preprocessing + normalisation + feature selection
    """
    
    def __init__(self, n_features=50, output_dim=128):
        self.n_features = n_features
        self.output_dim = output_dim
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.feature_selector = SelectKBest(f_classif, k=n_features)
        self.label_encoders = {}
        self.feature_names = []
        self.valid_columns = []
        self.is_fitted = False
        
    def _preprocess_categorical(self, df):
        """Encode les variables catégorielles"""
        df_processed = df.copy()
        
        for col in df.columns:
            if df[col].dtype == 'object':
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    # Fit seulement si on n'a pas encore fitté
                    if not self.is_fitted:
                        df_processed[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                    else:
                        # Pour transform, gérer les nouvelles catégories
                        unique_vals = set(df[col].astype(str).unique())
                        known_vals = set(self.label_encoders[col].classes_)
                        new_vals = unique_vals - known_vals
                        
                        if new_vals:
                            # Remplacer les nouvelles valeurs par la plus fréquente
                            most_frequent = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else self.label_encoders[col].classes_[0]
                            df_processed[col] = df[col].astype(str).replace(list(new_vals), most_frequent)
                        
                        df_processed[col] = self.label_encoders[col].transform(df_processed[col].astype(str))
                else:
                    # Transform uniquement
                    unique_vals = set(df[col].astype(str).unique())
                    known_vals = set(self.label_encoders[col].classes_)
                    new_vals = unique_vals - known_vals
                    
                    if new_vals:
                        most_frequent = self.label_encoders[col].classes_[0]
                        df_processed[col] = df[col].astype(str).replace(list(new_vals), most_frequent)
                    
                    df_processed[col] = self.label_encoders[col].transform(df_processed[col].astype(str))
        
        return df_processed
    
    def fit(self, X, y=None):
        """Fit l'encodeur sur les données d'entraînement"""
        print(f"TabularEncoder: Fitting sur {X.shape[0]} échantillons, {X.shape[1]} features")
        
        # Preprocessing catégoriel
        X_processed = self._preprocess_categorical(X)
        print(f"Après preprocessing catégoriel: {X_processed.shape}")
        
        # Nettoyer les colonnes problématiques (colonnes avec que des NaN)
        X_clean = X_processed.dropna(axis=1, how='all')
        print(f"Après suppression colonnes vides: {X_clean.shape}")
        
        # Sauvegarder les colonnes valides
        self.valid_columns = X_clean.columns.tolist()
        
        # Imputation des valeurs manquantes
        X_imputed_values = self.imputer.fit_transform(X_clean)
        X_imputed = pd.DataFrame(
            X_imputed_values,
            columns=X_clean.columns,
            index=X_clean.index
        )
        
        # Normalisation
        X_scaled_values = self.scaler.fit_transform(X_imputed)
        X_scaled = pd.DataFrame(
            X_scaled_values,
            columns=X_imputed.columns,
            index=X_imputed.index
        )
        
        # Feature selection (si on a un target y)
        if y is not None:
            # Ajuster k si on a moins de features que demandé
            actual_k = min(self.n_features, X_scaled.shape[1])
            self.feature_selector.k = actual_k
            
            X_selected = self.feature_selector.fit_transform(X_scaled, y)
            self.feature_names = X_scaled.columns[self.feature_selector.get_support()].tolist()
        else:
            # Pas de feature selection, garder toutes les features
            X_selected = X_scaled.values
            self.feature_names = X_scaled.columns.tolist()
        
        self.is_fitted = True
        print(f"TabularEncoder: Features sélectionnées: {len(self.feature_names)}")
        return self
    
    def transform(self, X):
        """Transform les données avec l'encodeur fitté"""
        if not self.is_fitted:
            raise ValueError("L'encodeur doit être fitté avant transform")
        
        # Preprocessing catégoriel
        X_processed = self._preprocess_categorical(X)
        
        # Utiliser seulement les colonnes valides identifiées lors du fit
        X_clean = X_processed[self.valid_columns]
        
        # Imputation
        X_imputed = pd.DataFrame(
            self.imputer.transform(X_clean),
            columns=X_clean.columns,
            index=X_clean.index
        )
        
        # Normalisation
        X_scaled = pd.DataFrame(
            self.scaler.transform(X_imputed),
            columns=X_imputed.columns,
            index=X_imputed.index
        )
        
        # Feature selection
        if hasattr(self.feature_selector, 'get_support'):
            X_selected = X_scaled[self.feature_names].values
        else:
            X_selected = X_scaled.values
        
        # Padding ou truncation pour avoir la dimension souhaitée
        if X_selected.shape[1] < self.output_dim:
            # Padding avec des zéros
            padding = np.zeros((X_selected.shape[0], self.output_dim - X_selected.shape[1]))
            X_output = np.concatenate([X_selected, padding], axis=1)
        else:
            # Truncation
            X_output = X_selected[:, :self.output_dim]
        
        return X_output
    
    def fit_transform(self, X, y=None):
        """Fit et transform en une seule étape"""
        return self.fit(X, y).transform(X)


class TextEncoder:
    """
    Encodeur basé sur Clinical BioBERT pour données textuelles
    """
    
    def __init__(self, model_name="emilyalsentzer/Bio_ClinicalBERT", output_dim=128, max_length=512):
        self.model_name = model_name
        self.output_dim = output_dim
        self.max_length = max_length
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def _load_model(self):
        """Charge le modèle Clinical BioBERT"""
        if self.tokenizer is None:
            print(f"TextEncoder: Chargement du modèle {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            print(f"TextEncoder: Modèle chargé sur {self.device}")
    
    def _clean_text(self, text):
        """Nettoie et prépare le texte"""
        if pd.isna(text) or text == '':
            return "no information available"
        
        # Convertir en string et nettoyer
        text = str(text).strip()
        
        # Remplacer les caractères problématiques
        text = text.replace('\n', ' ').replace('\r', ' ')
        text = ' '.join(text.split())  # Normaliser les espaces
        
        return text if len(text) > 0 else "no information available"
    
    def _encode_single_text(self, text):
        """Encode un seul texte avec BioBERT"""
        cleaned_text = self._clean_text(text)
        
        # Tokenisation
        inputs = self.tokenizer(
            cleaned_text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=True
        ).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Utiliser le CLS token comme représentation
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return embeddings.flatten()
    
    def fit(self, X, y=None):
        """Charge le modèle (pas d'entraînement nécessaire)"""
        self._load_model()
        return self
    
    def transform(self, X):
        """Transform les données textuelles en embeddings"""
        if self.tokenizer is None:
            self._load_model()
        
        print(f"TextEncoder: Encodage de {len(X)} textes")
        
        embeddings_list = []
        
        for idx, row in X.iterrows():
            # Concaténer tous les textes de la ligne
            text_parts = []
            for col in X.columns:
                if pd.notna(row[col]) and str(row[col]).strip():
                    text_parts.append(str(row[col]))
            
            combined_text = " ".join(text_parts) if text_parts else "no information"
            
            # Encoder le texte
            embedding = self._encode_single_text(combined_text)
            
            # Adapter à la dimension souhaitée
            if len(embedding) > self.output_dim:
                # Truncation
                embedding = embedding[:self.output_dim]
            else:
                # Padding
                padding = np.zeros(self.output_dim - len(embedding))
                embedding = np.concatenate([embedding, padding])
            
            embeddings_list.append(embedding)
        
        return np.array(embeddings_list)
    
    def fit_transform(self, X, y=None):
        """Fit et transform en une seule étape"""
        return self.fit(X, y).transform(X)


class TS2VecEncoder:
    """
    Encodeur basé sur TS2Vec pour données de séries temporelles CGM
    TS2Vec: Towards Universal Representation of Time Series
    Optimisé pour les données de glycémie continue (CGM)
    """
    
    def __init__(self, output_dim=128, max_seq_length=2880, d_model=320, n_heads=8, 
                 n_layers=3, dropout=0.1, mask_ratio=0.15):
        self.output_dim = output_dim
        self.max_seq_length = max_seq_length  # 2 jours à 1 min = 2880 points
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.mask_ratio = mask_ratio
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Modèle TS2Vec
        self.model = self._build_ts2vec_model()
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # Paramètres spécifiques CGM
        self.cgm_stats = {
            'glucose_mean': None,
            'glucose_std': None,
            'normal_range': (70, 180),  # mg/dL
            'hypo_threshold': 70,
            'hyper_threshold': 180
        }
    
    def _build_ts2vec_model(self):
        """Construit le modèle TS2Vec adapté pour CGM"""
        
        class SamePadConv(nn.Module):
            def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1):
                super().__init__()
                self.receptive_field = (kernel_size - 1) * dilation + 1
                padding = self.receptive_field // 2
                self.conv = nn.Conv1d(
                    in_channels, out_channels, kernel_size,
                    padding=padding, dilation=dilation, groups=groups
                )
                self.remove = 1 if self.receptive_field % 2 == 0 else 0
                
            def forward(self, x):
                out = self.conv(x)
                if self.remove > 0:
                    out = out[:, :, :-self.remove]
                return out
        
        class ConvBlock(nn.Module):
            def __init__(self, in_channels, out_channels, kernel_size, dilation, final=False):
                super().__init__()
                self.conv1 = SamePadConv(in_channels, out_channels, kernel_size, dilation=dilation)
                self.conv2 = SamePadConv(out_channels, out_channels, kernel_size, dilation=dilation)
                self.projector = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
                self.final = final
                
            def forward(self, x):
                residual = x if self.projector is None else self.projector(x)
                x = F.gelu(x)
                x = self.conv1(x)
                x = F.gelu(x)
                x = self.conv2(x)
                if not self.final:
                    x = x + residual
                return x
        
        class DilatedConvEncoder(nn.Module):
            def __init__(self, in_channels, channels, kernel_size):
                super().__init__()
                self.net = nn.Sequential(*[
                    ConvBlock(
                        channels[i-1] if i > 0 else in_channels,
                        channels[i],
                        kernel_size=kernel_size,
                        dilation=2**i,
                        final=(i == len(channels)-1)
                    )
                    for i in range(len(channels))
                ])
                
            def forward(self, x):
                return self.net(x)
        
        class TS2VecModel(nn.Module):
            def __init__(self, input_dims, output_dims, hidden_dims=64, depth=10, mask_ratio=0.15):
                super().__init__()
                self.input_dims = input_dims
                self.output_dims = output_dims
                self.hidden_dims = hidden_dims
                self.mask_ratio = mask_ratio
                
                # Input projection pour CGM (glucose + metadata)
                self.input_fc = nn.Linear(input_dims, hidden_dims)
                
                # Dilated convolutional encoder (cœur de TS2Vec)
                self.feature_extractor = DilatedConvEncoder(
                    in_channels=hidden_dims,
                    channels=[hidden_dims] * depth,
                    kernel_size=3
                )
                
                # Représentation encoder
                self.repr_dropout = nn.Dropout(0.1)
                
                # Projection head pour contrastive learning
                self.projection_head = nn.Sequential(
                    nn.Linear(hidden_dims, hidden_dims),
                    nn.ReLU(),
                    nn.Linear(hidden_dims, output_dims)
                )
                
            def forward(self, x, mask=None):
                # x: (batch, seq_len, input_dims)
                batch_size, seq_len = x.shape[:2]
                
                # Input projection
                x = self.input_fc(x)  # (batch, seq_len, hidden_dims)
                
                # Transpose pour conv1d: (batch, hidden_dims, seq_len)
                x = x.transpose(1, 2)
                
                # Feature extraction avec dilated convolutions
                x = self.feature_extractor(x)  # (batch, hidden_dims, seq_len)
                
                # Global average pooling
                x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)  # (batch, hidden_dims)
                
                # Dropout
                x = self.repr_dropout(x)
                
                # Projection finale
                out = self.projection_head(x)  # (batch, output_dims)
                
                return out
            
            def encode(self, x, mask=None, encoding_window=None):
                """Encode une séquence en représentation"""
                return self.forward(x, mask)
        
        return TS2VecModel(
            input_dims=1,  # Glucose values (peut être étendu)
            output_dims=self.output_dim,
            hidden_dims=self.d_model,
            depth=self.n_layers * 2,  # Plus de couches pour TS2Vec
            mask_ratio=self.mask_ratio
        ).to(self.device)
    
    def _extract_cgm_features(self, glucose_values):
        """Extrait les métriques cliniques CGM standards"""
        glucose_values = glucose_values[~np.isnan(glucose_values)]
        
        if len(glucose_values) == 0:
            return {
                'mean_glucose': 100.0,
                'glucose_std': 0.0,
                'cv': 0.0,
                'tir_70_180': 0.0,
                'tbr_70': 0.0,
                'tar_180': 0.0,
                'hypo_events': 0,
                'hyper_events': 0
            }
        
        features = {}
        
        # Métriques de base
        features['mean_glucose'] = np.mean(glucose_values)
        features['glucose_std'] = np.std(glucose_values)
        features['cv'] = (features['glucose_std'] / features['mean_glucose']) * 100 if features['mean_glucose'] > 0 else 0
        
        # Time in Range (TIR) - métriques cliniques standards
        features['tir_70_180'] = np.mean((glucose_values >= 70) & (glucose_values <= 180)) * 100
        features['tbr_70'] = np.mean(glucose_values < 70) * 100
        features['tar_180'] = np.mean(glucose_values > 180) * 100
        
        # Événements hypo/hyperglycémiques
        # Événement = séquence consécutive de valeurs en dehors de la plage
        hypo_events = 0
        hyper_events = 0
        
        in_hypo = False
        in_hyper = False
        
        for glucose in glucose_values:
            if glucose < 70:
                if not in_hypo:
                    hypo_events += 1
                    in_hypo = True
                in_hyper = False
            elif glucose > 180:
                if not in_hyper:
                    hyper_events += 1
                    in_hyper = True
                in_hypo = False
            else:
                in_hypo = False
                in_hyper = False
        
        features['hypo_events'] = hypo_events
        features['hyper_events'] = hyper_events
        
        return features
    
    def _process_cgm_data(self, X):
        """Traite les données CGM pour TS2Vec"""
        processed_sequences = []
        cgm_metadata = []
        
        for idx, row in X.iterrows():
            # Extraire toutes les valeurs de glucose pour ce patient
            glucose_values = []
            
            for col in X.columns:
                if pd.notna(row[col]):
                    try:
                        if isinstance(row[col], str):
                            # Parse string values (format CSV ou séparés par virgules)
                            values = [float(x.strip()) for x in row[col].split(',') if x.strip()]
                            glucose_values.extend(values)
                        elif isinstance(row[col], (list, np.ndarray)):
                            glucose_values.extend([float(x) for x in row[col] if not np.isnan(float(x))])
                        else:
                            val = float(row[col])
                            if not np.isnan(val):
                                glucose_values.append(val)
                    except (ValueError, AttributeError, TypeError):
                        continue
            
            # Nettoyer et valider les valeurs de glucose (plage physiologique)
            glucose_values = [g for g in glucose_values if 20 <= g <= 600]
            
            # Si pas assez de données, utiliser une séquence de base
            if len(glucose_values) < 10:
                glucose_values = [100.0] * 50  # Valeur normale par défaut
            
            # Limiter à max_seq_length
            if len(glucose_values) > self.max_seq_length:
                # Prendre les dernières valeurs (plus récentes)
                glucose_values = glucose_values[-self.max_seq_length:]
            
            processed_sequences.append(glucose_values)
            
            # Extraire métadonnées CGM
            metadata = self._extract_cgm_features(np.array(glucose_values))
            cgm_metadata.append(metadata)
        
        return processed_sequences, cgm_metadata
    
    def fit(self, X, y=None):
        """Fit l'encodeur TS2Vec (preprocessing + statistics)"""
        print(f"TS2VecEncoder: Fitting sur {len(X)} échantillons CGM")
        
        # Traiter les données CGM
        sequences, metadata = self._process_cgm_data(X)
        
        # Calculer les statistiques globales pour la normalisation
        all_values = []
        for seq in sequences:
            all_values.extend(seq)
        
        if all_values:
            all_values = np.array(all_values).reshape(-1, 1)
            self.scaler.fit(all_values)
            
            # Sauvegarder les stats CGM globales
            self.cgm_stats['glucose_mean'] = np.mean(all_values)
            self.cgm_stats['glucose_std'] = np.std(all_values)
            
        print(f"  📊 Stats CGM globales:")
        print(f"      Moyenne glucose: {self.cgm_stats['glucose_mean']:.1f} mg/dL")
        print(f"      Écart-type: {self.cgm_stats['glucose_std']:.1f} mg/dL")
        
        # Calculer des métriques moyennes
        if metadata:
            avg_tir = np.mean([m['tir_70_180'] for m in metadata])
            avg_cv = np.mean([m['cv'] for m in metadata])
            print(f"      TIR moyen: {avg_tir:.1f}%")
            print(f"      CV moyen: {avg_cv:.1f}%")
        
        self.is_fitted = True
        print(f"TS2VecEncoder: Fit terminé")
        return self
    
    def transform(self, X):
        """Transform les données CGM avec TS2Vec"""
        if not self.is_fitted:
            raise ValueError("L'encodeur doit être fitté avant transform")
        
        print(f"TS2VecEncoder: Encodage de {len(X)} séries CGM avec TS2Vec")
        
        # Traiter les données
        sequences, metadata = self._process_cgm_data(X)
        
        # Préparer les batch pour TS2Vec
        batch_sequences = []
        
        for seq in sequences:
            # Normaliser la séquence
            seq_array = np.array(seq).reshape(-1, 1)
            seq_normalized = self.scaler.transform(seq_array).flatten()
            
            # Padding si nécessaire
            if len(seq_normalized) < self.max_seq_length:
                padded_seq = np.zeros(self.max_seq_length)
                padded_seq[:len(seq_normalized)] = seq_normalized
            else:
                padded_seq = seq_normalized[:self.max_seq_length]
            
            batch_sequences.append(padded_seq)
        
        # Convertir en tensor pour TS2Vec
        # Shape: (batch_size, seq_length, 1)
        batch_tensor = torch.FloatTensor(batch_sequences).unsqueeze(-1).to(self.device)
        
        print(f"  🔄 Batch tensor shape: {batch_tensor.shape}")
        
        # Encoder avec TS2Vec
        self.model.eval()
        with torch.no_grad():
            # Utiliser la méthode encode de TS2Vec
            ts2vec_embeddings = self.model.encode(batch_tensor)
        
        # Convertir en numpy
        embeddings_np = ts2vec_embeddings.cpu().numpy()
        
        print(f"  ✅ TS2Vec embeddings shape: {embeddings_np.shape}")
        print(f"  📈 Encodage TS2Vec terminé: {len(sequences)} séries → {embeddings_np.shape[1]} features")
        
        return embeddings_np
    
    def fit_transform(self, X, y=None):
        """Fit et transform en une seule étape"""
        return self.fit(X, y).transform(X)


class MultiModalCrossAttention(nn.Module):
    """
    Cross-Attention mechanism pour fusion multimodale
    Chaque modalité attend aux autres modalités pour créer des représentations enrichies
    """
    def __init__(self, d_model=256, n_heads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        
        # Projections pour chaque modalité vers la même dimension
        self.tabular_projection = nn.Linear(50, d_model)  # Ajustable selon tes features
        self.text_projection = nn.Linear(100, d_model)
        self.timeseries_projection = nn.Linear(32, d_model)
        
        # Cross-attention layers
        self.tabular_cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.text_cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.timeseries_cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        
        # Feed-forward networks pour chaque modalité
        self.tabular_ffn = self._make_ffn(d_model)
        self.text_ffn = self._make_ffn(d_model)
        self.timeseries_ffn = self._make_ffn(d_model)
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(6)  # 2 par modalité (avant et après attention)
        ])
        
        # Final fusion layer avec activations qui préservent l'information
        self.fusion_layer = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),
            nn.GELU(),  # GELU au lieu de ReLU pour préserver plus d'information
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Tanh()   # Tanh pour garder les valeurs entre -1 et 1
        )
        
    def _make_ffn(self, d_model, d_ff=None):
        """Crée un feed-forward network avec activations améliorées"""
        if d_ff is None:
            d_ff = d_model * 4
        return nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),  # GELU au lieu de ReLU
            nn.Dropout(0.1),
            nn.Linear(d_ff, d_model)
        )
    
    def forward(self, tabular_features=None, text_features=None, timeseries_features=None, mask=None):
        """
        Forward pass avec cross-attention entre modalités
        
        Args:
            tabular_features: (batch_size, tabular_dim)
            text_features: (batch_size, text_dim)  
            timeseries_features: (batch_size, ts_dim)
            mask: masque pour les modalités manquantes
        """
        batch_size = None
        modalities = []
        projected_modalities = []
        
        # Projeter chaque modalité vers d_model et créer des tokens
        if tabular_features is not None:
            batch_size = tabular_features.size(0)
            tab_proj = self.tabular_projection(tabular_features)  # (batch, d_model)
            tab_tokens = tab_proj.unsqueeze(1)  # (batch, 1, d_model) - 1 token par modalité
            modalities.append(('tabular', tab_tokens, 0))
            projected_modalities.append(tab_proj)
        
        if text_features is not None:
            if batch_size is None:
                batch_size = text_features.size(0)
            text_proj = self.text_projection(text_features)
            text_tokens = text_proj.unsqueeze(1)  # (batch, 1, d_model)
            modalities.append(('text', text_tokens, 1))
            projected_modalities.append(text_proj)
            
        if timeseries_features is not None:
            if batch_size is None:
                batch_size = timeseries_features.size(0)
            ts_proj = self.timeseries_projection(timeseries_features)
            ts_tokens = ts_proj.unsqueeze(1)  # (batch, 1, d_model)
            modalities.append(('timeseries', ts_tokens, 2))
            projected_modalities.append(ts_proj)
        
        if len(modalities) < 2:
            # Si moins de 2 modalités, pas besoin de cross-attention
            if len(modalities) == 1:
                return modalities[0][1].squeeze(1)  # Retourner la seule modalité
            else:
                # Aucune modalité, retourner des zéros
                return torch.zeros(1, self.d_model)
        
        # Créer les contextes pour le cross-attention
        # Chaque modalité va attendre à toutes les autres modalités
        attended_modalities = []
        
        for i, (mod_name, mod_tokens, mod_idx) in enumerate(modalities):
            # Créer le contexte (toutes les autres modalités)
            other_modalities = [other_tokens for j, (_, other_tokens, _) in enumerate(modalities) if j != i]
            
            if len(other_modalities) > 0:
                # Concaténer les autres modalités comme contexte
                context = torch.cat(other_modalities, dim=1)  # (batch, num_other_modalities, d_model)
                
                # Cross-attention: la modalité courante attend au contexte
                if mod_name == 'tabular':
                    attended_output, _ = self.tabular_cross_attn(
                        query=mod_tokens,  # (batch, 1, d_model)
                        key=context,       # (batch, num_others, d_model)
                        value=context      # (batch, num_others, d_model)
                    )
                    # Residual connection + layer norm
                    attended_output = self.layer_norms[mod_idx * 2](mod_tokens + attended_output)
                    # Feed-forward
                    ffn_output = self.tabular_ffn(attended_output)
                    final_output = self.layer_norms[mod_idx * 2 + 1](attended_output + ffn_output)
                    
                elif mod_name == 'text':
                    attended_output, _ = self.text_cross_attn(
                        query=mod_tokens,
                        key=context,
                        value=context
                    )
                    attended_output = self.layer_norms[mod_idx * 2](mod_tokens + attended_output)
                    ffn_output = self.text_ffn(attended_output)
                    final_output = self.layer_norms[mod_idx * 2 + 1](attended_output + ffn_output)
                    
                elif mod_name == 'timeseries':
                    attended_output, _ = self.timeseries_cross_attn(
                        query=mod_tokens,
                        key=context,
                        value=context
                    )
                    attended_output = self.layer_norms[mod_idx * 2](mod_tokens + attended_output)
                    ffn_output = self.timeseries_ffn(attended_output)
                    final_output = self.layer_norms[mod_idx * 2 + 1](attended_output + ffn_output)
                
                attended_modalities.append(final_output.squeeze(1))  # (batch, d_model)
            else:
                # Si pas d'autres modalités, garder la modalité originale
                attended_modalities.append(projected_modalities[i])
        
        # Fusion finale
        if len(attended_modalities) > 1:
            # Concaténer toutes les modalités attendues
            concatenated = torch.cat(attended_modalities, dim=1)  # (batch, num_modalities * d_model)
            fused_representation = self.fusion_layer(concatenated)  # (batch, d_model)
        else:
            fused_representation = attended_modalities[0]
        
        return fused_representation
class CLSTransformerFusion(nn.Module):
    """
    Fusion basée sur un Transformer avec token CLS pour la classification finale
    Inspiré de BERT - le token CLS apprend à représenter l'ensemble des modalités
    """
    def __init__(self, d_model=256, n_heads=8, n_layers=3, dropout=0.1, n_classes=2):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.n_classes = n_classes
        
        # Token CLS learnable (comme dans BERT)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Projections pour chaque modalité vers d_model
        self.tabular_projection = None
        self.text_projection = None  
        self.ts_projection = None
        
        # Embeddings positionnels pour différencier les modalités
        self.position_embeddings = nn.Embedding(4, d_model)  # CLS + 3 modalités max
        
        # Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',  # GELU comme dans BERT
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Classification head basé sur le token CLS
        self.classification_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes)
        )
        
        # Tête pour extraction de features
        self.feature_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
        
        self.fitted = False
        
    def fit(self, tabular_dim=None, text_dim=None, ts_dim=None):
        """
        Initialise les projections selon les dimensions des modalités disponibles
        """
        print(f"  🔧 Initialisation CLSTransformerFusion:")
        print(f"     - d_model: {self.d_model}, heads: {self.n_heads}, layers: {self.n_layers}")
        
        # Créer les projections pour chaque modalité disponible
        if tabular_dim is not None:
            self.tabular_projection = nn.Linear(tabular_dim, self.d_model)
            print(f"     - Tabular projection: {tabular_dim} → {self.d_model}")
            
        if text_dim is not None:
            self.text_projection = nn.Linear(text_dim, self.d_model)
            print(f"     - Text projection: {text_dim} → {self.d_model}")
            
        if ts_dim is not None:
            self.ts_projection = nn.Linear(ts_dim, self.d_model)
            print(f"     - TimeSeries projection: {ts_dim} → {self.d_model}")
        
        self.fitted = True
        
    def forward(self, tabular_features=None, text_features=None, ts_features=None, return_cls_only=False):
        """
        Forward pass avec token CLS
        
        Args:
            tabular_features: Tensor de features tabulaires [batch, tabular_dim]
            text_features: Tensor de features textuelles [batch, text_dim]  
            ts_features: Tensor de features time series [batch, ts_dim]
            return_cls_only: Si True, ne retourne que le token CLS pour classification
        """
        batch_size = None
        sequence = []
        
        # Déterminer la taille du batch
        for features in [tabular_features, text_features, ts_features]:
            if features is not None:
                batch_size = features.shape[0]
                break
                
        if batch_size is None:
            raise ValueError("Aucune modalité fournie")
        
        # Ajouter le token CLS au début de la séquence
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [batch, 1, d_model]
        sequence.append(cls_tokens)
        position_ids = [0]  # CLS token à la position 0
        
        # Projeter et ajouter chaque modalité disponible
        current_pos = 1
        
        if tabular_features is not None and self.tabular_projection is not None:
            projected = self.tabular_projection(tabular_features).unsqueeze(1)  # [batch, 1, d_model]
            sequence.append(projected)
            position_ids.append(current_pos)
            current_pos += 1
            
        if text_features is not None and self.text_projection is not None:
            projected = self.text_projection(text_features).unsqueeze(1)  # [batch, 1, d_model]
            sequence.append(projected)
            position_ids.append(current_pos)
            current_pos += 1
            
        if ts_features is not None and self.ts_projection is not None:
            projected = self.ts_projection(ts_features).unsqueeze(1)  # [batch, 1, d_model]
            sequence.append(projected)
            position_ids.append(current_pos)
            current_pos += 1
        
        # Concaténer toute la séquence: [CLS, modalité1, modalité2, ...]
        sequence_tensor = torch.cat(sequence, dim=1)  # [batch, seq_len, d_model]
        
        # Ajouter les embeddings positionnels
        position_ids_tensor = torch.tensor(position_ids, device=sequence_tensor.device).unsqueeze(0)
        position_embeddings = self.position_embeddings(position_ids_tensor)  # [1, seq_len, d_model]
        sequence_tensor = sequence_tensor + position_embeddings
        
        # Passer dans le transformer
        transformer_output = self.transformer(sequence_tensor)  # [batch, seq_len, d_model]
        
        # Extraire le token CLS (première position)
        cls_output = transformer_output[:, 0, :]  # [batch, d_model]
        
        if return_cls_only:
            return cls_output
        
        # Passer le CLS token dans la tête de features pour obtenir la représentation finale
        features_output = self.feature_head(cls_output)  # [batch, d_model]
        
        return features_output
    
    def get_classification_logits(self, tabular_features=None, text_features=None, ts_features=None):
        """
        Obtient les logits de classification basés sur le token CLS
        """
        cls_token = self.forward(tabular_features, text_features, ts_features, return_cls_only=True)
        return self.classification_head(cls_token)
    
    def transform(self, reduced_features):
        """
        Transforme les features réduites en utilisant le token CLS
        Compatible avec l'interface existante
        """
        if not self.fitted:
            raise RuntimeError("Le modèle doit être fitted avant transformation")
        
        self.eval()
        results = []
        pt_ids = []
        
        # Obtenir tous les PtIDs uniques
        all_pt_ids = set()
        for modality_df in reduced_features.values():
            all_pt_ids.update(modality_df['PtID'].values)
        all_pt_ids = sorted(list(all_pt_ids))
        
        with torch.no_grad():
            for pt_id in all_pt_ids:
                # Extraire les features pour ce patient
                tabular_feat = None
                text_feat = None
                ts_feat = None
                
                if 'tabular' in reduced_features:
                    tabular_df = reduced_features['tabular']
                    tabular_row = tabular_df[tabular_df['PtID'] == pt_id]
                    if len(tabular_row) > 0:
                        feat_cols = [col for col in tabular_df.columns if col != 'PtID']
                        tabular_feat = torch.FloatTensor(tabular_row[feat_cols].values)
                
                if 'text' in reduced_features:
                    text_df = reduced_features['text']
                    text_row = text_df[text_df['PtID'] == pt_id]
                    if len(text_row) > 0:
                        feat_cols = [col for col in text_df.columns if col != 'PtID']
                        text_feat = torch.FloatTensor(text_row[feat_cols].values)
                
                if 'timeseries' in reduced_features:
                    ts_df = reduced_features['timeseries']
                    ts_row = ts_df[ts_df['PtID'] == pt_id]
                    if len(ts_row) > 0:
                        feat_cols = [col for col in ts_df.columns if col != 'PtID']
                        ts_feat = torch.FloatTensor(ts_row[feat_cols].values)
                
                # Au moins une modalité doit être disponible
                if tabular_feat is not None or text_feat is not None or ts_feat is not None:
                    # Forward pass avec le transformer + CLS token
                    fused_features = self.forward(tabular_feat, text_feat, ts_feat)
                    # Aplatir les dimensions si nécessaire (enlever la dimension batch)
                    features_flat = fused_features.squeeze().numpy()  # Enlever les dimensions de taille 1
                    results.append(features_flat)
                    pt_ids.append(pt_id)
        
        # Créer le DataFrame de résultats
        feature_names = [f'cls_transformer_feat_{i}' for i in range(self.d_model)]
        fused_df = pd.DataFrame(results, columns=feature_names)
        fused_df.insert(0, 'PtID', pt_ids)
        
        print(f"  🎯 CLS Transformer Fusion: {len(pt_ids)} patients → {self.d_model} features")
        
        return fused_df

class CrossAttentionFusion:
    """
    Wrapper pour utiliser le cross-attention dans le pipeline
    """
    def __init__(self, d_model=256, n_heads=8, device=None):
        self.d_model = d_model
        self.n_heads = n_heads
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = None
        self.is_fitted = False
        
    def fit(self, tabular_dim=50, text_dim=100, ts_dim=32):
        """Initialise le modèle de cross-attention avec les bonnes dimensions"""
        # Mettre à jour les dimensions dans le modèle
        self.model = MultiModalCrossAttention(
            d_model=self.d_model, 
            n_heads=self.n_heads
        ).to(self.device)
        
        # Ajuster les projections selon les vraies dimensions
        if tabular_dim != 50:
            self.model.tabular_projection = nn.Linear(tabular_dim, self.d_model).to(self.device)
        if text_dim != 100:
            self.model.text_projection = nn.Linear(text_dim, self.d_model).to(self.device)
        if ts_dim != 32:
            self.model.timeseries_projection = nn.Linear(ts_dim, self.d_model).to(self.device)
        
        self.model.eval()  # Mode évaluation (pas d'entraînement pour l'instant)
        self.is_fitted = True
        print(f"CrossAttentionFusion initialisé: tabular({tabular_dim}) + text({text_dim}) + ts({ts_dim}) → {self.d_model}")
        
    def transform(self, reduced_features):
        """
        Applique le cross-attention fusion sur les features réduites
        
        Args:
            reduced_features: Dict avec les DataFrames de chaque modalité
        """
        if not self.is_fitted:
            raise ValueError("Le modèle doit être fitté avant transform")
        
        # Préparer les données
        batch_data = {}
        all_patient_ids = set()
        
        # Collecter tous les PtID
        for modality, df in reduced_features.items():
            all_patient_ids.update(df['PtID'].values)
        
        all_patient_ids = sorted(list(all_patient_ids))
        n_patients = len(all_patient_ids)
        
        # Préparer les tenseurs pour chaque modalité
        for modality, df in reduced_features.items():
            # Créer un mapping PtID -> index
            patient_to_idx = {pid: i for i, pid in enumerate(all_patient_ids)}
            
            # Extraire les features (sans PtID)
            feature_cols = [col for col in df.columns if col != 'PtID']
            n_features = len(feature_cols)
            
            # Créer le tenseur avec les bonnes dimensions
            modality_tensor = torch.zeros(n_patients, n_features, device=self.device)
            
            for _, row in df.iterrows():
                pt_id = row['PtID']
                if pt_id in patient_to_idx:
                    idx = patient_to_idx[pt_id]
                    feature_values = torch.tensor(row[feature_cols].values, dtype=torch.float32)
                    modality_tensor[idx] = feature_values
            
            batch_data[modality] = modality_tensor
        
        # Appliquer le cross-attention
        with torch.no_grad():
            fused_features = self.model(
                tabular_features=batch_data.get('tabular'),
                text_features=batch_data.get('text'),
                timeseries_features=batch_data.get('timeseries')
            )
        
        # Convertir en DataFrame
        fused_array = fused_features.cpu().numpy()
        
        # POST-PROCESSING pour réduire les zéros
        print(f"  Post-processing: {(fused_array == 0.0).sum()} zéros sur {fused_array.size} valeurs")
        
        # 1. Appliquer une transformation pour réduire les zéros stricts
        # Remplacer les zéros par de très petites valeurs aléatoires
        zero_mask = (fused_array == 0.0)
        if zero_mask.sum() > 0:
            noise = np.random.normal(0, 1e-6, size=zero_mask.sum())
            fused_array[zero_mask] = noise
            print(f"  Remplacement de {zero_mask.sum()} zéros stricts par du bruit")
        
        # 2. Normalisation Min-Max pour avoir des valeurs dans [0, 1]
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0.01, 0.99))  # Éviter les 0 et 1 exacts
        fused_array_normalized = scaler.fit_transform(fused_array)
        
        # 3. Créer les noms de features
        feature_names = [f'cross_attn_feat_{i}' for i in range(fused_array_normalized.shape[1])]
        
        fused_df = pd.DataFrame(fused_array_normalized, columns=feature_names)
        fused_df.insert(0, 'PtID', all_patient_ids)
        
        # Statistiques finales
        final_zeros = (fused_array_normalized == 0.0).sum()
        final_percentage = (final_zeros / fused_array_normalized.size) * 100
        print(f"  Après post-processing: {final_zeros} zéros ({final_percentage:.1f}%)")
        
        return fused_df

def load_data_by_type(preprocessed_dir):
    """
    Charge les données par type depuis le dossier preprocessed
    """
    preprocessed_path = Path(preprocessed_dir)
    
    # Charger les données tabulaires
    tabular_dir = preprocessed_path / "tabular"
    tabular_data = None
    if tabular_dir.exists():
        print("Chargement des données tabulaires...")
        tabular_files = list(tabular_dir.glob("*.csv"))
        for file_path in tabular_files:
            print(f"  Traitement de {file_path.name}...")
            df = pd.read_csv(file_path)
            
            if 'PtID' not in df.columns:
                print(f"    Attention: Pas de colonne PtID dans {file_path.name}, ignoré")
                continue
            
            # Nettoyer les colonnes problématiques (colonnes avec que des NaN)
            cols_to_remove = ['label_encoded', 'Unnamed: 0']
            for col in cols_to_remove:
                if col in df.columns:
                    df = df.drop(col, axis=1)
                    print(f"    Suppression de la colonne: {col}")
            
            # Ajouter un suffixe unique pour éviter les conflits
            file_suffix = file_path.stem.replace('bbdd_', '')
            df_renamed = df.copy()
            
            # Renommer toutes les colonnes sauf PtID
            for col in df.columns:
                if col != 'PtID':
                    new_name = f"{col}_{file_suffix}"
                    df_renamed = df_renamed.rename(columns={col: new_name})
            
            if tabular_data is None:
                tabular_data = df_renamed
            else:
                tabular_data = pd.merge(tabular_data, df_renamed, on='PtID', how='outer')
                
        print(f"Données tabulaires: {tabular_data.shape if tabular_data is not None else 'Aucune'}")
    
    # Charger les données textuelles
    text_dir = preprocessed_path / "text"
    text_data = None
    if text_dir.exists():
        print("Chargement des données textuelles...")
        text_files = list(text_dir.glob("*.csv"))
        for file_path in text_files:
            print(f"  Traitement de {file_path.name}...")
            df = pd.read_csv(file_path)
            
            if 'PtID' not in df.columns:
                print(f"    Attention: Pas de colonne PtID dans {file_path.name}, ignoré")
                continue
                
            # Nettoyer les colonnes problématiques
            cols_to_remove = ['label_encoded', 'Unnamed: 0']
            for col in cols_to_remove:
                if col in df.columns:
                    df = df.drop(col, axis=1)
                    print(f"    Suppression de la colonne: {col}")
            
            # Ajouter un suffixe unique pour éviter les conflits
            file_suffix = file_path.stem.replace('bbdd_', '').replace('2', '')
            df_renamed = df.copy()
            
            # Renommer toutes les colonnes sauf PtID
            for col in df.columns:
                if col != 'PtID':
                    new_name = f"{col}_{file_suffix}"
                    df_renamed = df_renamed.rename(columns={col: new_name})
            
            if text_data is None:
                text_data = df_renamed
            else:
                text_data = pd.merge(text_data, df_renamed, on='PtID', how='outer')
                
        print(f"Données textuelles: {text_data.shape if text_data is not None else 'Aucune'}")
    
    # Charger les données de séries temporelles
    ts_dir = preprocessed_path / "time_series"
    ts_data = None
    if ts_dir.exists():
        print("Chargement des données de séries temporelles...")
        ts_files = list(ts_dir.glob("*.csv"))
        
        # Traiter chaque fichier et agréger par patient
        patient_aggregated_data = {}
        
        for file_path in ts_files:
            print(f"  Traitement de {file_path.name}...")
            
            # Déterminer le séparateur
            separator = '|' if 'BDataCGM' in file_path.name else ','
            df = pd.read_csv(file_path, sep=separator)
            
            # Nettoyer les colonnes problématiques
            cols_to_remove = ['label_encoded', 'Unnamed: 0']
            for col in cols_to_remove:
                if col in df.columns:
                    df = df.drop(col, axis=1)
                    print(f"    Suppression de la colonne: {col}")
            
            # Identifier la colonne PtID
            patient_id_col = None
            for col in ['PtID', 'patient_id', 'ptid', 'PatientID']:
                if col in df.columns:
                    patient_id_col = col
                    break
            
            if patient_id_col is None:
                print(f"    Attention: Pas de colonne PtID trouvée dans {file_path.name}, ignoré")
                continue
            
            print(f"    Colonne PtID trouvée: {patient_id_col}")
            print(f"    Shape avant agrégation: {df.shape}")
            
            # Agréger par patient
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols if col != patient_id_col]
            
            if len(numeric_cols) > 0:
                # Calculer des statistiques agrégées pour chaque patient
                file_suffix = file_path.stem.replace('_', '')
                
                agg_funcs = ['mean', 'std', 'min', 'max', 'count']
                patient_stats = []
                
                for pt_id in df[patient_id_col].unique():
                    if pd.isna(pt_id):
                        continue
                        
                    patient_data = df[df[patient_id_col] == pt_id]
                    stats_row = {'PtID': int(pt_id)}
                    
                    for col in numeric_cols:
                        if col in patient_data.columns:
                            col_data = pd.to_numeric(patient_data[col], errors='coerce').dropna()
                            if len(col_data) > 0:
                                for func in agg_funcs:
                                    if func == 'count':
                                        stats_row[f"{col}_{func}_{file_suffix}"] = len(col_data)
                                    elif func == 'mean':
                                        stats_row[f"{col}_{func}_{file_suffix}"] = col_data.mean()
                                    elif func == 'std':
                                        stats_row[f"{col}_{func}_{file_suffix}"] = col_data.std() if len(col_data) > 1 else 0
                                    elif func == 'min':
                                        stats_row[f"{col}_{func}_{file_suffix}"] = col_data.min()
                                    elif func == 'max':
                                        stats_row[f"{col}_{func}_{file_suffix}"] = col_data.max()
                    
                    patient_stats.append(stats_row)
                
                if patient_stats:
                    df_aggregated = pd.DataFrame(patient_stats)
                    print(f"    Shape après agrégation: {df_aggregated.shape}")
                    print(f"    Patients uniques: {len(df_aggregated)}")
                    
                    if ts_data is None:
                        ts_data = df_aggregated
                    else:
                        ts_data = pd.merge(ts_data, df_aggregated, on='PtID', how='outer')
                        
        print(f"Données time series: {ts_data.shape if ts_data is not None else 'Aucune'}")
    
    # Charger le roster pour les labels
    roster_path = preprocessed_path / "BPtRoster.txt"
    roster_data = None
    if roster_path.exists():
        roster_data = pd.read_csv(roster_path, sep='|')
        print(f"Roster: {roster_data.shape}")
    
    return tabular_data, text_data, ts_data, roster_data


def main():
    """
    Fonction principale pour encoder les trois types de données
    """
    print("=" * 60)
    print("ENCODAGE MULTI-MODAL DES DONNÉES MÉDICALES")
    print("=" * 60)
    
    # Chargement des données
    preprocessed_dir = Path("code/preprocessed")
    tabular_data, text_data, ts_data, roster_data = load_data_by_type(preprocessed_dir)
    
    # Initialiser les encodeurs
    tabular_encoder = TabularEncoder(n_features=112, output_dim=112)
    text_encoder = TextEncoder(output_dim=768)
    ts_encoder = TS2VecEncoder(output_dim=128, max_seq_length=2880)  # TS2Vec remplace TimeSeriesTransformerEncoder
    
    encoded_results = {}
    
    # Encoder les données tabulaires
    if tabular_data is not None:
        print("\n" + "="*40)
        print("ENCODAGE DONNÉES TABULAIRES")
        print("="*40)
        
        # Préparer les données (retirer PtID pour l'encodage)
        X_tabular = tabular_data.drop('PtID', axis=1, errors='ignore')
        
        # Joindre avec roster pour avoir les labels si disponible
        y_tabular = None
        if roster_data is not None:
            merged = pd.merge(tabular_data[['PtID']], roster_data, on='PtID', how='left')
            if 'BCaseControlStatus' in merged.columns:
                # Encoder les labels
                le = LabelEncoder()
                y_tabular = le.fit_transform(merged['BCaseControlStatus'].fillna('Unknown'))
        
        # Encoder
        tabular_embeddings = tabular_encoder.fit_transform(X_tabular, y_tabular)
        
        # Sauvegarder
        tabular_result = pd.DataFrame(tabular_embeddings)
        tabular_result.insert(0, 'PtID', tabular_data['PtID'].values)
        tabular_result.to_csv("encoded_features/tabular_encoded.csv", index=False)
        
        encoded_results['tabular'] = tabular_result
        print(f"Données tabulaires encodées: {tabular_embeddings.shape}")
    
    # Encoder les données textuelles
    if text_data is not None:
        print("\n" + "="*40)
        print("ENCODAGE DONNÉES TEXTUELLES")
        print("="*40)
        
        # Préparer les données
        X_text = text_data.drop('PtID', axis=1, errors='ignore')
        
        # Encoder
        text_embeddings = text_encoder.fit_transform(X_text)
        
        # Sauvegarder
        text_result = pd.DataFrame(text_embeddings)
        text_result.insert(0, 'PtID', text_data['PtID'].values)
        text_result.to_csv("encoded_features/text_encoded.csv", index=False)
        
        encoded_results['text'] = text_result
        print(f"Données textuelles encodées: {text_embeddings.shape}")
    
    # Encoder les données de séries temporelles
    if ts_data is not None:
        print("\n" + "="*40)
        print("ENCODAGE DONNÉES TIME SERIES")
        print("="*40)
        
        # Préparer les données (retirer PtID pour l'encodage)
        X_ts = ts_data.drop('PtID', axis=1, errors='ignore')
        
        # Encoder
        ts_embeddings = ts_encoder.fit_transform(X_ts)
        
        # Sauvegarder
        ts_result = pd.DataFrame(ts_embeddings)
        ts_result.insert(0, 'PtID', ts_data['PtID'].values)
        ts_result.to_csv("encoded_features/timeseries_encoded.csv", index=False)
        
        encoded_results['timeseries'] = ts_result
        print(f"Données time series encodées: {ts_embeddings.shape}")
    
    print("\n" + "="*60)
    print("ENCODAGE TERMINÉ")
    print("="*60)
    print("Fichiers créés dans le dossier 'encoded_features/':")
    for data_type, result in encoded_results.items():
        print(f"  - {data_type}_encoded.csv: {result.shape}")
    
    # NOUVELLE PARTIE: Neural Feature Selection + Fusion
    print("\n" + "="*60)
    print("DÉMARRAGE NEURAL FEATURE SELECTION + FUSION")
    print("="*60)
    
    # Demander à l'utilisateur de choisir la méthode de fusion
    fusion_method = choose_fusion_method()
    
    # Paramètres de réduction (ajustables)
    tabular_target_features = 50  # Réduire à 50 features
    text_target_features = 100    # Réduire à 100 features  
    ts_target_features = 32       # Réduire à 32 features
    
    # Appliquer neural feature selection + fusion choisie
    fused_data, reduced_features = neural_feature_selection_and_fusion(
        encoded_results=encoded_results,
        roster_data=roster_data,
        tabular_features=tabular_target_features,
        text_features=text_target_features, 
        ts_features=ts_target_features,
        fusion_method=fusion_method
    )
    
    # Sauvegarder les résultats
    print("\n" + "="*60)
    print("SAUVEGARDE DES RÉSULTATS")
    print("="*60)
    
    # Sauvegarder les features réduites par modalité
    for modality, features_df in reduced_features.items():
        filename = f"encoded_features/{modality}_reduced.csv"
        features_df.to_csv(filename, index=False)
        print(f"  - {filename}: {features_df.shape}")
    
    # Sauvegarder le dataset fusionné final
    final_filename = "encoded_features/fused_multimodal_dataset.csv"
    fused_data.to_csv(final_filename, index=False)
    print(f"  - {final_filename}: {fused_data.shape}")
    
    # Créer une version avec seulement les patients ayant toutes les modalités
    complete_data = fused_data.dropna()
    if len(complete_data) > 0:
        complete_filename = "encoded_features/complete_fused_dataset.csv"
        complete_data.to_csv(complete_filename, index=False)
        print(f"  - {complete_filename}: {complete_data.shape}")
        print(f"    (Patients avec toutes les modalités)")
    
    print("\n" + "="*60)
    print("PIPELINE COMPLET TERMINÉ !")
    print("="*60)
    print("📁 Fichiers disponibles:")
    print("  🔹 Encodages bruts: tabular_encoded.csv, text_encoded.csv, timeseries_encoded.csv")
    print("  🔹 Features réduites: tabular_reduced.csv, text_reduced.csv, timeseries_reduced.csv") 
    print("  🔹 Dataset fusionné: fused_multimodal_dataset.csv")
    print("  🔹 Dataset complet: complete_fused_dataset.csv")
    print("\n🚀 Prêt pour la classification ML !")


def neural_feature_selection_and_fusion(encoded_results, roster_data, 
                                        tabular_features=50, text_features=100, ts_features=32,
                                        fusion_method='cross_attention'):
    """
    Applique feature selection/reduction avec des réseaux de neurones sur chaque modalité 
    puis utilise la méthode de fusion choisie pour fusionner les modalités
    
    Args:
        encoded_results: Dict avec les résultats encodés de chaque modalité
        roster_data: DataFrame avec PtID et BCaseControlStatus
        tabular_features: Nombre de features à garder pour tabular
        text_features: Nombre de features à garder pour text  
        ts_features: Nombre de features à garder pour time series
        fusion_method: 'early' pour early fusion, 'cross_attention' pour cross-attention fusion
    """
    print("\n" + "="*60)
    print("NEURAL FEATURE SELECTION + FUSION")
    print("="*60)
    print(f"Méthode de fusion choisie: {fusion_method.upper()}")
    
    # Charger le roster pour avoir les labels
    if roster_data is not None:
        # Encoder les labels pour la feature selection
        le = LabelEncoder()
        labels = le.fit_transform(roster_data['BCaseControlStatus'].fillna('Unknown'))
        labels_df = pd.DataFrame({
            'PtID': roster_data['PtID'], 
            'BCaseControlStatus': roster_data['BCaseControlStatus'],
            'label_encoded': labels
        })
    else:
        labels_df = None
        print("Attention: Pas de labels disponibles, feature selection non-supervisée")
    
    reduced_features = {}
    
    # 1. NEURAL FEATURE SELECTION TABULAR
    if 'tabular' in encoded_results:
        print("\n" + "-"*40)
        print("NEURAL FEATURE SELECTION TABULAIRES")
        print("-"*40)
        
        tabular_df = encoded_results['tabular']
        X_tabular = tabular_df.drop('PtID', axis=1).values
        
        # Joindre avec les labels si disponible
        y_tabular = None
        if labels_df is not None:
            merged_tabular = pd.merge(tabular_df[['PtID']], labels_df, on='PtID', how='left')
            y_tabular = merged_tabular['label_encoded'].fillna(-1).values  # -1 pour les inconnus
        
        # Neural feature selection
        print(f"  Neural network feature selection: {X_tabular.shape[1]} → {tabular_features}")
        nn_selector_tabular = NeuralFeatureSelector(
            target_features=tabular_features,
            hidden_dims=[256, 128, 64],
            epochs=80,
            learning_rate=0.001,
            batch_size=16,
            patience=10
        )
        
        X_tabular_reduced = nn_selector_tabular.fit_transform(X_tabular, y_tabular)
        
        # Créer DataFrame avec les features réduites
        tabular_cols = [f'tabular_nn_feat_{i}' for i in range(X_tabular_reduced.shape[1])]
        reduced_tabular = pd.DataFrame(X_tabular_reduced, columns=tabular_cols)
        reduced_tabular.insert(0, 'PtID', tabular_df['PtID'].values)
        
        reduced_features['tabular'] = reduced_tabular
        print(f"  Résultat: {X_tabular_reduced.shape}")
        
        # Afficher les features les plus importantes
        feature_importance = nn_selector_tabular.feature_importance_
        top_features_idx = np.argsort(feature_importance)[-10:]  # Top 10
        print(f"  Top 10 features importantes (indices): {top_features_idx}")
    
    # 2. NEURAL FEATURE SELECTION TEXT
    if 'text' in encoded_results:
        print("\n" + "-"*40)
        print("NEURAL FEATURE SELECTION TEXTUELLES")
        print("-"*40)
        
        text_df = encoded_results['text']
        X_text = text_df.drop('PtID', axis=1).values
        
        # Joindre avec les labels si disponible
        y_text = None
        if labels_df is not None:
            merged_text = pd.merge(text_df[['PtID']], labels_df, on='PtID', how='left')
            y_text = merged_text['label_encoded'].fillna(-1).values
        
        # Neural feature selection pour les embeddings text
        print(f"  Neural network feature selection: {X_text.shape[1]} → {text_features}")
        nn_selector_text = NeuralFeatureSelector(
            target_features=text_features,
            hidden_dims=[512, 256, 128],  # Plus large pour les embeddings text
            epochs=60,
            learning_rate=0.0005,
            batch_size=16,
            patience=8
        )
        
        X_text_reduced = nn_selector_text.fit_transform(X_text, y_text)
        
        # Créer DataFrame avec les features réduites
        text_cols = [f'text_nn_feat_{i}' for i in range(X_text_reduced.shape[1])]
        reduced_text = pd.DataFrame(X_text_reduced, columns=text_cols)
        reduced_text.insert(0, 'PtID', text_df['PtID'].values)
        
        reduced_features['text'] = reduced_text
        print(f"  Résultat: {X_text_reduced.shape}")
    
    # 3. NEURAL FEATURE SELECTION TIME SERIES
    if 'timeseries' in encoded_results:
        print("\n" + "-"*40)
        print("NEURAL FEATURE SELECTION TIME SERIES")
        print("-"*40)
        
        ts_df = encoded_results['timeseries']
        X_ts = ts_df.drop('PtID', axis=1).values
        
        # Joindre avec les labels si disponible
        y_ts = None
        if labels_df is not None:
            merged_ts = pd.merge(ts_df[['PtID']], labels_df, on='PtID', how='left')
            y_ts = merged_ts['label_encoded'].fillna(-1).values
        
        # Neural feature selection pour time series
        print(f"  Neural network feature selection: {X_ts.shape[1]} → {ts_features}")
        nn_selector_ts = NeuralFeatureSelector(
            target_features=min(ts_features, X_ts.shape[1]),
            hidden_dims=[128, 64, 32],  # Plus petit pour time series
            epochs=100,
            learning_rate=0.001,
            batch_size=16,
            patience=15
        )
        
        X_ts_reduced = nn_selector_ts.fit_transform(X_ts, y_ts)
        
        # Créer DataFrame avec les features réduites
        ts_cols = [f'ts_nn_feat_{i}' for i in range(X_ts_reduced.shape[1])]
        reduced_ts = pd.DataFrame(X_ts_reduced, columns=ts_cols)
        reduced_ts.insert(0, 'PtID', ts_df['PtID'].values)
        
        reduced_features['timeseries'] = reduced_ts
        print(f"  Résultat: {X_ts_reduced.shape}")
    
    # 4. FUSION DES MODALITÉS SELON LA MÉTHODE CHOISIE
    print("\n" + "-"*40)
    print(f"FUSION - MÉTHODE: {fusion_method.upper()}")
    print("-"*40)
    
    # Déterminer les dimensions de chaque modalité (nécessaire pour les statistiques)
    modality_dims = {}
    for modality, features_df in reduced_features.items():
        feature_cols = [col for col in features_df.columns if col != 'PtID']
        modality_dims[modality] = len(feature_cols)
    
    if fusion_method == 'early':
        # Early Fusion - Concaténation simple
        print("  🔗 Application Early Fusion (concaténation)...")
        fused_data = simple_early_fusion(reduced_features, labels_df)
        fusion_method_name = "EARLY FUSION"
    
    elif fusion_method == 'cross_attention':
        # Cross-Attention Fusion
        print("  🧠 Application Cross-Attention Fusion...")
        
        # Initialiser le modèle de cross-attention
        cross_attention_fusion = CrossAttentionFusion(d_model=256, n_heads=8)
        
        # Afficher les dimensions de chaque modalité
        for modality, dim in modality_dims.items():
            print(f"    {modality}: {dim} features")
        
        # Fitter le modèle avec les bonnes dimensions
        cross_attention_fusion.fit(
            tabular_dim=modality_dims.get('tabular', 50),
            text_dim=modality_dims.get('text', 100), 
            ts_dim=modality_dims.get('timeseries', 32)
        )
        
        # Appliquer le cross-attention fusion
        print("    Applying cross-attention between modalities...")
        cross_attn_features = cross_attention_fusion.transform(reduced_features)
        print(f"    Cross-attention output: {cross_attn_features.shape}")
        
        # Joindre avec les labels du roster
        if labels_df is not None:
            fused_data = pd.merge(labels_df[['PtID', 'BCaseControlStatus']], 
                                 cross_attn_features, on='PtID', how='left')
            print(f"    Fusion avec roster: {fused_data.shape}")
        else:
            fused_data = cross_attn_features
            print(f"    Pas de roster disponible: {fused_data.shape}")
        
        fusion_method_name = "CROSS-ATTENTION FUSION"
    
    elif fusion_method == 'cls_transformer':
        # CLS Transformer Fusion
        print("  🎯 Application CLS Transformer Fusion...")
        
        # Initialiser le modèle CLS Transformer
        cls_transformer = CLSTransformerFusion(d_model=256, n_heads=8, n_layers=3, dropout=0.1, n_classes=2)
        
        # Afficher les dimensions de chaque modalité
        for modality, dim in modality_dims.items():
            print(f"    {modality}: {dim} features")
        
        # Fitter le modèle avec les bonnes dimensions
        cls_transformer.fit(
            tabular_dim=modality_dims.get('tabular', 50),
            text_dim=modality_dims.get('text', 100), 
            ts_dim=modality_dims.get('timeseries', 32)
        )
        
        # Appliquer le CLS transformer fusion
        print("    Applying CLS transformer fusion...")
        cls_features = cls_transformer.transform(reduced_features)
        print(f"    CLS transformer output: {cls_features.shape}")
        
        # Joindre avec les labels du roster
        if labels_df is not None:
            fused_data = pd.merge(labels_df[['PtID', 'BCaseControlStatus']], 
                                 cls_features, on='PtID', how='left')
            print(f"    Fusion avec roster: {fused_data.shape}")
        else:
            fused_data = cls_features
            print(f"    Pas de roster disponible: {fused_data.shape}")
        
        fusion_method_name = "CLS TRANSFORMER FUSION"
    
    else:
        raise ValueError(f"Méthode de fusion inconnue: {fusion_method}")
    
    # Statistiques finales
    print(f"\n" + "-"*40)
    print(f"STATISTIQUES FUSION FINALE ({fusion_method_name})")
    print("-"*40)
    
    feature_cols = [col for col in fused_data.columns if col not in ['PtID', 'BCaseControlStatus']]
    missing_counts = fused_data[feature_cols].isnull().sum(axis=1)
    
    print(f"  Patients total: {len(fused_data)}")
    print(f"  Features fusionnées: {len(feature_cols)}")
    
    # Calculer les dimensions d'origine pour les statistiques
    original_features = 0
    if fusion_method == 'cross_attention':
        # Pour cross-attention, on calcule les dimensions depuis les modalités réduites
        for modality, features_df in reduced_features.items():
            modal_feature_cols = [col for col in features_df.columns if col != 'PtID']
            original_features += len(modal_feature_cols)
    else:
        # Pour early fusion, les features sont la somme des modalités réduites
        original_features = len(feature_cols)
    
    print(f"  Dimension finale: {len(feature_cols)}")
    
    # Statistiques par modalité originale
    patients_with_all_modalities = 0
    patients_with_partial_data = 0
    
    for _, row in fused_data.iterrows():
        pt_id = row['PtID']
        has_modalities = []
        
        for modality in reduced_features.keys():
            modality_df = reduced_features[modality]
            has_modality = pt_id in modality_df['PtID'].values
            has_modalities.append(has_modality)
        
        if all(has_modalities):
            patients_with_all_modalities += 1
        elif any(has_modalities):
            patients_with_partial_data += 1
    
    print(f"  Patients avec toutes les modalités: {patients_with_all_modalities}")
    print(f"  Patients avec données partielles: {patients_with_partial_data}")
    
    # Distribution par modalité
    for modality in reduced_features.keys():
        modality_df = reduced_features[modality]
        patients_with_modality = len(set(fused_data['PtID']) & set(modality_df['PtID']))
        patients_without_modality = len(fused_data) - patients_with_modality
        print(f"  Patients avec {modality}: {patients_with_modality}")
        print(f"  Patients sans {modality}: {patients_without_modality}")
    
    if 'BCaseControlStatus' in fused_data.columns:
        target_dist = fused_data['BCaseControlStatus'].value_counts()
        print(f"\n  Distribution de la variable cible:")
        for status, count in target_dist.items():
            percentage = (count / len(fused_data)) * 100
            print(f"    {status}: {count} ({percentage:.1f}%)")
    
    print(f"\n  🚀 Cross-attention fusion terminée !")
    print(f"  📊 Features d'origine: {sum(modality_dims.values())}")
    print(f"  📊 Features fusionnées: {len(feature_cols)}")
    print(f"  📊 Réduction: {((sum(modality_dims.values()) - len(feature_cols)) / sum(modality_dims.values()) * 100):.1f}%")
    
    return fused_data, reduced_features
    


def simple_early_fusion(reduced_features, labels_df=None):
    """
    Simple early fusion par concaténation des features réduites
    
    Args:
        reduced_features: Dict avec les DataFrames de chaque modalité
        labels_df: DataFrame avec les labels du roster
    """
    print("\n" + "-"*40)
    print("EARLY FUSION - CONCATENATION")
    print("-"*40)
    
    # Commencer avec le roster pour avoir tous les patients et les labels
    if labels_df is not None:
        fused_data = labels_df[['PtID', 'BCaseControlStatus']].copy()
        print(f"Base avec roster: {fused_data.shape}")
    else:
        # Si pas de roster, commencer avec la première modalité disponible
        first_modality = list(reduced_features.keys())[0]
        fused_data = reduced_features[first_modality][['PtID']].copy()
        print(f"Base sans roster: {fused_data.shape}")
    
    # Fusionner chaque modalité
    total_features = 0
    for modality, features_df in reduced_features.items():
        print(f"  Ajout {modality}: {features_df.shape[1]-1} features (NN-selected)")
        fused_data = pd.merge(fused_data, features_df, on='PtID', how='left')
        total_features += features_df.shape[1] - 1  # -1 pour PtID
        print(f"    → Dimensions après fusion: {fused_data.shape}")
    
    # Statistiques finales
    print(f"\n" + "-"*40)
    print("STATISTIQUES FUSION FINALE (EARLY FUSION)")
    print("-"*40)
    
    feature_cols = [col for col in fused_data.columns if col not in ['PtID', 'BCaseControlStatus']]
    missing_counts = fused_data[feature_cols].isnull().sum(axis=1)
    
    print(f"  Patients total: {len(fused_data)}")
    print(f"  Features total (concaténées): {len(feature_cols)}")
    print(f"  Patients avec toutes les modalités: {(missing_counts == 0).sum()}")
    print(f"  Patients avec données partielles: {(missing_counts > 0).sum()}")
    
    # Distribution par modalité
    for modality in reduced_features.keys():
        modality_cols = [col for col in feature_cols if col.startswith(f"{modality}_")]
        modality_missing = fused_data[modality_cols].isnull().all(axis=1).sum()
        print(f"  Patients sans {modality}: {modality_missing}")
    
    if 'BCaseControlStatus' in fused_data.columns:
        target_dist = fused_data['BCaseControlStatus'].value_counts()
        print(f"\n  Distribution de la variable cible:")
        for status, count in target_dist.items():
            percentage = (count / len(fused_data)) * 100
            print(f"    {status}: {count} ({percentage:.1f}%)")
    
    print(f"\n  🚀 Early fusion terminée !")
    print(f"  📊 Features d'origine: {sum(len([col for col in df.columns if col != 'PtID']) for df in reduced_features.values())}")
    print(f"  📊 Features fusionnées: {len(feature_cols)}")
    
    return fused_data


def cross_attention_fusion(reduced_features, labels_df=None):
    """
    Cross-attention fusion des features réduites
    
    Args:
        reduced_features: Dict avec les DataFrames de chaque modalité
        labels_df: DataFrame avec les labels du roster
    """
    print("\n" + "-"*40)
    print("CROSS-ATTENTION FUSION")
    print("-"*40)
    
    # Initialiser le modèle de cross-attention
    cross_attention_fusion_model = CrossAttentionFusion(d_model=256, n_heads=8)
    
    # Déterminer les dimensions de chaque modalité
    modality_dims = {}
    for modality, features_df in reduced_features.items():
        feature_cols = [col for col in features_df.columns if col != 'PtID']
        modality_dims[modality] = len(feature_cols)
        print(f"  {modality}: {len(feature_cols)} features")
    
    # Fitter le modèle avec les bonnes dimensions
    cross_attention_fusion_model.fit(
        tabular_dim=modality_dims.get('tabular', 50),
        text_dim=modality_dims.get('text', 100), 
        ts_dim=modality_dims.get('timeseries', 32)
    )
    
    # Appliquer le cross-attention fusion
    print("  Applying cross-attention between modalities...")
    cross_attn_features = cross_attention_fusion_model.transform(reduced_features)
    print(f"  Cross-attention output: {cross_attn_features.shape}")
    
    # Joindre avec les labels du roster
    if labels_df is not None:
        fused_data = pd.merge(labels_df[['PtID', 'BCaseControlStatus']], 
                             cross_attn_features, on='PtID', how='left')
        print(f"  Fusion avec roster: {fused_data.shape}")
    else:
        fused_data = cross_attn_features
        print(f"  Pas de roster disponible: {fused_data.shape}")
    
    # Statistiques finales
    print(f"\n" + "-"*40)
    print("STATISTIQUES FUSION FINALE (CROSS-ATTENTION)")
    print("-"*40)
    
    feature_cols = [col for col in fused_data.columns if col not in ['PtID', 'BCaseControlStatus']]
    missing_counts = fused_data[feature_cols].isnull().sum(axis=1)
    
    print(f"  Patients total: {len(fused_data)}")
    print(f"  Features fusionnées (cross-attention): {len(feature_cols)}")
    print(f"  Dimension finale: {cross_attn_features.shape[1] - 1}")  # -1 pour PtID
    
    # Statistiques par modalité originale
    patients_with_all_modalities = 0
    patients_with_partial_data = 0
    
    for _, row in fused_data.iterrows():
        pt_id = row['PtID']
        has_modalities = []
        
        for modality in reduced_features.keys():
            modality_df = reduced_features[modality]
            has_modality = pt_id in modality_df['PtID'].values
            has_modalities.append(has_modality)
        
        if all(has_modalities):
            patients_with_all_modalities += 1
        elif any(has_modalities):
            patients_with_partial_data += 1
    
    print(f"  Patients avec toutes les modalités: {patients_with_all_modalities}")
    print(f"  Patients avec données partielles: {patients_with_partial_data}")
    
    # Distribution par modalité
    for modality in reduced_features.keys():
        modality_df = reduced_features[modality]
        patients_with_modality = len(set(fused_data['PtID']) & set(modality_df['PtID']))
        patients_without_modality = len(fused_data) - patients_with_modality
        print(f"  Patients avec {modality}: {patients_with_modality}")
        print(f"  Patients sans {modality}: {patients_without_modality}")
    
    if 'BCaseControlStatus' in fused_data.columns:
        target_dist = fused_data['BCaseControlStatus'].value_counts()
        print(f"\n  Distribution de la variable cible:")
        for status, count in target_dist.items():
            percentage = (count / len(fused_data)) * 100
            print(f"    {status}: {count} ({percentage:.1f}%)")
    
    print(f"\n  🚀 Cross-attention fusion terminée !")
    print(f"  📊 Features d'origine: {sum(modality_dims.values())}")
    print(f"  📊 Features fusionnées: {len(feature_cols)}")
    print(f"  📊 Réduction: {((sum(modality_dims.values()) - len(feature_cols)) / sum(modality_dims.values()) * 100):.1f}%")
    
    return fused_data


def choose_fusion_method():
    """
    Permet à l'utilisateur de choisir la méthode de fusion
    """
    print("\n" + "="*60)
    print("CHOIX DE LA MÉTHODE DE FUSION")
    print("="*60)
    print("Trois méthodes de fusion sont disponibles :")
    print()
    print("1. 🔗 EARLY FUSION (Concaténation simple)")
    print("   - Concatène directement les features de chaque modalité")
    print("   - Rapide et simple")
    print("   - Dimension finale = somme des features de chaque modalité")
    print()
    print("2. 🧠 CROSS-ATTENTION FUSION (Fusion intelligente)")
    print("   - Chaque modalité 'attend' aux autres modalités")
    print("   - Interactions complexes entre modalités")
    print("   - Dimension finale fixe (256 features)")
    print()
    print("3. 🎯 CLS TRANSFORMER FUSION (Token CLS + Classification)")
    print("   - Token CLS learnable comme dans BERT")
    print("   - Transformer encoder pour fusion automatique")
    print("   - Classification head intégrée")
    print("   - Dimension finale fixe (256 features)")
    print()
    
    while True:
        try:
            choice = input("Votre choix (1, 2 ou 3) : ").strip()
            if choice == '1':
                print("\n✅ Early Fusion sélectionnée")
                return 'early'
            elif choice == '2':
                print("\n✅ Cross-Attention Fusion sélectionnée")
                return 'cross_attention'
            elif choice == '3':
                print("\n✅ CLS Transformer Fusion sélectionnée")
                return 'cls_transformer'
            else:
                print("❌ Veuillez entrer 1, 2 ou 3")
        except KeyboardInterrupt:
            print("\n\n❌ Arrêt du programme")
            exit(1)
        except Exception as e:
            print(f"❌ Erreur : {e}")

if __name__ == "__main__":
    # Créer le dossier de sortie
    Path("encoded_features").mkdir(exist_ok=True)
    
    # Lancer l'encodage complet avec feature selection et fusion
    main()



