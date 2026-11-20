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
import os
import sys

warnings.filterwarnings('ignore')


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
    preprocessed_dir = Path("preprocessed")  # data/preprocessed (Figshare package), see README
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
    
    # Labels of the participants with the three modalities, in the order of the roster -> data/labels.csv
    common_ids = set.intersection(*(set(r['PtID']) for r in encoded_results.values()))
    labels = roster_data[['PtID', 'BCaseControlStatus']]
    labels = labels[labels['PtID'].isin(common_ids)].dropna()
    labels.to_csv("labels.csv", index=False)
    print(f"  - labels.csv: {labels.shape}")


if __name__ == "__main__":
    # All paths are relative to the data/ folder of the repository: inputs in data/preprocessed/, outputs in
    # data/encoded_features/ and data/labels.csv. Usage: python preprocessing/processing_tables.py
    os.chdir(Path(__file__).resolve().parents[1] / "data")
    # Créer le dossier de sortie
    Path("encoded_features").mkdir(exist_ok=True)
    
    # Lancer l'encodage des trois modalités
    main()


