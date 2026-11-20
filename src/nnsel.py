import numpy as np, torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler
class FeatureSelectionNN(nn.Module):
    """
    Feed-forward Neural Network pour la sélection de features
    """
    def __init__(self, input_dim, output_dim, hidden_dims=[256, 128, 64], dropout=0.3):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Construire les couches
        #  Constructs model layers
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
        pass #print(f"    Entraînement NN pour sélection de features: {X.shape[1]} → {self.target_features}")
        
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
        pass #print(f"    Features sélectionnées par NN: {len(self.selected_features_)}")
        
        return self
    
    def _train_supervised(self, X, y):
        """Entraînement supervisé pour classification"""
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        
        # Dataset et DataLoader
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=(len(dataset)%self.batch_size==1))
        
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
                    pass #print(f"    Early stopping à l'époque {epoch}")
                    break
            
            if (epoch + 1) % 20 == 0:
                pass #print(f"    Époque {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")
    
    def _train_unsupervised(self, X):
        """Entraînement non-supervisé (autoencoder)"""
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Dataset et DataLoader
        dataset = torch.utils.data.TensorDataset(X)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=(len(dataset)%self.batch_size==1))
        
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
                pass #print(f"    Époque {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")
    
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

