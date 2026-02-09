import torch
import torch.nn.functional as F
import numpy as np
import pickle
from typing import Tuple, Dict, Optional
from collections import deque
from .model import LOBModel
from core.config import ModelConfig
import logging

logger = logging.getLogger(__name__)

class MambaLOBPredictor:
    """
    Agent RL pour le Market Making (Inference uniquement).
    Utilise une architecture Mamba-LOB (SSM + CNN) avec Invariant Normalization.
    """
    def __init__(self, 
                 model_path: str,
                 config: Optional[ModelConfig] = None,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.config = config if config else ModelConfig()
        self.model = self._load_model(model_path)
        
        # SOTA: Use the common LOBFeatureProcessor for consistency
        from data.processor import LOBFeatureProcessor, MarketConfig
        self.market_config = MarketConfig()
        self.feature_processor = LOBFeatureProcessor(market_config=self.market_config)
        
        # Buffer pour la séquence historique
        self.history = deque(maxlen=self.config.window_size)

    def _load_model(self, model_path: str) -> Optional[LOBModel]:
        """Charge le modèle RL pré-entraîné."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            # Reconstruire la config depuis le checkpoint si possible
            if 'model_config' in checkpoint:
                self.config = ModelConfig(**checkpoint['model_config']) if isinstance(checkpoint['model_config'], dict) else checkpoint['model_config']
            
            model = LOBModel(self.config)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)
            model.to(self.device)
            model.eval()
            logger.info(f"Mamba Agent Loaded from {model_path} with window_size={self.config.window_size}")
            return model
        except Exception as e:
            logger.error(f"Failed to load Mamba model from {model_path}: {e}")
            return None

    def process_orderbook(self, orderbook: Dict) -> np.ndarray:
        """Extrait les features via le processeur central pour cohérence SOTA."""
        return self.feature_processor.process(orderbook)

    def get_action(self, orderbook: Dict) -> Tuple[float, float, bool]:
        """
        Retourne l'action de l'agent RL.
        Returns: (AI_Bias, AI_Spread_Adj, Is_Ready)
        """
        # 1. Extraire et Bufferiser
        features = self.process_orderbook(orderbook)
        self.history.append(features)
        
        # 2. Vérifier si on a assez d'historique (Sequence length from config)
        if len(self.history) < self.config.window_size:
            return 0.0, 0.0, False
            
        # 3. Créer le tenseur d'état (1, 50, 40)
        state_tensor = np.array(self.history) # (50, 40)
        
        try:
            # 4. Inférence Agent
            if self.model is None:
                return 0.0, 0.0, False # Untrained / No Model State

            # select_action retourne numpy array action
            # Mock de state dict pour select_action
            state = {'lob_features': state_tensor}
            
            # L'action est typiquement [Bias, Spread] (valeurs tanh entre -1 et 1)
            raw_action = self.model.select_action(state)
            
            ai_bias = float(raw_action[0]) # -1 (Sell) à +1 (Buy)
            ai_spread = float(raw_action[1]) # -1 (Tight) à +1 (Wide)
            
            return ai_bias, ai_spread, True
            
        except Exception as e:
            logger.error(f"Inference Error: {e}")
            return 0.0, 0.0, False
