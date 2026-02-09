# data_loader.py
import json
import os
import glob
import zipfile
import pickle
import logging
import numpy as np
from collections import OrderedDict
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class LOBBook:
    """Maintains a full OrderBook image from snapshots and deltas."""
    __slots__ = ['bids', 'asks', 'last_ts', 'is_initialized']

    def __init__(self):
        self.bids = {}
        self.asks = {}
        self.last_ts = 0
        self.is_initialized = False

    def update(self, packet):
        data = packet.get('data', {})
        packet_type = packet.get('type')

        if packet_type == 'snapshot':
            self.bids = {float(p): float(q) for p, q in data.get('b', [])}
            self.asks = {float(p): float(q) for p, q in data.get('a', [])}
            self.is_initialized = True
        elif packet_type == 'delta' and self.is_initialized:
            for p, q in data.get('b', []):
                price, qty = float(p), float(q)
                if qty == 0:
                    self.bids.pop(price, None)
                else:
                    self.bids[price] = qty
            for p, q in data.get('a', []):
                price, qty = float(p), float(q)
                if qty == 0:
                    self.asks.pop(price, None)
                else:
                    self.asks[price] = qty

        self.last_ts = packet.get('ts', 0)
        return self.is_initialized

    def get_orderbook(self):
        sorted_bids = sorted(self.bids.items(), key=lambda x: x[0], reverse=True)[:10]
        sorted_asks = sorted(self.asks.items(), key=lambda x: x[0])[:10]
        return {
            'bids': [[str(p), str(q)] for p, q in sorted_bids],
            'asks': [[str(p), str(q)] for p, q in sorted_asks],
            'local_timestamp': self.last_ts
        }


class MarketDataLoader:
    """Charge et gère les données du carnet d'ordres.

    Modes:
    - Fichier unique (.json, .data, .data.zip): charge tout en mémoire
    - Répertoire (lazy): stocke la liste des fichiers, charge un fichier
      aléatoire par appel à reset() pour limiter la RAM (~200MB par jour)

    Features:
    - Pickle cache: ZIP files are decompressed once, cached as .pkl for fast reload
    - LRU memory cache: keeps last 3 files in RAM (~600MB max)
    - Train/eval split: chronological split (first 83% train, last 17% eval)
    """

    # LRU memory cache shared across instances
    _lru_cache: OrderedDict = OrderedDict()
    _lru_max_size: int = 3

    def __init__(self, data_path: str, split: Optional[str] = None):
        """
        Args:
            data_path: Path to data directory or single file.
            split: 'train', 'eval', or None (all data). Chronological split.
        """
        logger.info(f"Initialisation du DataLoader avec: {data_path}, split={split}")

        self._lazy_mode = False
        self._file_paths: List[str] = []
        self._current_file: Optional[str] = None
        self._cache_dir: Optional[str] = None

        if os.path.isdir(data_path):
            # Mode lazy: on ne charge PAS tout d'un coup
            zip_files = sorted(glob.glob(os.path.join(data_path, '*_ob200.data.zip')))
            data_files = sorted(glob.glob(os.path.join(data_path, '*_ob200.data')))
            all_files = sorted(set(zip_files + data_files))
            if not all_files:
                raise FileNotFoundError(f"Aucun fichier *_ob200.data[.zip] trouvé dans {data_path}")

            # Setup cache directory
            self._cache_dir = os.path.join(data_path, 'cache')
            os.makedirs(self._cache_dir, exist_ok=True)

            # Chronological train/eval split
            if split == 'train':
                n_train = max(1, int(len(all_files) * 0.83))
                self._file_paths = all_files[:n_train]
                logger.info(f"Train split: {len(self._file_paths)}/{len(all_files)} fichiers")
            elif split == 'eval':
                n_train = max(1, int(len(all_files) * 0.83))
                self._file_paths = all_files[n_train:]
                if not self._file_paths:
                    # Fallback: use last file
                    self._file_paths = all_files[-1:]
                logger.info(f"Eval split: {len(self._file_paths)}/{len(all_files)} fichiers")
            else:
                self._file_paths = all_files

            self._lazy_mode = True
            logger.info(f"Mode lazy: {len(self._file_paths)} fichiers trouvés")

            # Charger le premier fichier pour initialiser
            self.data = self._load_file_cached(self._file_paths[0])
            self._current_file = self._file_paths[0]
        elif data_path.endswith('.data.zip'):
            self.data = self._load_zip_data(data_path)
        elif data_path.endswith('.data'):
            self.data = self._load_data_file(data_path)
        else:
            self.data = self._load_json_data(data_path)

        logger.info(f"Données chargées: {len(self.data)} échantillons")
        self.current_idx = 0

        if self.data:
            sample = self.data[0]
            mid = (float(sample['bids'][0][0]) + float(sample['asks'][0][0])) / 2
            logger.info(f"Premier échantillon - mid_price: {mid}, ts: {sample.get('local_timestamp', 'N/A')}")

    def _load_file(self, path: str) -> List[Dict]:
        """Charge un fichier (.data.zip ou .data) et retourne les orderbooks."""
        if path.endswith('.data.zip'):
            return self._load_zip_data(path)
        elif path.endswith('.data'):
            return self._load_data_file(path)
        else:
            return self._load_json_data(path)

    def _ensure_cache(self, path: str) -> str:
        """Ensure a pickle cache exists for a data file. Returns cache path."""
        if self._cache_dir is None:
            return None
        basename = os.path.basename(path).replace('.data.zip', '.pkl').replace('.data', '.pkl')
        cache_path = os.path.join(self._cache_dir, basename)
        if not os.path.exists(cache_path):
            logger.info(f"Creating pickle cache: {cache_path}")
            data = self._load_file(path)
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Cache created: {len(data)} orderbooks → {cache_path}")
        return cache_path

    def _load_file_cached(self, path: str) -> List[Dict]:
        """Load a file using LRU memory cache + pickle disk cache."""
        # 1. Check LRU memory cache
        if path in MarketDataLoader._lru_cache:
            MarketDataLoader._lru_cache.move_to_end(path)
            logger.info(f"LRU cache hit: {os.path.basename(path)}")
            return MarketDataLoader._lru_cache[path]

        # 2. Check pickle disk cache
        cache_path = self._ensure_cache(path)
        if cache_path and os.path.exists(cache_path):
            logger.info(f"Loading from pickle cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
        else:
            data = self._load_file(path)

        # 3. Store in LRU memory cache
        MarketDataLoader._lru_cache[path] = data
        if len(MarketDataLoader._lru_cache) > MarketDataLoader._lru_max_size:
            evicted_key, _ = MarketDataLoader._lru_cache.popitem(last=False)
            logger.info(f"LRU cache evicted: {os.path.basename(evicted_key)}")

        return data

    def _load_zip_data(self, zip_path: str) -> List[Dict]:
        """Charge un fichier .data.zip et reconstruit les orderbooks."""
        logger.info(f"Chargement du fichier ZIP: {zip_path}")
        if not os.path.exists(zip_path):
            raise FileNotFoundError(f"Le fichier n'existe pas: {zip_path}")

        orderbooks = []
        book = LOBBook()

        try:
            with zipfile.ZipFile(zip_path, 'r') as z:
                file_name = z.namelist()[0]
                with z.open(file_name) as f:
                    for line in f:
                        try:
                            line_str = line.decode('utf-8').strip()
                            if not line_str:
                                continue
                            packet = json.loads(line_str)

                            if not book.update(packet):
                                continue

                            ob = book.get_orderbook()
                            if len(ob['bids']) >= 10 and len(ob['asks']) >= 10:
                                orderbooks.append(ob)

                        except Exception:
                            continue

            logger.info(f"ZIP chargé: {len(orderbooks)} orderbooks extraits")
        except Exception as e:
            logger.error(f"Erreur lors du chargement ZIP: {str(e)}")
            raise

        return orderbooks

    def _load_data_file(self, data_path: str) -> List[Dict]:
        """Charge un fichier .data non compressé."""
        logger.info(f"Chargement du fichier .data: {data_path}")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Le fichier n'existe pas: {data_path}")

        orderbooks = []
        book = LOBBook()

        with open(data_path, 'r') as f:
            for line in f:
                try:
                    line_str = line.strip()
                    if not line_str:
                        continue
                    packet = json.loads(line_str)

                    if not book.update(packet):
                        continue

                    ob = book.get_orderbook()
                    if len(ob['bids']) >= 10 and len(ob['asks']) >= 10:
                        orderbooks.append(ob)

                except Exception:
                    continue

        logger.info(f"Fichier .data chargé: {len(orderbooks)} orderbooks extraits")
        return orderbooks

    def _load_json_data(self, path: str) -> List[Dict]:
        """Charge les données depuis un fichier JSON lines (legacy)."""
        logger.info(f"Chargement du fichier JSON: {path}")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Le fichier n'existe pas: {path}")

        try:
            with open(path, 'r') as f:
                data = [json.loads(line) for line in f]
            logger.info(f"Données JSON chargées: {len(data)} lignes")
            return sorted(data, key=lambda x: x['local_timestamp'])
        except Exception as e:
            logger.error(f"Erreur lors du chargement JSON: {str(e)}")
            raise

    def get_next_orderbook(self) -> Optional[Dict]:
        """Retourne le prochain orderbook du LOB."""
        if self.current_idx >= len(self.data):
            logger.info("Fin des données atteinte")
            return None

        orderbook = self.data[self.current_idx]
        self.current_idx += 1

        if self.current_idx % 10000 == 0:
            logger.info(f"Processing orderbook {self.current_idx}/{len(self.data)}")

        return orderbook

    def reset(self, random_offset: bool = False, episode_length: int = 1001):
        """Reset le loader pour un nouvel épisode.

        En mode lazy (répertoire), charge un fichier aléatoire à chaque reset.
        Ensuite, place l'index à une position aléatoire dans le fichier.

        Args:
            random_offset: Si True, démarre à une position aléatoire dans les données.
            episode_length: Nombre d'orderbooks nécessaires par épisode (steps + 1).
        """
        if self._lazy_mode:
            # Charger un fichier aléatoire (= un jour aléatoire)
            new_file = np.random.choice(self._file_paths)
            if new_file != self._current_file:
                self.data = self._load_file_cached(new_file)
                self._current_file = new_file
                logger.info(f"Fichier chargé: {os.path.basename(new_file)} ({len(self.data)} orderbooks)")

        # Position aléatoire dans le fichier courant
        if random_offset and len(self.data) > episode_length:
            max_start = len(self.data) - episode_length
            self.current_idx = np.random.randint(0, max_start)
            logger.info(f"Data loader reset à offset {self.current_idx}/{len(self.data)}")
        else:
            self.current_idx = 0
            logger.info("Reset du data loader")

    @property
    def n_files(self) -> int:
        """Nombre de fichiers disponibles (1 si mode fichier unique)."""
        return len(self._file_paths) if self._lazy_mode else 1
