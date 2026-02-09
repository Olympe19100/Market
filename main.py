import os
import asyncio
import logging
import yaml
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict

from core.config import config
from monitoring.logger import setup_logger
from market_maker import MarketMaker

async def run_market_maker(config_path: str, debug: bool = False):
    """
    Lance le market maker avec la configuration spécifiée.
    """
    try:
        # Charger la configuration
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Créer le dossier de logs
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        # Configurer le logger
        log_file = log_dir / f"market_maker_{datetime.now():%Y%m%d_%H%M%S}.log"
        logger = setup_logger(
            log_level="DEBUG" if debug else "INFO",
            log_file=log_file,
            console_output=True
        )
        
        logger.info("Starting Market Maker with configuration from %s", config_path)
        
        # Ajouter les clés API de l'environnement
        config_dict['binance']['api_key'] = os.environ.get('BINANCE_API_KEY')
        config_dict['binance']['api_secret'] = os.environ.get('BINANCE_API_SECRET')
        
        if not config_dict['binance']['api_key'] or not config_dict['binance']['api_secret']:
            raise ValueError("BINANCE_API_KEY and BINANCE_API_SECRET must be set in environment")
        
        # Créer la configuration
        config = config.from_dict(config_dict)
        
        # Créer et démarrer le market maker
        market_maker = MarketMaker(config)
        
        # Gestionnaire de signaux pour arrêt propre
        loop = asyncio.get_event_loop()
        
        async def shutdown(signal=None):
            """Arrête proprement le market maker."""
            if signal:
                logger.info(f"Received exit signal {signal.name}")
            
            tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
            logger.info(f"Cancelling {len(tasks)} outstanding tasks")
            
            # Arrêter le market maker
            await market_maker.stop()
            
            # Annuler les tâches restantes
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            
            loop.stop()
            logger.info("Shutdown complete")
        
        # Enregistrer les gestionnaires de signaux
        for sig in ('SIGINT', 'SIGTERM'):
            loop.add_signal_handler(
                getattr(signal, sig),
                lambda s=sig: asyncio.create_task(shutdown(s))
            )
        
        # Démarrer le market maker
        try:
            await market_maker.start()
            
            # Boucle de monitoring
            while True:
                # Afficher les métriques périodiquement
                metrics = market_maker.get_metrics()
                position = market_maker.get_position()
                
                logger.info("\nCurrent Metrics:")
                logger.info("-" * 50)
                logger.info(f"Total PnL: {metrics['total_pnl']:.8f} USDC")
                logger.info(f"Position: {position.quantity:.8f} @ {position.entry_price:.8f}")
                logger.info(f"Unrealized PnL: {position.unrealized_pnl:.8f} USDC")
                logger.info(f"Win Rate: {metrics['win_rate']*100:.2f}%")
                logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
                logger.info(f"Max Drawdown: {metrics['max_drawdown']:.8f} USDC")
                logger.info("-" * 50)
                
                await asyncio.sleep(60)  # Mise à jour toutes les minutes
                
        except Exception as e:
            logger.error(f"Error in market maker: {e}")
            await shutdown()
        finally:
            await shutdown()
            
    except Exception as e:
        logger.error(f"Failed to start market maker: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Start the Market Maker')
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    args = parser.parse_args()
    
    try:
        asyncio.run(run_market_maker(args.config, args.debug))
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        raise