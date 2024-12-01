import logging
import os
from datetime import datetime
from typing import Dict, Any
import json
from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(self, config):
        self.config = config
        self.log_dir = config.LOG_DIR
        
        # Set up file logger
        self.logger = self._setup_file_logger()
        
        # Set up tensorboard writer
        self.writer = SummaryWriter(log_dir=os.path.join(self.log_dir, 'tensorboard'))
        
    def _setup_file_logger(self) -> logging.Logger:
        """Set up file logger with formatting"""
        logger = logging.getLogger('P-Companion')
        logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler
        fh = logging.FileHandler(
            os.path.join(self.log_dir, f'training_{datetime.now():%Y%m%d_%H%M%S}.log')
        )
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        return logger
    
    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = '') -> None:
        """Log metrics to both file and tensorboard"""
        # Log to file
        metric_str = ' '.join(f'{k}: {v:.4f}' for k, v in metrics.items())
        self.logger.info(f'{prefix} Step {step}: {metric_str}')
        
        # Log to tensorboard
        for name, value in metrics.items():
            self.writer.add_scalar(f'{prefix}/{name}', value, step)
    
    def log_config(self, config: Dict[str, Any]) -> None:
        """Log configuration parameters"""
        self.logger.info('Configuration:')
        config_str = json.dumps(config, indent=2)
        self.logger.info(f'\n{config_str}')
        
    def close(self) -> None:
        """Close tensorboard writer"""
        self.writer.close()