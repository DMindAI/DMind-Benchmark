"""
Configuration manager module for loading API configurations from models.yml
"""

import os
import yaml
import logging
from typing import Dict, List, Optional, Any, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("config_manager.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ConfigManager")

class ModelConfigManager:
    """Model configuration manager for loading and providing API configurations"""
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern to ensure only one instance exists"""
        if cls._instance is None:
            cls._instance = super(ModelConfigManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize configuration manager"""
        if self._initialized:
            return

        self.config_path = os.path.abspath(os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "models.yml"
        ))
        logger.info(f"Calculated config file path: {self.config_path}")
        
        self.api_base = None
        self.evaluation_api_base = None
        self.api_keys = {}  # Store API keys {name: key}
        self.api_key_models = {}  # Store model names associated with API keys {name: model_name}
        self.models = {}
        self._load_config()
        self._initialized = True
        
    def _load_config(self):
        """Load configuration from config file"""
        try:
            logger.info(f"Loading configuration from {self.config_path}")
            if not os.path.exists(self.config_path):
                logger.error(f"Config file does not exist: {self.config_path}")
                
                # Try to find backup location
                backup_path = os.path.abspath(os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                    "models.yml"
                ))
                logger.info(f"Trying backup path: {backup_path}")
                
                if os.path.exists(backup_path):
                    logger.info(f"Found config file at backup location: {backup_path}")
                    self.config_path = backup_path
                else:
                    logger.error(f"Config file not found at backup location: {backup_path}")
                    return
                
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                
            # Load API base
            self.api_base = config.get('api_base')
            logger.info(f"Loaded API base: {self.api_base}")
            
            # Load evaluation API base (if exists)
            self.evaluation_api_base = config.get('evaluation_api_base')
            logger.info(f"Loaded evaluation API base: {self.evaluation_api_base}")
            
            # Load API keys
            api_keys = config.get('api_keys', [])
            for key_info in api_keys:
                key = key_info.get('key')
                name = key_info.get('name')
                model_name = key_info.get('model_name')  # Read model name
                
                if key and name:
                    self.api_keys[name] = key
                    # If model name is specified, save it
                    if model_name:
                        self.api_key_models[name] = model_name
                        logger.info(f"API key {name} associated with model: {model_name}")
                        
            logger.info(f"Loaded {len(self.api_keys)} API keys, {len(self.api_key_models)} of which specify a model name")
            
            # Load model configurations
            models = config.get('models', [])
            for model in models:
                name = model.get('name')
                if name:
                    self.models[name] = model
            logger.info(f"Loaded {len(self.models)} model configurations")
            
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
    
    def get_api_base(self) -> Optional[str]:
        """Get API base URL"""
        return self.api_base
    
    def get_evaluation_api_base(self) -> Optional[str]:
        """Get evaluation API base URL, returns regular API base if not set"""
        return self.evaluation_api_base or self.api_base
    
    def get_api_key(self, key_name: str) -> Optional[str]:
        """
        Get API key by name
        
        Args:
            key_name: API key name
            
        Returns:
            Optional[str]: API key, None if it doesn't exist
        """
        return self.api_keys.get(key_name)
    
    def get_api_key_with_model(self, key_name: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Get API key and its associated model name by key name
        
        Args:
            key_name: API key name
            
        Returns:
            Tuple[Optional[str], Optional[str]]: (API key, model name), None for fields that don't exist
        """
        api_key = self.api_keys.get(key_name)
        model_name = self.api_key_models.get(key_name)
        return api_key, model_name
    
    def get_model_config(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get model configuration
        
        Args:
            model_name: Model name
            
        Returns:
            Optional[Dict[str, Any]]: Model configuration, None if it doesn't exist
        """
        return self.models.get(model_name)
    
    def get_all_model_names(self) -> List[str]:
        """
        Get all model names
        
        Returns:
            List[str]: List of model names
        """
        return list(self.models.keys())
    
    def get_third_party_api_config(self, model_name: str = None) -> Dict[str, Any]:
        """
        Get third-party API configuration (for evaluation models)
        
        Note: If the specified model is not found, it will try to use default evaluation model configuration
        
        Args:
            model_name: Optional model name
            
        Returns:
            Dict[str, Any]: API configuration, containing api_base and api_key
        """
        # Try to use evaluation-specific API key and model
        api_key, key_model_name = self.get_api_key_with_model("claude_eval")
        
        # Default API configuration
        default_config = {
            "api_base": self.get_evaluation_api_base(),
            "api_key": api_key,
            "model": key_model_name or "claude-3-7-sonnet-20250219"
        }
        
        # If API key doesn't exist, fall back to backup value
        if not default_config["api_key"]:
            default_config["api_key"] = "sk-sjkpMQ7WsWk5jUShcqhK4RSe3GEooupy8jsy7xQkbg6eQaaX"
            
        # Prioritize evaluation models
        eval_models = ["claude_evaluation", "gpt4_evaluation"]
        
        # If model name is not specified, use default evaluation model
        if not model_name:
            # Try to use configured evaluation models
            for eval_model_name in eval_models:
                model_config = self.get_model_config(eval_model_name)
                if model_config:
                    return self._get_api_config_from_model(model_config, default_config)
            return default_config
            
        # Try to get configuration for the specified model
        model_config = self.get_model_config(model_name)
        if not model_config:
            logger.warning(f"Model configuration not found: {model_name}, trying to use default evaluation model")
            # Try to use configured evaluation models
            for eval_model_name in eval_models:
                model_config = self.get_model_config(eval_model_name)
                if model_config:
                    return self._get_api_config_from_model(model_config, default_config)
            return default_config
        
        return self._get_api_config_from_model(model_config, default_config)
    
    def _get_api_config_from_model(self, model_config: Dict[str, Any], default_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract API configuration from model configuration
        
        Args:
            model_config: Model configuration
            default_config: Default configuration (used when model configuration is missing certain values)
            
        Returns:
            Dict[str, Any]: API configuration
        """
        # Check if model has its own API base URL
        model_api_base = model_config.get('api_base')
        
        # Get API key name from model configuration
        api_key_name = model_config.get('api_key')
        if not api_key_name:
            logger.warning(f"No API key name in model configuration, using default configuration")
            return default_config
        
        # Get API key and associated model name
        api_key, key_model_name = self.get_api_key_with_model(api_key_name)
        if not api_key:
            logger.warning(f"API key not found: {api_key_name}, using default configuration")
            return default_config
        
        # Determine which model name to use: prioritize model name associated with API key, then use model field from model config
        model_name = key_model_name or model_config.get('model', default_config["model"])
        
        # Return configuration
        return {
            "api_base": model_api_base or self.get_evaluation_api_base() or default_config["api_base"],
            "api_key": api_key,
            "model": model_name
        }

# Create global instance
config_manager = ModelConfigManager() 