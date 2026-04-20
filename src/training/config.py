import yaml
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """
    تحميل إعدادات yaml من ملف.
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def merge_configs(data_config: Dict, model_config: Dict, train_config: Dict) -> Dict:
    """
    دمج الإعدادات الثلاثة في قاموس واحد.
    """
    config = {}
    config.update(data_config.get('data', {}))
    config.update(model_config.get('model', {}))
    config.update(train_config.get('training', {}))
    return config