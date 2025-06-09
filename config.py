"""
Sistema de configuração para o GTA (Git Terminal Assistant)
Permite configurar parâmetros como modelos LLM para cada agente
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional

# Caminho padrão para o arquivo de configuração
CONFIG_PATH = os.environ.get('GTA_CONFIG_PATH', Path.home() / '.gta_config.json')

# Configurações padrão
DEFAULT_CONFIG = {
    "models": {
        "FileAgent": "llama3.1:8b",
        "TestAgent": "llama3.1:8b", 
        "GitAgent": "llama3.1:8b",
        "ChatAgent": "llama3.1:8b", 
        "CodeAgent": "llama3.1:8b"
    },
    "default_model": "llama3.1:8b"
}

# Cache para a configuração carregada
_config_cache: Optional[Dict[str, Any]] = None

def load_config() -> Dict[str, Any]:
    """Carrega a configuração do arquivo ou usa os padrões se não existir"""
    global _config_cache
    
    # Retorna o cache se já estiver carregado
    if _config_cache is not None:
        return _config_cache
    
    config_path = Path(CONFIG_PATH)
    
    # Se o arquivo existir, carrega a configuração
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)
                
            # Combina com os padrões para garantir que todos os campos existam
            config = DEFAULT_CONFIG.copy()
            config.update(loaded_config)
            
            # Garante que todos os agentes tenham uma configuração de modelo
            for agent in DEFAULT_CONFIG['models']:
                if agent not in config['models']:
                    config['models'][agent] = config['default_model']
            
            _config_cache = config
            return config
        except Exception as e:
            print(f"Erro ao carregar configuração: {e}")
            print("Usando configurações padrão.")
            _config_cache = DEFAULT_CONFIG
            return DEFAULT_CONFIG
    else:
        # Cria o arquivo de configuração com valores padrão
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            save_config(DEFAULT_CONFIG)
            print(f"Arquivo de configuração criado em {config_path}")
            _config_cache = DEFAULT_CONFIG
            return DEFAULT_CONFIG
        except Exception as e:
            print(f"Erro ao criar arquivo de configuração: {e}")
            _config_cache = DEFAULT_CONFIG
            return DEFAULT_CONFIG

def save_config(config: Dict[str, Any]) -> bool:
    """Salva a configuração em arquivo JSON"""
    global _config_cache
    
    config_path = Path(CONFIG_PATH)
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4)
        
        # Atualiza o cache
        _config_cache = config
        return True
    except Exception as e:
        print(f"Erro ao salvar configuração: {e}")
        return False

def get_model_for_agent(agent_name: str) -> str:
    """Retorna o modelo configurado para um agente específico"""
    config = load_config()
    
    # Verifica se o agente tem um modelo configurado
    if agent_name in config['models']:
        return config['models'][agent_name]
    
    # Caso contrário, retorna o modelo padrão
    return config['default_model']

def set_model_for_agent(agent_name: str, model_name: str) -> bool:
    """Configura um modelo específico para um agente"""
    config = load_config()
    
    # Atualiza a configuração
    if 'models' not in config:
        config['models'] = {}
    
    config['models'][agent_name] = model_name
    
    # Salva a configuração
    return save_config(config)

def set_default_model(model_name: str) -> bool:
    """Configura o modelo padrão"""
    config = load_config()
    config['default_model'] = model_name
    return save_config(config)
