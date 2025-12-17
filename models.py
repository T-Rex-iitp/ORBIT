"""
Model Configuration Module
쉽게 모델을 추가/관리할 수 있는 모듈
"""
import json
import os
from pathlib import Path

# 설정 파일 경로
CONFIG_FILE = Path(__file__).parent / "models_config.json"

# 기본 설정
DEFAULT_CONFIG = {
    "ollama_url": "http://localhost:11434",
    "custom_models": [
        # 예시: 직접 추가한 모델들
        # {
        #     "name": "my-local-model",
        #     "path": "/path/to/model.gguf",
        #     "description": "My custom model"
        # }
    ],
    "model_aliases": {
        # 모델 별칭 설정
        # "gpt": "llama3.2",
        # "claude": "deepseek-r1"
    },
    "default_model": None,
    "model_settings": {
        # 모델별 기본 설정
        # "llama3.2": {
        #     "temperature": 0.7,
        #     "top_p": 0.9,
        #     "context_length": 4096
        # }
    }
}


class ModelManager:
    def __init__(self):
        self.config = self._load_config()

    def _load_config(self):
        """설정 파일 로드"""
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    # 기본값과 병합
                    for key, value in DEFAULT_CONFIG.items():
                        if key not in config:
                            config[key] = value
                    return config
            except Exception as e:
                print(f"Config load error: {e}")
        return DEFAULT_CONFIG.copy()

    def save_config(self):
        """설정 파일 저장"""
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)

    def get_ollama_url(self):
        """Ollama URL 반환"""
        return self.config.get("ollama_url", "http://localhost:11434")

    def set_ollama_url(self, url):
        """Ollama URL 설정"""
        self.config["ollama_url"] = url
        self.save_config()

    def get_custom_models(self):
        """커스텀 모델 목록 반환"""
        return self.config.get("custom_models", [])

    def add_custom_model(self, name, path, description=""):
        """커스텀 모델 추가"""
        model = {
            "name": name,
            "path": path,
            "description": description
        }

        # 중복 체크
        for m in self.config["custom_models"]:
            if m["name"] == name:
                m.update(model)
                self.save_config()
                return {"success": True, "message": "Model updated"}

        self.config["custom_models"].append(model)
        self.save_config()
        return {"success": True, "message": "Model added"}

    def remove_custom_model(self, name):
        """커스텀 모델 제거"""
        self.config["custom_models"] = [
            m for m in self.config["custom_models"] if m["name"] != name
        ]
        self.save_config()
        return {"success": True}

    def get_model_alias(self, alias):
        """별칭으로 실제 모델명 반환"""
        return self.config.get("model_aliases", {}).get(alias, alias)

    def set_model_alias(self, alias, model_name):
        """모델 별칭 설정"""
        if "model_aliases" not in self.config:
            self.config["model_aliases"] = {}
        self.config["model_aliases"][alias] = model_name
        self.save_config()

    def get_default_model(self):
        """기본 모델 반환"""
        return self.config.get("default_model")

    def set_default_model(self, model_name):
        """기본 모델 설정"""
        self.config["default_model"] = model_name
        self.save_config()

    def get_model_settings(self, model_name):
        """모델별 설정 반환"""
        return self.config.get("model_settings", {}).get(model_name, {})

    def set_model_settings(self, model_name, settings):
        """모델별 설정 저장"""
        if "model_settings" not in self.config:
            self.config["model_settings"] = {}
        self.config["model_settings"][model_name] = settings
        self.save_config()

    def get_all_config(self):
        """전체 설정 반환"""
        return self.config

    def update_config(self, new_config):
        """전체 설정 업데이트"""
        self.config.update(new_config)
        self.save_config()


# 싱글톤 인스턴스
model_manager = ModelManager()


# === 편의 함수들 ===

def add_model(name, path, description=""):
    """
    새 모델 추가

    사용법:
        from models import add_model
        add_model("my-model", "/path/to/model.gguf", "My custom LLM")
    """
    return model_manager.add_custom_model(name, path, description)


def remove_model(name):
    """모델 제거"""
    return model_manager.remove_custom_model(name)


def list_custom_models():
    """커스텀 모델 목록"""
    return model_manager.get_custom_models()


def set_alias(alias, model_name):
    """
    모델 별칭 설정

    사용법:
        from models import set_alias
        set_alias("gpt", "llama3.2")  # 'gpt' 입력시 llama3.2 사용
    """
    model_manager.set_model_alias(alias, model_name)


def set_default(model_name):
    """기본 모델 설정"""
    model_manager.set_default_model(model_name)


# === 자주 쓰는 모델 경로 템플릿 ===

COMMON_MODEL_PATHS = {
    "huggingface": "/mnt/hdd/huggingface-models",
    "ollama": "/mnt/hdd/ollama/models",
    "llava": "/mnt/hdd/llava/hub",
}


def get_model_path(location, model_name):
    """
    일반적인 모델 경로 생성

    사용법:
        path = get_model_path("huggingface", "Qwen/Qwen2-7B")
    """
    base = COMMON_MODEL_PATHS.get(location, location)
    return os.path.join(base, model_name)
