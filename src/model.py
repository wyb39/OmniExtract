"""
Model Configuration Management Module

Provides a unified interface to configure and manage different types of language models, including OpenAI, Azure, vLLM, Ollama, Qwen, DeepSeek, and others.
"""

from dsp import Any
from pydantic import BaseModel, PrivateAttr
from typing import Literal
import baseUtil
import os
from loguru import logger
import dspy
from secure_api_key import SecureAPIKeyManager

ModelProvider = Literal[
    "openai",
    "azure",
    "vllm",
    "ollama",
    "qwen",
    "deepseek",
    "gemini",
    "anthropic",
    "sglang",
    "openrouter",
    "custom",
]


class ModelSettings(BaseModel):
    model_name: str = ""
    model_type: ModelProvider = "openai"
    api_base: str = ""
    api_key: str | None = None
    model_usage: Literal["main", "visual", "prompt_generation", "judge", "coder"] = (
        "main"
    )
    temperature: float = 0.0
    max_tokens: int | None = 8000
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    setting_status: bool = False
    _lm: Any | None = PrivateAttr(default=None)

    def save_model_settings(self):
        try:
            save_message = ""
            encryption_key_env = os.getenv("OMNI_EXTRACT_ENCRYPTION_KEY")
            if encryption_key_env:
                success = SecureAPIKeyManager.store_api_key_with_env_key(self.api_key, self.model_usage)
                if success:
                    save_message = "Model settings stored successfully."
                else:
                    logger.error("Failed to store API key with encryption key")
                    save_message = "Failed to store API key with encryption key. Params except API Key have been saved. Maybe use the environment variable like OPENAI_API_KEY could work."
            else:
                try:
                    SecureAPIKeyManager.store_api_key(self.api_key, self.model_usage)
                    save_message = f"Model settings stored successfully."
                except Exception as e:
                    logger.error(f"Error storing API key: {e}")
                    save_message = "Failed to store API key. Params except API Key have been saved. Maybe use the environment variable like OPENAI_API_KEY could work."
            
            settings_without_api_key = self.model_copy()
            settings_without_api_key.api_key = None
            
            model_settings_file_path = os.path.join(
                baseUtil.get_root_path(), "settings"
            )
            with open(
                os.path.join(
                    model_settings_file_path, f"model_settings_{self.model_usage}.json"
                ),
                "w",
            ) as f:
                f.write(settings_without_api_key.model_dump_json())
            self._lm = None
            
            # Reload the corresponding global model instance
            usage_to_var = {
                "main": "model_setting_instance",
                "visual": "model_setting_instance_image",
                "prompt_generation": "model_setting_instance_prompt",
                "judge": "model_setting_instance_judge",
                "coder": "model_setting_instance_coder",
            }
            
            if self.model_usage in usage_to_var:
                var_name = usage_to_var[self.model_usage]
                if var_name in globals():
                    global_instance = globals()[var_name]
                    if self is not global_instance:
                        new_settings = ModelSettings.load_model_settings(self.model_usage)
                        for field in new_settings.model_fields:
                            setattr(global_instance, field, getattr(new_settings, field))
                        global_instance._lm = None
                        logger.info(f"Reloaded global model instance: {var_name}")
            
            return save_message
        except (IOError, OSError) as e:
            logger.error(f"Error saving model settings: {e}")
            raise

    @staticmethod
    def load_model_settings(
        model_usage: Literal["main", "visual", "prompt_generation", "judge", "coder"],
    ) -> "ModelSettings":
        """load model settings from json file"""
        settings_file = os.path.join(
            baseUtil.get_root_path(), "settings", f"model_settings_{model_usage}.json"
        )

        if os.path.exists(settings_file):
            try:
                with open(settings_file, "r", encoding="utf-8") as f:
                    settings = ModelSettings.model_validate_json(f.read())
                
                # Get API key from secure storage
                try:
                    if os.getenv("OMNI_EXTRACT_ENCRYPTION_KEY"):
                        settings.api_key = SecureAPIKeyManager.get_api_key_from_env(key_type=model_usage)
                    else:
                        settings.api_key = SecureAPIKeyManager.get_api_key(model_usage)
                except Exception as e:
                    logger.warning(f"Failed to load API key from secure storage: {e}")
                return settings
            except (IOError, OSError, ValueError) as e:
                logger.warning(
                    f"failed to load model settings: {e}, using default settings"
                )
                return ModelSettings(model_usage=model_usage)

        return ModelSettings(model_usage=model_usage)

    @staticmethod
    def load_model_settings_without_api_key(
        model_usage: Literal["main", "visual", "prompt_generation", "judge", "coder"],
    ) -> "ModelSettings":
        """load model settings from json file without loading API key"""
        settings_file = os.path.join(
            baseUtil.get_root_path(), "settings", f"model_settings_{model_usage}.json"
        )

        if os.path.exists(settings_file):
            try:
                with open(settings_file, "r", encoding="utf-8") as f:
                    settings = ModelSettings.model_validate_json(f.read())
                # Ensure api_key is None for security
                settings.api_key = None
                return settings
            except (IOError, OSError, ValueError) as e:
                logger.warning(
                    f"failed to load model settings: {e}, using default settings"
                )
                return ModelSettings(model_usage=model_usage)

        return ModelSettings(model_usage=model_usage)

    def configure_model(self) -> dspy.LM:
        """create DSPy LM from settings"""
        if self._lm is not None:
            return self._lm
        # Model configuration mapping
        model_configs = {
            "ollama": {
                "model_name_prefix": "",
                "default_api_base": "http://localhost:11434",
                "default_api_key": "",
                "custom_llm_provider": "ollama",
            },
            "vllm": {
                "model_name_prefix": "hosted_vllm/",
                "default_api_base": "http://localhost:8000/v1",
                "default_api_key": "EMPTY",
            },
            "qwen": {
                "model_name_prefix": "openai/",
                "default_api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "default_api_key": "custom",
            },
            "deepseek": {
                "model_name_prefix": "",
                "default_api_base": "https://api.deepseek.com",
                "default_api_key": "custom",
                "custom_llm_provider": "deepseek",
            },
            "openai": {
                "model_name_prefix": "openai/",
                "default_api_base": "https://api.openai.com/v1",
                "default_api_key": "custom",
                "custom_llm_provider": "openai",
            },
            "gemini": {
                "model_name_prefix": "",
                "default_api_base": "https://generativelanguage.googleapis.com/v1beta",
                "default_api_key": "custom",
                "custom_llm_provider": "gemini",
            },
            "anthropic": {
                "model_name_prefix": "",
                "default_api_base": "https://api.anthropic.com",
                "default_api_key": "custom",
                "custom_llm_provider": "anthropic",
            },
            "sglang": {
                "model_name_prefix": "",
                "default_api_base": "http://localhost:30000/v1",
                "default_api_key": "EMPTY",
                "custom_llm_provider": "openrouter",
            },
            "openrouter": {
                "model_name_prefix": "",
                "default_api_base": "https://openrouter.ai/api/v1",
                "default_api_key": "custom",
                "custom_llm_provider": "openrouter",
            },
            "custom": {
                "model_name_prefix": "openai/",
                "default_api_base": "",
                "default_api_key": "custom",
                "custom_llm_provider": "openai",
            },
        }

        # Parameter validation
        if not self.model_name or not self.model_type:
            raise ValueError("Model and Model Type cannot be empty")

        if self.model_type not in model_configs:
            raise ValueError(
                f"The MODEL TYPE IS NOT SUPPORTED: {self.model_type}. "
                f"Try to use an openai-like api with a model type 'custom'.The model_name, api_base and api_key are required for custom models."
            )

        config = model_configs[self.model_type]

        if not self.api_base:
            if self.model_type == "custom":
                raise ValueError("For 'custom' model type, 'api_base' must be provided.")
            self.api_base = config["default_api_base"]

        if not self.api_key:
            default_key = config["default_api_key"]
            if default_key == "custom":
                env_key = f"{self.model_type.upper()}_API_KEY"
                self.api_key = os.getenv(env_key)
                if not self.api_key:
                    raise ValueError(f"Please set the environment parameter: {env_key}")
            else:
                self.api_key = default_key

        # create dspy lm params
        params : dict[str, Any] = {
            "model": f"{config['model_name_prefix']}{self.model_name}",
            "model_type": "chat",
            "api_base": self.api_base,
            "api_key": self.api_key,
        }

        # add custom_llm_provider
        if "custom_llm_provider" in config:
            params["custom_llm_provider"] = config["custom_llm_provider"]

        # add optional params
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens

        if self.temperature > 0:
            params["temperature"] = self.temperature

        # sglang params
        if self.model_type == "sglang":
            extra_body = {}
            if self.top_p is not None:
                extra_body["top_p"] = self.top_p
            if self.top_k is not None:
                extra_body["top_k"] = self.top_k
            if self.min_p is not None:
                extra_body["min_p"] = self.min_p
            if extra_body:
                params["extra_body"] = extra_body
        else:
            if self.top_p is not None:
                params["top_p"] = self.top_p
            if self.top_k is not None:
                params["top_k"] = self.top_k
            if self.min_p is not None:
                params["min_p"] = self.min_p

        try:
            llm = dspy.LM(**params)
            logger.info(
                f"Success to create model configure: {self.model_name} ({self.model_type})"
            )
            self._lm = llm
            return llm
        except Exception as e:
            logger.error(f"Failed to create model configure: {e}")
            raise

    def test_call(self, prompt: str) -> dict[str, Any]:
        try:
            llm = self.configure_model()
            outputs = llm(prompt=prompt)
            success = bool(outputs) and any(isinstance(o, str) and bool(o.strip()) for o in outputs)
            return {"success": success, "outputs": outputs}
        except Exception as e:
            logger.error(f"Model test_call failed: {e}")
            return {"success": False, "error": str(e), "outputs": []}


MODEL_USAGE_TYPES = ["main", "visual", "prompt_generation", "judge", "coder", "openrouter"]


def get_model_settings(
    usage_type: Literal["main", "visual", "prompt_generation", "judge", "coder"],
    include_api_key: bool = False
) -> ModelSettings:
    """get model settings by usage type"""
    if usage_type not in MODEL_USAGE_TYPES:
        raise ValueError(
            f"Invalid model usage type: {usage_type}. Optional values: {MODEL_USAGE_TYPES}"
        )
    if include_api_key:
        return ModelSettings.load_model_settings(usage_type)
    else:
        return ModelSettings.load_model_settings_without_api_key(usage_type)


model_setting_instance = get_model_settings("main", include_api_key=True)
model_setting_instance_image = get_model_settings("visual", include_api_key=True)
model_setting_instance_prompt = get_model_settings("prompt_generation", include_api_key=True)
model_setting_instance_judge = get_model_settings("judge", include_api_key=True)
model_setting_instance_coder = get_model_settings("coder", include_api_key=True)
