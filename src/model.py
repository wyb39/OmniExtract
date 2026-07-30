"""
Model Configuration Management Module

Provides a unified interface to configure and manage different types of language models, including OpenAI, vLLM, Ollama, Qwen, DeepSeek, and others.
"""

from pydantic import BaseModel, Field, PrivateAttr, field_validator, model_validator
from threading import RLock
from typing import Any, ClassVar, Literal
import baseUtil
import os
from loguru import logger
import dspy
from secure_api_key import SecureAPIKeyManager
from token_usage import record_provider_usage

ModelProvider = Literal[
    "openai",
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

ReasoningEffort = Literal["low", "medium", "high"]

_SEEN_DSPY_RESPONSES: set[int] = set()
_SEEN_DSPY_RESPONSES_LOCK = RLock()


def _is_uncached_dspy_response(response: Any, cache_enabled: bool) -> bool:
    """Return False when DSPy returned the same in-memory cached response."""

    if not cache_enabled or response is None:
        return True
    response_id = id(response)
    with _SEEN_DSPY_RESPONSES_LOCK:
        if response_id in _SEEN_DSPY_RESPONSES:
            return False
        _SEEN_DSPY_RESPONSES.add(response_id)
        return True


class TokenTrackingLM(dspy.LM):
    """DSPy LM that forwards each provider usage block to the active report."""

    def __call__(self, prompt=None, messages=None, **kwargs):
        cache_enabled = bool(kwargs.get("cache", getattr(self, "cache", True)))
        outputs = super().__call__(prompt=prompt, messages=messages, **kwargs)
        # DSPy stores the exact outputs list in its history entry.  Identity
        # matching remains correct when several threads share one LM.
        entry = next(
            (
                item
                for item in reversed(self.history)
                if item.get("outputs") is outputs
            ),
            None,
        )
        if entry is not None and _is_uncached_dspy_response(
            entry.get("response"),
            cache_enabled,
        ):
            record_provider_usage(entry.get("usage", {}))
        return outputs


class ModelSettings(BaseModel):
    TOP_K_PROVIDERS: ClassVar[set[str]] = {
        "vllm",
        "ollama",
        "qwen",
        "gemini",
        "anthropic",
        "sglang",
        "openrouter",
        "custom",
    }
    MIN_P_PROVIDERS: ClassVar[set[str]] = {
        "vllm",
        "ollama",
        "sglang",
        "openrouter",
        "custom",
    }

    model_name: str = ""
    model_type: ModelProvider = "openai"
    api_base: str = ""
    api_key: str | None = None
    model_usage: Literal["main", "visual", "prompt_generation", "judge", "coder"] = (
        "main"
    )
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=8000, gt=0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    top_k: int | None = Field(default=None, gt=0)
    min_p: float | None = Field(default=None, ge=0.0, le=1.0)
    cache_for_optimization: bool = True
    cache_for_other: bool = False
    thinking_enabled: bool = False
    reasoning_effort: ReasoningEffort = "medium"
    thinking_budget_tokens: int | None = Field(default=None, ge=1)
    setting_status: bool = False
    _lm: Any | None = PrivateAttr(default=None)
    _optimization_lm: Any | None = PrivateAttr(default=None)

    @field_validator("model_name", "api_base", mode="before")
    @classmethod
    def strip_text_settings(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value

    def _uses_openai_reasoning_defaults(self) -> bool:
        bare_name = self.model_name.rsplit("/", 1)[-1].lower()
        return self.model_type == "openai" and bare_name.startswith(
            ("o1", "o3", "o4", "gpt-5")
        )

    @model_validator(mode="after")
    def validate_provider_parameters(self) -> "ModelSettings":
        if self.top_k is not None and self.model_type not in self.TOP_K_PROVIDERS:
            raise ValueError(
                f"top_k is not supported for provider '{self.model_type}'"
            )
        if self.min_p is not None and self.model_type not in self.MIN_P_PROVIDERS:
            raise ValueError(
                f"min_p is not supported for provider '{self.model_type}'"
            )
        if self.thinking_enabled and self.model_type == "anthropic":
            budget = self.thinking_budget_tokens or 1024
            if budget < 1024:
                raise ValueError(
                    "Anthropic thinking_budget_tokens must be at least 1024"
                )
            if self.max_tokens is None or self.max_tokens <= budget:
                raise ValueError(
                    "Anthropic max_tokens must be greater than "
                    "thinking_budget_tokens"
                )
        if (
            self._uses_openai_reasoning_defaults()
            and self.model_name.rsplit("/", 1)[-1].lower().startswith("o1-")
            and (self.max_tokens is None or self.max_tokens < 5000)
        ):
            raise ValueError(
                "DSPy 2.5.41 requires max_tokens >= 5000 for OpenAI o1 models"
            )
        return self

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
            self._optimization_lm = None
            
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
                        global_instance._optimization_lm = None
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

    @staticmethod
    def _normalise_model_name(
        model_name: str,
        model_type: ModelProvider,
        target_prefix: str,
    ) -> str:
        removable_prefixes = {
            "openai": ("openai/",),
            "vllm": ("hosted_vllm/", "vllm/"),
            "ollama": ("ollama/",),
            "qwen": ("qwen/",),
            "deepseek": ("deepseek/",),
            "gemini": ("gemini/",),
            "anthropic": ("anthropic/",),
            "openrouter": ("openrouter/",),
            "custom": ("custom/",),
        }.get(model_type, ())

        normalised = model_name.strip()
        for prefix in removable_prefixes:
            if normalised.startswith(prefix):
                normalised = normalised[len(prefix):]
                break
        if target_prefix and not normalised.startswith(target_prefix):
            normalised = f"{target_prefix}{normalised}"
        return normalised

    def _apply_thinking_parameters(
        self,
        params: dict[str, Any],
        extra_body: dict[str, Any],
    ) -> None:
        if not self.thinking_enabled:
            if self.model_type == "qwen":
                extra_body["enable_thinking"] = False
            return

        budget = self.thinking_budget_tokens
        if self.model_type == "openai":
            extra_body["reasoning_effort"] = self.reasoning_effort
        elif self.model_type == "openrouter":
            reasoning: dict[str, Any] = {"enabled": True}
            if budget is not None:
                reasoning["max_tokens"] = budget
            else:
                reasoning["effort"] = self.reasoning_effort
            params["reasoning"] = reasoning
        elif self.model_type == "anthropic":
            params["thinking"] = {
                "type": "enabled",
                "budget_tokens": budget or 1024,
            }
        elif self.model_type == "qwen":
            extra_body["enable_thinking"] = True
            if budget is not None:
                extra_body["thinking_budget"] = budget
        elif self.model_type in {"vllm", "sglang"}:
            extra_body["chat_template_kwargs"] = {
                "enable_thinking": True,
            }
        elif self.model_type == "custom":
            extra_body["reasoning_effort"] = self.reasoning_effort
            if budget is not None:
                extra_body["thinking_budget"] = budget
        else:
            logger.debug(
                "Provider {} uses model-native thinking behavior; no explicit "
                "thinking parameter is emitted by the current LiteLLM adapter.",
                self.model_type,
            )

    def configure_model(self, *, for_optimization: bool = False) -> dspy.LM:
        """create DSPy LM from settings"""
        cached_lm = self._optimization_lm if for_optimization else self._lm
        if cached_lm is not None:
            return cached_lm
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
                "default_api_base": "",
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
                "default_api_key": "EMPTY",
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

        api_base = self.api_base
        if not api_base:
            if self.model_type == "custom":
                raise ValueError("For 'custom' model type, 'api_base' must be provided.")
            api_base = config["default_api_base"]

        api_key = self.api_key
        if not api_key:
            default_key = config["default_api_key"]
            if default_key == "custom":
                env_keys = {
                    "qwen": ("DASHSCOPE_API_KEY", "QWEN_API_KEY"),
                }.get(
                    self.model_type,
                    (f"{self.model_type.upper()}_API_KEY",),
                )
                api_key = next(
                    (os.getenv(key) for key in env_keys if os.getenv(key)),
                    None,
                )
                if not api_key:
                    raise ValueError(
                        "Please set one of the environment parameters: "
                        + ", ".join(env_keys)
                    )
            else:
                api_key = default_key

        model_name = self._normalise_model_name(
            self.model_name,
            self.model_type,
            config["model_name_prefix"],
        )

        # create dspy lm params
        params : dict[str, Any] = {
            "model": model_name,
            "model_type": "chat",
            "api_key": api_key,
            "cache": (
                self.cache_for_optimization
                if for_optimization
                else self.cache_for_other
            ),
        }
        if api_base:
            params["api_base"] = api_base

        # add custom_llm_provider
        if "custom_llm_provider" in config:
            params["custom_llm_provider"] = config["custom_llm_provider"]

        # add optional params
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens

        effective_temperature = self.temperature
        if (
            self.thinking_enabled
            and self.model_type in {"openai", "anthropic"}
        ) or self._uses_openai_reasoning_defaults():
            effective_temperature = 1.0
        params["temperature"] = effective_temperature

        extra_body: dict[str, Any] = {}
        if self.model_type == "sglang":
            if self.top_p is not None:
                extra_body["top_p"] = self.top_p
            if self.top_k is not None:
                extra_body["top_k"] = self.top_k
            if self.min_p is not None:
                extra_body["min_p"] = self.min_p
        else:
            if self.top_p is not None:
                params["top_p"] = self.top_p
            if self.top_k is not None:
                if self.model_type in {"ollama", "gemini", "anthropic"}:
                    params["top_k"] = self.top_k
                else:
                    extra_body["top_k"] = self.top_k
            if self.min_p is not None:
                if self.model_type == "ollama":
                    params["min_p"] = self.min_p
                else:
                    extra_body["min_p"] = self.min_p

        self._apply_thinking_parameters(params, extra_body)
        if extra_body:
            params["extra_body"] = extra_body

        try:
            llm = TokenTrackingLM(**params)
            logger.info(
                f"Success to create model configure: {self.model_name} ({self.model_type})"
            )
            if for_optimization:
                self._optimization_lm = llm
            else:
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


MODEL_USAGE_TYPES = ["main", "visual", "prompt_generation", "judge", "coder"]


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
