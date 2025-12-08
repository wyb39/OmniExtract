"""
secure_api_key.py
Cross-platform secure storage for user API keys

This module provides secure API key storage functionality based on system keychain and encryption technologies.
Implementation principles:
1. Use the secure storage service provided by the operating system (Windows Credential Manager/Linux Secret Service/macOS Keychain)
2. Randomly generate a 256-bit local key and store it in the system keychain
3. Encrypt the user API Key using AES encryption with the local key
4. Store the encrypted ciphertext in a regular JSON configuration file
"""
import json
import os
import base64
import getpass
import time
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import keyring
from loguru import logger

SERVICE_NAME = "OmniExtract"
KEY_NAME = "device_encryption_key"
CONFIG_DIR = Path.home() / ".config" / "omniextract"
CONFIG_FILE = CONFIG_DIR / "secure_config.json"

CONFIG_DIR.mkdir(parents=True, exist_ok=True)


class SecureAPIKeyManager:
    """Secure API Key Manager"""
    
    @staticmethod
    def _get_or_create_device_key() -> bytes:
        """
        Get or create a device key
        When called for the first time, generate a new key and store it in the system keychain
        Subsequent calls retrieve directly from the system keychain
        """
        try:
            encoded_key = keyring.get_password(SERVICE_NAME, KEY_NAME)
            if encoded_key:
                return encoded_key.encode()
            key = Fernet.generate_key()
            keyring.set_password(SERVICE_NAME, KEY_NAME, key.decode())
            return key
        except Exception as e:
            logger.warning(f"Unable to access system keychain: {e}")
            raise
   
    @classmethod
    def store_api_key(cls, api_key: str | None, key_type: str = "main") -> bool:
        """
        Securely store an API key of specified type
        Args:
            api_key: The API key to store
            key_type: Type of API key (main, prompt_generation, judge, coder)
        Returns:
            bool: Whether storage was successful
        """
        if key_type not in ["main", "prompt_generation", "judge", "coder"]:
            logger.error(f"Invalid key type '{key_type}'. Valid types are: main, prompt_generation, judge, coder")
            return False
        try:
            if api_key is None:
                encrypted_data = None
            else:
                device_key = cls._get_or_create_device_key()
                fernet = Fernet(device_key)
                encrypted_data = fernet.encrypt(api_key.encode())
            if CONFIG_FILE.exists():
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
            else:
                config_data = {"version": "1.1", "keys": {}}
            config_data["keys"][key_type] = {
                "ciphertext": encrypted_data.decode() if encrypted_data is not None else None,
                "last_updated": json.dumps({"__type__": "datetime", "isoformat": time.strftime("%Y-%m-%dT%H:%M:%SZ")})
            }
            if "version" not in config_data:
                config_data["version"] = "1.1"
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Failed to store API key: {e}")
            return False
    
    @classmethod
    def get_api_key(cls, key_type: str = "main") -> str | None:
        """
        Get the decrypted API key of specified type
        Returns:
            str | None: The decrypted API key or None if not set
        Raises:
            FileNotFoundError: Configuration file does not exist, please set API key first
            KeyError: Specified key type does not exist
            Exception: Decryption failed
        """
        if not CONFIG_FILE.exists():
            raise FileNotFoundError("API key configuration file does not exist, please set API key first")
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            if "keys" in config_data:
                if key_type not in config_data["keys"]:
                    raise KeyError(f"API key of type '{key_type}' does not exist. Available types: {list(config_data['keys'].keys())}")
                ciphertext = config_data["keys"][key_type]["ciphertext"]
            else:
                if key_type != "main":
                    raise KeyError(f"Only 'main' key type exists in old configuration format")
                ciphertext = config_data["ciphertext"]
            if ciphertext is None:
                return None
            device_key = cls._get_or_create_device_key()
            fernet = Fernet(device_key)
            decrypted_data = fernet.decrypt(ciphertext.encode())
            return decrypted_data.decode()
        except Exception as e:
            logger.error(f"Failed to get API key: {e}")
            raise
    
    @classmethod
    def delete_api_key(cls, key_type: str | None = None) -> bool:
        """
        Delete the stored API key(s)
        Args:
            key_type: Type of API key to delete. If None, deletes all keys and config file.
        Returns:
            bool: Whether deletion was successful
        """
        try:
            if key_type is None:
                if CONFIG_FILE.exists():
                    CONFIG_FILE.unlink()
                try:
                    keyring.delete_password(SERVICE_NAME, KEY_NAME)
                except:
                    pass
                return True
            else:
                if key_type not in ["main", "prompt_generation", "judge", "coder"]:
                    logger.error(f"Invalid key type '{key_type}'. Valid types are: main, prompt_generation, judge, coder")
                    return False
                if not CONFIG_FILE.exists():
                    logger.warning(f"Configuration file does not exist, nothing to delete for type '{key_type}'")
                    return True
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                if "keys" in config_data and key_type in config_data["keys"]:
                    del config_data["keys"][key_type]
                    if not config_data["keys"]:
                        CONFIG_FILE.unlink()
                    else:
                        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                            json.dump(config_data, f, indent=2, ensure_ascii=False)
                    return True
                else:
                    logger.warning(f"Key type '{key_type}' does not exist in configuration")
                    return True
        except Exception as e:
            logger.warning(f"Error occurred while deleting API key: {e}")
            return False
    
    @classmethod
    def store_api_key_with_env_key(cls, api_key: str | None, key_type: str = "main") -> bool:
        """
        Encrypt API key with environment variable key if exists, otherwise generate a random key and return that key
        Args:
            api_key: The API key to store
            key_type: Type of API key (main, prompt_generation, judge, coder)
        Returns:
            bool: Whether storage was successful
        """
        if key_type not in ["main", "prompt_generation", "judge", "coder"]:
            logger.error(f"Invalid key type '{key_type}'. Valid types are: main, prompt_generation, judge, coder")
            return False
        try:
            encryption_key_from_env = os.getenv("OMNI_EXTRACT_ENCRYPTION_KEY")
            if not encryption_key_from_env:
                logger.error("Environment variable OMNI_EXTRACT_ENCRYPTION_KEY not set")
                return False
            else:
                # Use PBKDF2HMAC to derive a key from the environment variable
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=b'omniextract_salt',  # A fixed salt for key derivation
                    iterations=100000,
                )
                encryption_key = base64.urlsafe_b64encode(kdf.derive(encryption_key_from_env.encode())).decode()
            
            if api_key is None:
                encrypted_data = None
            else:
                fernet = Fernet(encryption_key.encode())
                encrypted_data = fernet.encrypt(api_key.encode())
            if CONFIG_FILE.exists():
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
            else:
                config_data = {"version": "1.1", "keys": {}}
            config_data["keys"][key_type] = {
                "ciphertext": encrypted_data.decode() if encrypted_data is not None else None,
                "last_updated": json.dumps({"__type__": "datetime", "isoformat": time.strftime("%Y-%m-%dT%H:%M:%SZ")})
            }
            if "version" not in config_data:
                config_data["version"] = "1.1"
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Failed to store API key: {e}")
            return False


    
    @classmethod
    def get_api_key_from_env(cls, env_var_name: str = "OMNI_EXTRACT_ENCRYPTION_KEY", key_type: str = "main") -> str | None:
        """
        Get decryption key from environment variable and decrypt API key
        Args:
            env_var_name: Name of the environment variable storing the encryption key
            key_type: Type of API key to retrieve (main, prompt_generation, judge, coder)
        Returns:
            str: The decrypted API key
        Raises:
            ValueError: Environment variable not set or empty
            FileNotFoundError: Configuration file does not exist
            KeyError: Specified key type does not exist
            Exception: Decryption failed
        """
        encryption_key_from_env = os.getenv(env_var_name)
        if not encryption_key_from_env:
            logger.error(f"Environment variable {env_var_name} is not set or empty")
            raise ValueError(f"Environment variable {env_var_name} is not set or empty")
        
        # Use PBKDF2HMAC to derive a key from the environment variable
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'omniextract_salt',  # A fixed salt for key derivation
            iterations=100000,
        )
        encryption_key = base64.urlsafe_b64encode(kdf.derive(encryption_key_from_env.encode())).decode()

        if not CONFIG_FILE.exists():
            logger.error("API key configuration file does not exist")
            raise FileNotFoundError("API key configuration file does not exist")
        try:
            with open(CONFIG_FILE, 'r') as f:
                config_data = json.load(f)
            key_data = config_data["keys"].get(key_type)
            if key_data is None:
                logger.error(f"API key of type '{key_type}' not found in configuration file")
                raise KeyError(f"API key of type '{key_type}' not found in configuration file")
            
            ciphertext = key_data.get("ciphertext")
            if ciphertext is None:
                return None

            fernet = Fernet(encryption_key.encode())
            decrypted_data = fernet.decrypt(ciphertext.encode())
            return decrypted_data.decode()
        except Exception as e:
            logger.error(f"Failed to decrypt API key: {e}")
            raise Exception(f"Failed to decrypt API key: {e}")
    