import os
import sys
from abc import ABC, abstractmethod
from configparser import ConfigParser
from remedee.common.utils.keychain import get_env
import logging

CONFIG_DIR = os.getenv("XDG_CONFIG_HOME", os.path.expanduser("~/.config"))
CONFIG_PATH = os.path.join(CONFIG_DIR, "remedee/remedee_zsh.ini")


class BaseClient(ABC):
    """Base class for all clients"""

    api_type: str = None
    system_prompt = "You are a zsh shell expert, please help me complete the following command, you should only output the completed command, no need to include any other explanation. Do not put completed command in a code block."

    @abstractmethod
    def get_completion(self, full_command: str) -> str:
        pass


class OpenAIClient(BaseClient):
    """
    config keys:
        - api_type="openai"
        - api_key (required)
        - base_url (optional): defaults to "https://api.openai.com/v1".
        - organization (optional): defaults to None
        - model (optional): defaults to "gpt-4o-mini"
        - temperature (optional): defaults to 1.0.
    """

    api_type = "openai"
    default_model = os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4o-mini")

    def __init__(self, config: dict):
        try:
            from openai import OpenAI
        except ImportError:
            print(
                "OpenAI library is not installed. Please install it using 'pip install openai'"
            )
            sys.exit(1)

        self.config = config
        self.config["model"] = self.config.get("model", self.default_model)
        api_key = get_env("OPENAI_API_KEY", 'open-ai-api', 'asr')
        self.client = OpenAI(
            api_key=api_key,
            base_url=self.config.get("base_url", "https://api.openai.com/v1"),
            organization=self.config.get("organization"),
        )

    def get_completion(self, full_command: str) -> str:
        return self.request(self.system_prompt, full_command)
    
    def request(self, system_prompt: str, user_prompt: str, history=[]) -> str:
        messages = [{"role": "system", "content": system_prompt}] 
        messages +=  history
        messages += [{"role": "user", "content": user_prompt}]

        for message in messages:
            logging.debug(f"{message['role']}: {message['content']}")
            
        response = self.client.chat.completions.create(
            model=self.config["model"],
            messages=messages,
            temperature=float(self.config.get("temperature", 0.5)),
        )
        res = response.choices[0].message.content
        history.append({"role": "assistant", "content": res})
        return res


class GoogleGenAIClient(BaseClient):
    """
    config keys:
        - api_type="gemeni"
        - api_key (required)
        - model (optional): defaults to "gemini-1.5-pro-latest"
    """

    api_type = "gemeni"
    default_model = os.getenv("GOOGLE_GENAI_DEFAULT_MODEL", "gemini-1.5-pro-latest")

    def __init__(self, config: dict):
        try:
            import google.generativeai as genai
        except ImportError:
            print(
                "Google Generative AI library is not installed. Please install it using 'pip install google-generativeai'"
            )
            sys.exit(1)

        self.config = config
        genai.configure(api_key=self.config["api_key"])
        self.config["model"] = config.get("model", self.default_model)
        self.model = genai.GenerativeModel(self.config["model"])

    def get_completion(self, full_command: str) -> str:
        chat = self.model.start_chat(history=[])
        prompt = f"{self.system_prompt}\n\n{full_command}"
        response = chat.send_message(prompt)
        return response.text


class GroqClient(BaseClient):
    """
    config keys:
        - api_type="groq"
        - api_key (required)
        - model (optional): defaults to "llama-3.2-11b-text-preview"
        - temperature (optional): defaults to 1.0.
    """
    
    api_type = "groq"
    default_model = os.getenv("GROQ_DEFAULT_MODEL", "llama-3.2-11b-text-preview")
    
    def __init__(self, config: dict):
        try:
            from groq import Groq
        except ImportError:
            print(
                "Groq library is not installed. Please install it using 'pip install groq'"
            )
            sys.exit(1)

        self.config = config
        self.config["model"] = self.config.get("model", self.default_model)
        self.client = Groq(
            api_key=self.config["api_key"],
        )
    
    def get_completion(self, full_command: str) -> str:
        response = self.client.chat.completions.create(
            model=self.config["model"],
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": full_command},
            ],
            temperature=float(self.config.get("temperature", 1.0)),
        )
        return response.choices[0].message.content


class MistralClient(BaseClient):
    """
    config keys:
        - api_type="mistral"
        - api_key (required)
        - model (optional): defaults to "codestral-latest"
        - temperature (optional): defaults to 1.0.
    """
    
    api_type = "mistral"
    default_model = os.getenv("MISTRAL_DEFAULT_MODEL", "codestral-latest")
    
    def __init__(self, config: dict):
        try:
            from mistralai import Mistral
        except ImportError:
            print(
                "Mistral library is not installed. Please install it using 'pip install mistralai'"
            )
            sys.exit(1)
        
        self.config = config
        self.config["model"] = self.config.get("model", self.default_model)
        self.client = Mistral(
            api_key=self.config["api_key"],
        )
        
    def get_completion(self, full_command: str) -> str:
        response = self.client.chat.complete(
            model=self.config["model"],
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": full_command},
            ],
            temperature=float(self.config.get("temperature", 1.0)),
        )
        return response.choices[0].message.content


class ClientFactory:
    api_types = [OpenAIClient.api_type, GoogleGenAIClient.api_type, GroqClient.api_type, MistralClient.api_type]

    @classmethod
    def create(cls):
        config_parser = ConfigParser()
        config_parser.read(CONFIG_PATH)
        service = config_parser["service"]["service"]
        try:
            config = {k: v for k, v in config_parser[service].items()}
        except KeyError:
            raise KeyError(f"Config for service {service} is not defined")

        api_type = config["api_type"]
        match api_type:
            case OpenAIClient.api_type:
                return OpenAIClient(config)
            case GoogleGenAIClient.api_type:
                return GoogleGenAIClient(config)
            case GroqClient.api_type:
                return GroqClient(config)
            case MistralClient.api_type:
                return MistralClient(config)
            case _:
                raise KeyError(
                    f"Specified API type {api_type} is not one of the supported services {cls.api_types}"
                )
