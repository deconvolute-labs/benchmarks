from abc import ABC, abstractmethod

import openai

from dcv_benchmark.models.experiments_config import LLMConfig


class BaseLLM(ABC):
    """
    Abstract base class for Large Language Model providers.
    """

    @abstractmethod
    def generate(self, system_message: str, user_message: str) -> str | None:
        """
        Generates a text response from the LLM based on the provided messages.

        Args:
            system_message: The system instruction.
            user_message: The user query or prompt content.

        Returns:
            The string content of the model's response or None.
        """
        pass


class OpenAILLM(BaseLLM):
    """
    Concrete implementation using the OpenAI API (Chat Completion).
    Requires the env variable 'OPENAI_API_KEY' with the api key.
    """

    def __init__(self, config: LLMConfig):
        """
        Initializes the OpenAI client with the specified model and temperature.

        Args:
            config: Configuration object containing 'model' and 'temperature'.
        """
        self.client = openai.Client()
        self.model = config.model
        self.temperature = config.temperature

    def generate(self, system_message: str, user_message: str) -> str | None:
        """
        Calls OpenAI ChatCompletion API.

        Args:
            system_message: System role content.
            user_message: User role content.

        Returns:
            The content of the first choice message from the API response or None.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            temperature=self.temperature,
        )
        return response.choices[0].message.content


def create_llm(config: LLMConfig) -> BaseLLM:
    """
    Factory function to instantiate an LLM provider based on configuration.

    Args:
        config: The LLM configuration object.

    Returns:
        An instance of a class inheriting from BaseLLM.

    Raises:
        ValueError: If the provider specified in config is not supported.
    """
    if config.provider == "openai":
        return OpenAILLM(config)
    raise ValueError(f"Unsupported LLM provider: {config.provider}")
