from unittest.mock import MagicMock, patch

import pytest

from dcv_benchmark.components.llms import OpenAILLM, create_llm
from dcv_benchmark.models.experiments_config import LLMConfig


@pytest.fixture
def openai_config():
    return LLMConfig(provider="openai", model="gpt-4", temperature=0.7)


def test_create_openai_llm(openai_config):
    """It should return an OpenAILLM instance correctly initialized."""
    # We patch openai.Client to avoid needing an API key or making network calls
    with patch("dcv_benchmark.components.llms.openai.Client"):
        llm = create_llm(openai_config)

    assert isinstance(llm, OpenAILLM)
    assert llm.model == "gpt-4"
    assert llm.temperature == 0.7


def test_unsupported_provider():
    """It should raise ValueError for unknown providers."""
    mock_config = MagicMock()
    mock_config.provider = "unsupported_provider"

    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        create_llm(mock_config)
