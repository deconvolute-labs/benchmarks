from unittest.mock import MagicMock, patch

import pytest

from dcv_benchmark.models.experiments_config import TargetConfig
from dcv_benchmark.targets.basic_rag import BasicRAG


@pytest.fixture
def mock_config():
    config = MagicMock(spec=TargetConfig)
    config.llm = MagicMock()
    config.llm.provider = "mock_provider"
    config.llm.model = "mock_model"
    config.embedding = MagicMock()
    config.retriever = MagicMock()

    # Defaults
    config.generate = True
    config.defense = MagicMock()
    config.defense.canary = None
    config.defense.language = None
    config.defense.yara = None  # Start with no YARA

    config.prompt_template = MagicMock()
    config.prompt_template.file = "t.yaml"
    config.prompt_template.key = "k"

    config.system_prompt = MagicMock()
    config.system_prompt.file = "s.yaml"
    config.system_prompt.key = "sk"

    return config


@pytest.fixture
def basic_rag(mock_config):
    with (
        patch("dcv_benchmark.targets.basic_rag.create_llm") as mock_create_llm,
        patch("dcv_benchmark.targets.basic_rag.create_vector_store"),
        patch("dcv_benchmark.targets.basic_rag.load_prompt_text"),
    ):
        mock_llm = MagicMock()
        mock_create_llm.return_value = mock_llm

        rag = BasicRAG(mock_config)
        rag.llm = mock_llm
        return rag


def test_scan_hit_blocking(basic_rag, mock_config):
    """
    Case 1: Threat Detected in Forced Context -> Blocked.
    Should return attack_detected=True, content="[Blocked...]", no LLM call.
    """
    # Enable Signature Detector via config mocking
    # Note: BasicRAG.__init__ checks config.defense.yara.enabled
    # But since we already init'd, we manually patch signature_detector
    mock_detector = MagicMock()

    # Setup Hit
    mock_result = MagicMock()
    mock_result.threat_detected = True
    mock_result.metadata = "Found Bad Thing"
    mock_detector.check.return_value = mock_result

    basic_rag.signature_detector = mock_detector

    scan_context = ["malicious context"]

    response = basic_rag.invoke(user_query="irrelevant", forced_context=scan_context)

    # Assertions
    assert response.attack_detected is True
    assert response.detection_reason == "Signature Scan: Found Bad Thing"
    assert "Blocked" in response.content

    # Ensure LLM NOT called
    basic_rag.llm.generate.assert_not_called()

    # Ensure Scan checked the context
    mock_detector.check.assert_called_with("malicious context")


def test_scan_miss_scan_mode(basic_rag, mock_config):
    """
    Case 2: No Threat Detected + generate=False (Scan Mode).
    Should return attack_detected=False, empty content, no LLM call.
    """
    # Enable Signature Detector (Miss)
    mock_detector = MagicMock()
    mock_result = MagicMock()
    mock_result.threat_detected = False
    mock_detector.check.return_value = mock_result

    basic_rag.signature_detector = mock_detector

    # Set to Scan Mode
    basic_rag.config.generate = False

    scan_context = ["clean context"]

    response = basic_rag.invoke(user_query="irrelevant", forced_context=scan_context)

    # Assertions
    assert response.attack_detected is False
    assert response.content == ""
    assert response.metadata.get("skipped_generation") is True

    # Ensure LLM NOT called
    basic_rag.llm.generate.assert_not_called()


def test_scan_miss_generate_mode(basic_rag, mock_config):
    """
    Case 3: No Threat Detected + generate=True (Normal Mode).
    Should proceed to LLM generation.
    """
    mock_detector = MagicMock()
    mock_result = MagicMock()
    mock_result.threat_detected = False
    mock_detector.check.return_value = mock_result

    basic_rag.signature_detector = mock_detector
    basic_rag.config.generate = True

    basic_rag.llm.generate.return_value = "LLM Response"

    response = basic_rag.invoke(user_query="query", forced_context=["clean"])

    assert response.content == "LLM Response"
    assert "skipped_generation" not in response.metadata

    basic_rag.llm.generate.assert_called_once()
