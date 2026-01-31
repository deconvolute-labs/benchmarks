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

    # Mock nested defense structure
    config.defense = MagicMock()
    # Ingestion
    config.defense.ingestion = MagicMock()
    config.defense.ingestion.signature_detector = MagicMock()
    config.defense.ingestion.signature_detector.enabled = False

    config.defense.ingestion.ml_detector = MagicMock()
    config.defense.ingestion.ml_detector.enabled = False

    # Generation
    config.defense.generation = MagicMock()

    config.defense.generation.prompt_guard = MagicMock()
    config.defense.generation.prompt_guard.enabled = False

    config.defense.generation.canary_detector = MagicMock()
    config.defense.generation.canary_detector.enabled = False

    config.defense.generation.language_detector = MagicMock()
    config.defense.generation.language_detector.enabled = False

    # Default generate to True (Normal Mode)
    config.generate = True

    # Mock system_prompt and prompt_template as objects with path/key
    config.prompt_template = MagicMock()
    config.prompt_template.file = "template_path.yaml"
    config.prompt_template.key = "template_key"

    config.system_prompt = MagicMock()
    config.system_prompt.file = "system_path.yaml"
    config.system_prompt.key = "system_key"

    return config


@pytest.fixture
def basic_rag(mock_config):
    with (
        patch("dcv_benchmark.targets.basic_rag.create_llm") as mock_create_llm,
        patch("dcv_benchmark.targets.basic_rag.create_vector_store") as mock_create_vs,
        patch("dcv_benchmark.targets.basic_rag.load_prompt_text") as mock_load_prompt,
    ):
        mock_llm = MagicMock()
        mock_create_llm.return_value = mock_llm

        mock_vs = MagicMock()
        mock_create_vs.return_value = mock_vs

        # Configure load_prompt_text to return different strings based on inputs
        def side_effect(path, key):
            if path == "template_path.yaml":
                return "{context}\n\n{query}"
            if path == "system_path.yaml":
                return "You are a helpful assistant."
            return "default text"

        mock_load_prompt.side_effect = side_effect

        rag = BasicRAG(mock_config)
        rag.llm = mock_llm
        rag.vector_store = mock_vs
        return rag


def test_init_full_config(mock_config):
    with (
        patch("dcv_benchmark.targets.basic_rag.create_llm") as mock_create_llm,
        patch("dcv_benchmark.targets.basic_rag.create_vector_store") as mock_create_vs,
        patch("dcv_benchmark.targets.basic_rag.load_prompt_text") as mock_load_prompt,
    ):
        BasicRAG(mock_config)

        mock_create_llm.assert_called_once_with(mock_config.llm)
        mock_create_vs.assert_called_once_with(
            mock_config.retriever, mock_config.embedding
        )

        # Verify prompt loading
        assert mock_load_prompt.call_count == 2
        mock_load_prompt.assert_any_call(path="system_path.yaml", key="system_key")
        mock_load_prompt.assert_any_call(path="template_path.yaml", key="template_key")


def test_init_no_retriever(mock_config):
    mock_config.retriever = None
    with (
        patch("dcv_benchmark.targets.basic_rag.create_llm") as mock_create_llm,
        patch("dcv_benchmark.targets.basic_rag.create_vector_store") as mock_create_vs,
        patch("dcv_benchmark.targets.basic_rag.load_prompt_text"),
    ):
        rag = BasicRAG(mock_config)

        mock_create_llm.assert_called_once()
        mock_create_vs.assert_not_called()
        assert rag.vector_store is None


def test_init_canary_enabled(mock_config):
    # Enable canary in nested config
    mock_config.defense.generation.canary_detector.enabled = True
    mock_config.defense.generation.canary_detector.settings = {}

    with (
        patch("dcv_benchmark.targets.basic_rag.CanaryDetector") as MockCanary,
        patch("dcv_benchmark.targets.basic_rag.create_llm"),
        patch("dcv_benchmark.targets.basic_rag.create_vector_store"),
        patch("dcv_benchmark.targets.basic_rag.load_prompt_text"),
    ):
        BasicRAG(mock_config)

        # Check if canary detector was initialized in the layers
        # BasicRAG now stores detectors in .layers list or similar?
        # Let's check BasicRAG implementation.
        # It calls self._init_defenses(config)
        # Inside: self.generation_layers.append(CanaryDetector(...))
        # We can inspect rag.generation_layers or similar if exposed,
        # or check MockCanary called.
        MockCanary.assert_called_once()


def test_ingest_with_store(basic_rag):
    docs = ["doc1", "doc2"]
    basic_rag.ingest(docs)
    basic_rag.vector_store.add_documents.assert_called_once_with(docs)


def test_ingest_no_store(basic_rag):
    basic_rag.vector_store = None
    docs = ["doc1", "doc2"]
    basic_rag.ingest(docs)


def test_invoke_basic_flow(basic_rag):
    user_query = "Hello"
    basic_rag.vector_store.search.return_value = ["context1"]
    basic_rag.llm.generate.return_value = "Response"

    response = basic_rag.invoke(user_query)

    basic_rag.vector_store.search.assert_called_once_with(user_query)
    basic_rag.llm.generate.assert_called_once()
    assert response.content == "Response"
    assert response.used_context == ["context1"]


def test_invoke_retrieve_only(basic_rag):
    basic_rag.vector_store.search.return_value = ["context1"]
    response = basic_rag.invoke("query", retrieve_only=True)

    assert response.content == ""
    assert response.used_context == ["context1"]
    basic_rag.llm.generate.assert_not_called()


def test_invoke_forced_context(basic_rag):
    basic_rag.llm.generate.return_value = "Response"

    forced = ["forced1"]
    # Pass user_query first
    response = basic_rag.invoke("query", forced_context=forced)

    basic_rag.vector_store.search.assert_not_called()
    assert response.used_context == forced
    assert response.content == "Response"


def test_invoke_canary_protection(basic_rag):
    # Enable Canary manually on the instance

    mock_canary_layer = MagicMock()
    # BasicRAG uses self.canary attribute
    basic_rag.canary = mock_canary_layer

    mock_canary_layer.inject.return_value = ("guarded_prompt", "token123")

    # Mock result so detected is False (safe)
    mock_result = MagicMock()
    mock_result.threat_detected = False
    mock_canary_layer.check.return_value = mock_result
    mock_canary_layer.clean.return_value = "Cleaned Response"

    basic_rag.llm.generate.return_value = "Raw Response"

    response = basic_rag.invoke(user_query="query")

    # Verify inject called with loaded system prompt (from fixture side_effect)
    mock_canary_layer.inject.assert_called_once_with("You are a helpful assistant.")

    mock_canary_layer.check.assert_called_once()
    mock_canary_layer.clean.assert_called_once()
    assert response.content == "Cleaned Response"


def test_invoke_canary_triggered(basic_rag):
    mock_canary_layer = MagicMock()
    basic_rag.canary = mock_canary_layer

    mock_canary_layer.inject.return_value = ("guarded_prompt", "token123")

    mock_result = MagicMock()
    # It might use .detected or .threat_detected depending on actual implementation
    # Assuming BasicRAG logic uses .threat_detected based on check() return
    mock_result.threat_detected = True
    mock_canary_layer.check.return_value = mock_result

    basic_rag.llm.generate.return_value = "Raw Response"

    response = basic_rag.invoke("query")

    assert response.attack_detected is True
    assert response.detection_reason == "Canary Integrity Check Failed"
    assert "Response blocked" in response.content


def test_invoke_no_llm(basic_rag):
    basic_rag.llm = None
    # Assuming BasicRAG handles None LLM gracefully (e.g. scan mode or error)
    # If using invoke without retrieve_only, it probably crashes or
    # returns error if generate=True.

    # If generate=False (Scan Mode), it returns "blocked" or "scan"
    basic_rag.config.generate = False
    response = basic_rag.invoke("query")
    # Scan mode returns metadata
    assert response.metadata.get("stage") == "scan"
