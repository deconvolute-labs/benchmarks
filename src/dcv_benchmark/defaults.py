from typing import Final

# LLM Defaults
DEFAULT_LLM_PROVIDER: Final[str] = "openai"
DEFAULT_LLM_MODEL: Final[str] = "gpt-4.1-mini"
DEFAULT_LLM_TEMPERATURE: Final[float] = 0.0

# Embedding Defaults
DEFAULT_EMBEDDING_PROVIDER: Final[str] = "openai"
DEFAULT_EMBEDDING_MODEL: Final[str] = "text-embedding-3-small"

# Retriever Defaults
DEFAULT_RETRIEVER_PROVIDER: Final[str] = "chromadb"
DEFAULT_RETRIEVER_K: Final[int] = 5

# Prompt Defaults
DEFAULT_SYSTEM_PROMPT_KEY: Final[str] = "standard"
DEFAULT_TEMPLATE_KEY: Final[str] = "rag_standard_v1"
