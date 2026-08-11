# semantic_search/embedder.py

import logging
import os
from typing import Dict, List, Optional
import numpy as np
import httpx

logger = logging.getLogger(__name__)


# OpenRouter defaults (preserve backward compatibility)
DEFAULT_MODEL = "google/gemini-embedding-001"
DEFAULT_DIMENSION = 3072

# Local provider defaults — Ollama with nomic-embed-text out of the box.
# Override via LOCAL_EMBEDDING_BASE_URL / LOCAL_EMBEDDING_MODEL /
# LOCAL_EMBEDDING_DIMENSION when using a different server or model.
# For Turkish, intfloat/multilingual-e5-large (1024 dims, prompt_style=e5)
# served via HuggingFace TEI is the recommended setup — see README.
LOCAL_DEFAULT_BASE_URL = "http://localhost:11434/v1"
LOCAL_DEFAULT_MODEL = "nomic-embed-text"
LOCAL_DEFAULT_DIMENSION = 768

# Prompt-template styles. Embedding models are trained with specific
# prefixes — using the wrong style silently degrades retrieval quality.
#   - "gemini": "task: {task} | query: {text}" / "title: {title} | text: {text}"
#               (matches google/gemini-embedding-001, the OpenRouter default)
#   - "e5":     "query: {text}" / "passage: {text}"
#               (matches intfloat/multilingual-e5-* models — best for Turkish)
#   - "raw":    no prefix; pass text through as-is
PROMPT_STYLES = ("gemini", "e5", "raw")
DEFAULT_PROMPT_STYLE = "gemini"


def _format_query(prompt_style: str, query: str, task: str) -> str:
    if prompt_style == "e5":
        return f"query: {query}"
    if prompt_style == "raw":
        return query
    # gemini (default)
    return f"task: {task} | query: {query}"


def _format_document(prompt_style: str, doc: str, title: str) -> str:
    if prompt_style == "e5":
        return f"passage: {doc}"
    if prompt_style == "raw":
        return doc
    # gemini (default)
    return f"title: {title} | text: {doc}"


def _resolve_prompt_style(explicit: Optional[str], default: str) -> str:
    style = (explicit or os.getenv("EMBEDDING_PROMPT_STYLE") or default).strip().lower()
    if style not in PROMPT_STYLES:
        raise ValueError(
            f"Unknown EMBEDDING_PROMPT_STYLE {style!r}; expected one of {PROMPT_STYLES}"
        )
    return style


def is_openrouter_available() -> bool:
    """Check if OpenRouter API key is available."""
    return bool(os.getenv("OPENROUTER_API_KEY"))


def is_orcarouter_available() -> bool:
    """Check if OrcaRouter API key is available."""
    return bool(os.getenv("ORCAROUTER_API_KEY"))


def is_gemini_available() -> bool:
    """Check if a direct Google Gemini API key is available.

    This is a custom extension to the upstream project: it lets users who
    already have a Google AI Studio / Cloud Gemini key use the
    ``gemini-embedding-001`` model directly, without going through OpenRouter
    or OrcaRouter. Provider priority: local -> orca -> openrouter -> gemini.
    """
    return bool(os.getenv("GEMINI_API_KEY"))


def is_sentence_transformer_available() -> bool:
    """Check if the user opted into the lokal Sentence-Transformers embedder.

    Provider priority: local -> orcarouter -> openrouter -> gemini
    -> sentence_transformer. SentenceTransformer requires no API key and no
    network access at runtime — it works fully offline (after the one-time
    model download). Useful when Gemini API is region-blocked.
    """
    return os.getenv("EMBEDDING_PROVIDER", "").strip().lower() == "sentence_transformer"


def is_local_embedding_configured() -> bool:
    """Check if the user opted into a local embedding endpoint."""
    return os.getenv("EMBEDDING_PROVIDER", "").strip().lower() == "local"


def is_semantic_search_available() -> bool:
    """Returns True if any embedding provider is configured."""
    return (
        is_local_embedding_configured()
        or is_openrouter_available()
        or is_orcarouter_available()
        or is_gemini_available()
        or is_sentence_transformer_available()
    )


def _coerce_dimension(value, env_name: str, default: int) -> int:
    """Parse a dimension value (int or str) with clear error messages."""
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"{env_name} must be an integer, got {value!r}"
        ) from e
    if parsed <= 0:
        raise ValueError(f"Embedding dimension must be positive, got {parsed}")
    return parsed


class _BaseOpenAICompatibleEmbedder:
    """
    Shared encode/similarity logic for embedders backed by the OpenAI Python
    SDK. Subclasses configure ``client``, ``model``, ``dimension``, and
    optionally ``_extra_headers`` (e.g. OpenRouter ranking headers).
    """

    # Subclasses may override; sent on every embeddings.create call when set.
    _extra_headers: Dict[str, str] = {}

    # Set by subclasses
    client = None
    model: str = ""
    dimension: int = 0
    prompt_style: str = DEFAULT_PROMPT_STYLE

    def encode_query(self, query: str, task: str = "search result") -> np.ndarray:
        """
        Encode a search query. Prefix is selected by ``self.prompt_style``.

        Args:
            query: The search query text
            task: Task hint used by the gemini-style prefix; ignored for
                e5/raw styles.

        Returns:
            Numpy array of embeddings (``self.dimension`` elements).
        """
        text = _format_query(self.prompt_style, query, task)

        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text,
                encoding_format="float",
                extra_headers=self._extra_headers or None,
            )

            embedding = np.array(response.data[0].embedding, dtype=np.float32)

            # L2 normalize for cosine similarity
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            logger.debug(f"Encoded query: {query[:50]}... -> shape: {embedding.shape}")
            return embedding

        except Exception as e:
            logger.error(f"Failed to encode query: {e}")
            raise

    def encode_documents(self, documents: List[str], titles: Optional[List[str]] = None) -> np.ndarray:
        """
        Encode multiple documents with a batch API call.

        Args:
            documents: List of document texts
            titles: Optional list of document titles

        Returns:
            Numpy array of embeddings (N x ``self.dimension``).
        """
        if not documents:
            return np.array([])

        texts = []
        for i, doc in enumerate(documents):
            title = titles[i] if titles and i < len(titles) else "none"
            texts.append(_format_document(self.prompt_style, doc, title))

        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts,
                encoding_format="float",
                extra_headers=self._extra_headers or None,
            )

            embeddings = np.array(
                [d.embedding for d in sorted(response.data, key=lambda x: x.index)],
                dtype=np.float32,
            )

            # L2 normalize each embedding for cosine similarity
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-8)

            logger.info(f"Encoded {len(documents)} documents -> shape: {embeddings.shape}")
            return embeddings

        except Exception as e:
            logger.error(f"Failed to encode documents: {e}")
            raise

    def compute_similarity(self, query_embedding: np.ndarray, document_embeddings: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between query and documents.

        Args:
            query_embedding: Query embedding (``self.dimension``,)
            document_embeddings: Document embeddings (N x ``self.dimension``)

        Returns:
            Similarity scores (N,)
        """
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Embeddings are already L2-normalized.
        similarities = np.dot(document_embeddings, query_embedding.T).squeeze()
        return similarities


class OpenRouterEmbedder(_BaseOpenAICompatibleEmbedder):
    """
    Embedder using OpenRouter's embedding API.

    The model and dimension are configurable so users can pick any OpenRouter
    embedding model (e.g. when one becomes paid). Configuration precedence:
    explicit constructor args > environment variables > defaults.

    Environment variables:
        OPENROUTER_API_KEY (required): OpenRouter credential
        OPENROUTER_EMBEDDING_MODEL (optional): override the embedding model id
        OPENROUTER_EMBEDDING_DIMENSION (optional): override the vector size

    Defaults preserve backward compatibility: ``google/gemini-embedding-001``
    at 3072 dimensions.
    """

    _extra_headers = {
        "HTTP-Referer": "https://yargimcp.com",
        "X-Title": "Yargi MCP Server",
    }

    def __init__(
        self,
        model: Optional[str] = None,
        dimension: Optional[int] = None,
        prompt_style: Optional[str] = None,
    ):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is not set")

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package is required. Install with: pip install openai")

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.model = model or os.getenv("OPENROUTER_EMBEDDING_MODEL") or DEFAULT_MODEL
        self.dimension = _coerce_dimension(
            dimension if dimension is not None else os.getenv("OPENROUTER_EMBEDDING_DIMENSION"),
            "OPENROUTER_EMBEDDING_DIMENSION",
            DEFAULT_DIMENSION,
        )
        # Default to gemini-style prefix for OpenRouter — matches the default
        # google/gemini-embedding-001 model. Override via constructor or
        # EMBEDDING_PROMPT_STYLE env var when picking a different model.
        self.prompt_style = _resolve_prompt_style(prompt_style, "gemini")

        logger.info(
            f"OpenRouter Embedder initialized with model: {self.model} "
            f"(dimension={self.dimension}, prompt_style={self.prompt_style})"
        )


class OrcaRouterEmbedder(_BaseOpenAICompatibleEmbedder):
    """
    Embedder using OrcaRouter's OpenAI-compatible embedding API.

    OrcaRouter is a production AI gateway that proxies 200+ models on a single
    OpenAI-compatible endpoint. The model and dimension are configurable so
    users can pick any embedding model the gateway routes. Configuration
    precedence: explicit constructor args > environment variables > defaults.

    Environment variables:
        ORCAROUTER_API_KEY (required): OrcaRouter credential (sk-orca-...)
        ORCAROUTER_EMBEDDING_MODEL (optional): override the embedding model id
        ORCAROUTER_EMBEDDING_DIMENSION (optional): override the vector size

    Defaults: ``google/gemini-embedding-001`` at 3072 dimensions (multilingual,
    matches the OpenRouter default — good for Turkish legal text).
    """

    def __init__(
        self,
        model: Optional[str] = None,
        dimension: Optional[int] = None,
        prompt_style: Optional[str] = None,
    ):
        api_key = os.getenv("ORCAROUTER_API_KEY")
        if not api_key:
            raise ValueError("ORCAROUTER_API_KEY environment variable is not set")

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package is required. Install with: pip install openai")

        self.client = OpenAI(
            base_url="https://api.orcarouter.ai/v1",
            api_key=api_key,
        )
        self.model = model or os.getenv("ORCAROUTER_EMBEDDING_MODEL") or DEFAULT_MODEL
        self.dimension = _coerce_dimension(
            dimension if dimension is not None else os.getenv("ORCAROUTER_EMBEDDING_DIMENSION"),
            "ORCAROUTER_EMBEDDING_DIMENSION",
            DEFAULT_DIMENSION,
        )
        # Same gemini-style default as the OpenRouter embedder — matches the
        # multilingual google/gemini-embedding-001 default model.
        self.prompt_style = _resolve_prompt_style(prompt_style, "gemini")

        logger.info(
            f"OrcaRouter Embedder initialized with model: {self.model} "
            f"(dimension={self.dimension}, prompt_style={self.prompt_style})"
        )


class GeminiEmbedder:
    """
    Direct Google Gemini embedder — custom extension to the upstream project.

    Uses the native Google ``generativelanguage`` REST API via ``httpx`` so
    no extra SDK dependency is needed (httpx is already a project dep).
    This is the cheapest path to ``gemini-embedding-001`` (3072 dims,
    multilingual, state-of-the-art for Turkish legal text) if you already
    have a Google AI Studio API key — no OpenRouter middleman, no extra
    cost markup.

    Environment variables:
        GEMINI_API_KEY (required): Google AI Studio / Cloud API key
        GEMINI_EMBEDDING_MODEL (optional): default ``gemini-embedding-001``
        GEMINI_EMBEDDING_DIMENSION (optional): default ``3072``
        EMBEDDING_PROMPT_STYLE (optional): kept for API parity but ignored —
            Gemini uses taskType instead of prompt prefixes.
    """

    GEMINI_ENDPOINT = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "{model}:{method}"
    )

    def __init__(
        self,
        model: Optional[str] = None,
        dimension: Optional[int] = None,
        prompt_style: Optional[str] = None,  # ignored, kept for API parity
    ):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set")

        self.api_key = api_key
        self.model = (
            model
            or os.getenv("GEMINI_EMBEDDING_MODEL")
            or "gemini-embedding-001"
        )
        self.dimension = _coerce_dimension(
            dimension if dimension is not None else os.getenv("GEMINI_EMBEDDING_DIMENSION"),
            "GEMINI_EMBEDDING_DIMENSION",
            3072,
        )
        # Gemini uses taskType instead of text prefixes. We keep the attribute
        # for parity with the other embedders but it is not used.
        self.prompt_style = "gemini"
        # Sync httpx client — matches the OpenAI SDK's sync behavior used by
        # the other embedders. Proxy destek: HTTPS_PROXY/HTTP_PROXY env var'lari
        # otomatik kullanilir (Google Gemini API bazi bolgelerde bolge kısıtlamasi
        # oldugu icin kullanici VPN/proxy ile erismek isteyebilir).
        proxy = os.getenv("GEMINI_HTTPS_PROXY") or os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
        client_kwargs = {"timeout": 120.0}
        if proxy:
            client_kwargs["proxy"] = proxy
            logger.info(f"GeminiEmbedder: proxy kullanilacak: {proxy}")
        self._client = httpx.Client(**client_kwargs)

        logger.info(
            f"Gemini Embedder initialized with model: {self.model} "
            f"(dimension={self.dimension})"
        )

    def _embed_single(self, text: str, task_type: str) -> np.ndarray:
        url = self.GEMINI_ENDPOINT.format(
            model=self.model, method="embedContent"
        )
        payload = {
            "model": f"models/{self.model}",
            "content": {"parts": [{"text": text}]},
            "taskType": task_type,
        }
        # Optionally cap dimensionality if user requested a smaller output
        if self.dimension and self.dimension != 3072:
            payload["outputDimensionality"] = self.dimension

        try:
            response = self._client.post(
                url,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": self.api_key,
                },
            )
            response.raise_for_status()
            data = response.json()
            embedding = np.array(data["embedding"]["values"], dtype=np.float32)

            # L2 normalize for cosine similarity
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            return embedding
        except Exception as e:
            logger.error(f"GeminiEmbedder: failed to embed single text: {e}")
            raise

    def _embed_batch(self, texts: List[str], task_type: str) -> np.ndarray:
        # Gemini supports up to 100 texts per batchEmbedContents call.
        # We chunk to stay under that limit.
        all_embeddings: List[np.ndarray] = []
        BATCH_SIZE = 100
        for i in range(0, len(texts), BATCH_SIZE):
            chunk = texts[i : i + BATCH_SIZE]
            url = self.GEMINI_ENDPOINT.format(
                model=self.model, method="batchEmbedContents"
            )
            payload = {
                "model": f"models/{self.model}",
                "requests": [
                    {
                        "model": f"models/{self.model}",
                        "content": {"parts": [{"text": t}]},
                        "taskType": task_type,
                    }
                    for t in chunk
                ],
            }
            if self.dimension and self.dimension != 3072:
                for req in payload["requests"]:
                    req["outputDimensionality"] = self.dimension

            try:
                response = self._client.post(
                    url,
                    json=payload,
                    headers={
                        "Content-Type": "application/json",
                        "x-goog-api-key": self.api_key,
                    },
                )
                response.raise_for_status()
                data = response.json()
                for emb in data["embeddings"]:
                    arr = np.array(emb["values"], dtype=np.float32)
                    norm = np.linalg.norm(arr)
                    if norm > 0:
                        arr = arr / norm
                    all_embeddings.append(arr)
            except Exception as e:
                logger.error(
                    f"GeminiEmbedder: failed to embed batch (offset={i}, size={len(chunk)}): {e}"
                )
                raise

        return np.array(all_embeddings, dtype=np.float32)

    def encode_query(self, query: str, task: str = "search result") -> np.ndarray:
        """Encode a search query using RETRIEVAL_QUERY task type."""
        return self._embed_single(query, task_type="RETRIEVAL_QUERY")

    def encode_documents(
        self, documents: List[str], titles: Optional[List[str]] = None
    ) -> np.ndarray:
        """Encode documents using RETRIEVAL_DOCUMENT task type.

        Titles are ignored — Gemini's batch endpoint doesn't have a separate
        title field, and the title is expected to be merged into the text by
        the caller if needed.
        """
        if not documents:
            return np.array([])
        return self._embed_batch(documents, task_type="RETRIEVAL_DOCUMENT")

    def compute_similarity(
        self, query_embedding: np.ndarray, document_embeddings: np.ndarray
    ) -> np.ndarray:
        """Compute cosine similarity. Embeddings are L2-normalized."""
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        return np.dot(document_embeddings, query_embedding.T).squeeze()


class SentenceTransformerEmbedder:
    """
    Lokal Sentence-Transformers embedder — hic API key / bolge kistlamasi yok.

    Bu sinif, ``sentence-transformers`` kutuphanesi ile HuggingFace modellerini
    lokal olarak calistirir. İlk kullanimda model belirtilen HuggingFace
    cache'ine indirilir (bir defalik ~1-2GB download), sonraki kullanimlarda
    disk'ten yuklenir. Turkce hukuki metinler icin onerilen modeller:

    - ``intfloat/multilingual-e5-base`` (~1.1GB, 768-dim) — hiz + kalite dengesi
    - ``intfloat/multilingual-e5-large`` (~2.2GB, 1024-dim) — en iyi kalite
    - ``BAAI/bge-m3`` (~2.3GB, 1024-dim) — multilingual, uzun metin desteği
    - ``sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`` (~470MB, 384-dim)
      — en hizli, dusuk RAM gereksinimi

    E5 modelleri query/passage on ekleri bekler (prompt_style=e5). sentence-
    transformers bu on ekleri otomatik ekler; bu yuzden bu sinif _BaseOpenAICompatibleEmbedder
    'dan kalitim almaz — encode_query / encode_documents metodlarini kendisi uygular.

    Environment variables:
        EMBEDDING_PROVIDER=sentence_transformer (bunu secer)
        SENTENCE_TRANSFORMER_MODEL (default: intfloat/multilingual-e5-base)
        SENTENCE_TRANSFORMER_DEVICE (default: auto — cpu/cuda/mps)
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers package is required. Install with: "
                "pip install sentence-transformers"
            )

        self.model_name = (
            model_name
            or os.getenv("SENTENCE_TRANSFORMER_MODEL")
            or "intfloat/multilingual-e5-base"
        )
        device = device or os.getenv("SENTENCE_TRANSFORMER_DEVICE") or "cpu"

        logger.info(
            f"SentenceTransformerEmbedder: model yukleniyor: {self.model_name} (device={device})..."
        )
        self._model = SentenceTransformer(self.model_name, device=device)
        self.dimension = self._model.get_sentence_embedding_dimension()
        # E5 modelleri query:/passage: on ekleri bekler — sentence-transformers
        # bunlari otomatik ekler (model.config'ten).
        self.prompt_style = "e5"

        logger.info(
            f"SentenceTransformerEmbedder hazir: model={self.model_name} "
            f"dimension={self.dimension}"
        )

    def encode_query(self, query: str, task: str = "search result") -> np.ndarray:
        """Encode a search query. E5 prefix otomatik eklenir."""
        # sentence-transformers 3.x+ prompt destegi var; E5 modelleri icin
        # "query: " on ekini otomatik ekler. Eski surumler icin manuel ekliyoruz.
        text = query
        if "e5" in self.model_name.lower() and not query.lower().startswith("query:"):
            text = f"query: {query}"

        embedding = self._model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.array(embedding, dtype=np.float32)

    def encode_documents(
        self, documents: List[str], titles: Optional[List[str]] = None
    ) -> np.ndarray:
        """Encode multiple documents. E5 prefix otomatik eklenir. Batch'li."""
        if not documents:
            return np.array([])

        # E5 icin "passage: " on eki
        if "e5" in self.model_name.lower():
            texts = [
                f"passage: {doc}" if not doc.lower().startswith("passage:") else doc
                for doc in documents
            ]
        else:
            texts = documents

        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=32,
            show_progress_bar=False,
        )
        logger.info(f"Encoded {len(documents)} documents -> shape: {embeddings.shape}")
        return np.array(embeddings, dtype=np.float32)

    def compute_similarity(
        self, query_embedding: np.ndarray, document_embeddings: np.ndarray
    ) -> np.ndarray:
        """Cosine similarity (embeddings L2-normalize edilmis)."""
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        return np.dot(document_embeddings, query_embedding.T).squeeze()


class LocalEmbedder(_BaseOpenAICompatibleEmbedder):
    """
    Embedder for a local OpenAI-compatible embedding server — Ollama,
    llama.cpp, vLLM, LM Studio, etc. Zero new Python dependencies; just
    point the existing OpenAI SDK at a local base URL.

    Environment variables:
        EMBEDDING_PROVIDER=local              (selects this provider)
        LOCAL_EMBEDDING_BASE_URL              (default: http://localhost:11434/v1)
        LOCAL_EMBEDDING_MODEL                 (default: nomic-embed-text)
        LOCAL_EMBEDDING_DIMENSION             (default: 768)
        LOCAL_EMBEDDING_API_KEY               (optional; ignored by most local servers)

    Setup (Ollama):
        $ ollama serve
        $ ollama pull nomic-embed-text          # or bge-m3 for better Turkish

    The dimension MUST match the model's actual output size (e.g. 768 for
    nomic-embed-text, 1024 for bge-m3, 1024 for mxbai-embed-large).
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        dimension: Optional[int] = None,
        api_key: Optional[str] = None,
        prompt_style: Optional[str] = None,
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package is required. Install with: pip install openai")

        self.base_url = (
            base_url
            or os.getenv("LOCAL_EMBEDDING_BASE_URL")
            or LOCAL_DEFAULT_BASE_URL
        )
        # Most local servers don't validate the key — use a placeholder so
        # the OpenAI SDK doesn't error on the missing-key check.
        effective_key = (
            api_key
            or os.getenv("LOCAL_EMBEDDING_API_KEY")
            or "no-key-needed"
        )

        self.client = OpenAI(base_url=self.base_url, api_key=effective_key)
        self.model = model or os.getenv("LOCAL_EMBEDDING_MODEL") or LOCAL_DEFAULT_MODEL
        self.dimension = _coerce_dimension(
            dimension if dimension is not None else os.getenv("LOCAL_EMBEDDING_DIMENSION"),
            "LOCAL_EMBEDDING_DIMENSION",
            LOCAL_DEFAULT_DIMENSION,
        )
        # Default to e5 prefix for local — the recommended Turkish setup
        # (multilingual-e5-large). Override via EMBEDDING_PROMPT_STYLE when
        # using a different model family (e.g. nomic, bge).
        self.prompt_style = _resolve_prompt_style(prompt_style, "e5")

        logger.info(
            f"Local Embedder initialized: model={self.model} "
            f"base_url={self.base_url} dimension={self.dimension} "
            f"prompt_style={self.prompt_style}"
        )


def get_embedder():
    """
    Factory that picks the embedder based on EMBEDDING_PROVIDER.

    - ``EMBEDDING_PROVIDER=local`` -> ``LocalEmbedder`` (Ollama etc.)
    - ``EMBEDDING_PROVIDER=sentence_transformer`` -> ``SentenceTransformerEmbedder``
    - ``ORCAROUTER_API_KEY`` set -> ``OrcaRouterEmbedder``
    - ``OPENROUTER_API_KEY`` set -> ``OpenRouterEmbedder``
    - ``GEMINI_API_KEY`` set -> ``GeminiEmbedder`` (custom extension)

    Raises:
        ValueError: If no provider is configured.
    """
    if is_local_embedding_configured():
        return LocalEmbedder()
    if is_sentence_transformer_available():
        return SentenceTransformerEmbedder()
    if is_orcarouter_available():
        return OrcaRouterEmbedder()
    if is_openrouter_available():
        return OpenRouterEmbedder()
    if is_gemini_available():
        return GeminiEmbedder()
    raise ValueError(
        "No embedding provider configured. Set one of: OPENROUTER_API_KEY, "
        "ORCAROUTER_API_KEY, GEMINI_API_KEY (direct Google Gemini), "
        "EMBEDDING_PROVIDER=local (Ollama etc.), or "
        "EMBEDDING_PROVIDER=sentence_transformer (offline, no API key)."
    )
