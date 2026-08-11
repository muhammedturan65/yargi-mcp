# Yargı MCP — Türkçe Kurulum ve Kullanım Kılavuzu

> Bu dosya, `saidsurucu/yargi-mcp` reposunun kurulumu, semantic search özelliğinin **doğrudan Gemini API key** ile çalıştırılması ve hukuki yapay zeka sistemi geliştirmek için pratik kullanım yollarını açıklar.
>
> Orijinal repoya ek olarak: `GeminiEmbedder` sınıfı eklenmiştir (OpenRouter middleman olmadan Google'ın `gemini-embedding-001` modeline doğrudan erişim).

---

## İçindekiler

1. [Proje Özeti](#1-proje-özeti)
2. [Hızlı Başlangıç (3 dakika)](#2-hızlı-başlangıç-3-dakika)
3. [Gemini API Key Alma](#3-gemini-api-key-alma)
4. [Semantic Search Nasıl Çalışır](#4-semantic-search-nasıl-çalışır)
5. [Tüm MCP Tool'ları (29 adet)](#5-tüm-mcp-toolları-29-adet)
6. [MCP İstemci Konfigürasyon Örnekleri](#6-mcp-i̇stemci-konfigürasyon-örnekleri)
7. [HTTP Server Modu (Streamable HTTP)](#7-http-server-modu-streamable-http)
8. [Docker ve Railway Deployment](#8-docker-ve-railway-deployment)
9. [Hukuki Yapay Zeka Sistemi Mimari Önerisi](#9-hukuki-yapay-zeka-sistemi-mimari-önerisi)
10. [Sorun Giderme](#10-sorun-giderme)

---

## 1. Proje Özeti

**Yargı MCP**, Türk hukuk kaynaklarına (mahkeme kararları, kurum kararları, özelgeler) Model Context Protocol (MCP) üzerinden erişim sağlayan bir **FastMCP** sunucusudur. MCP destekleyen herhangi bir LLM istemcisi (Claude Desktop, Cursor, 5ire, Gemini CLI, Google Antigravity, ChatGPT) bu araçları kullanabilir.

### Kapsam — 16 Hukuk Kaynağı

| # | Kaynak | Açıklama |
|---|--------|---------|
| 1 | **Yargıtay** | Yargıtay kararları (Bedesten üzerinden, çift API) |
| 2 | **Danıştay** | Danıştay kararları (Bedesten üzerinden) |
| 3 | **Anayasa Mahkemesi** | Norm denetimi + Bireysel başvuru (birleşik API) |
| 4 | **KİK** | Kamu İhale Kurulu kararları (uyuşmazlık, düzenleyici, mahkeme) |
| 5 | **GİB** | Gelir İdaresi Başkanlığı özelgeleri (18.000+ özelge) |
| 6 | **BDDK** | Bankacılık Düzenleme ve Denetleme Kurumu kararları |
| 7 | **BTK** | Bilgi Teknolojileri ve İletişim Kurulu kararları |
| 8 | **KVKK** | Kişisel Verileri Koruma Kurulu kararları |
| 9 | **Rekabet Kurumu** | Rekabet Kurulu kararları (5 tür: birleşme, ihlal vb.) |
| 10 | **Sayıştay** | Sayıştay kararları (Genel Kurul, Temyiz, Daire) |
| 11 | **Sigorta Tahkim** | Sigorta Tahkim Komisyonu Hakem Karar Dergileri (1-64) |
| 12 | **Emsal (UYAP)** | UYAP emsal karar veritabanı |
| 13 | **Uyuşmazlık Mahkemesi** | Yetki uyuşmazlığı kararları |
| 14 | **Bedesten** | Çoklu mahkeme türleri için ortak API (Yerel, İstinaf, KYB) |
| 15 | **Sağlık Kontrolü** | Yargıtay + Bedesten sunucu canlılık kontrolü |
| 16 | **Deep Research** | ChatGPT Deep Research uyumlu minimal tool'lar |

### Teknik Özellikler

- **Dil/Çatı:** Python 3.11+, FastMCP 2.10+, Pydantic v2, httpx
- **Toplam tool sayısı:** 28 (semantic search kapalıyken) / 29 (semantic search açıkken)
- **Token optimizasyonu:** %61.8 token tasarrufu (14.061 → 5.369 token)
- **Filtreleme:** 87 daire/kurul (52 Yargıtay + 27 Danıştay + 8 Sayıştay)
- **Lisans:** MIT (açık kaynak)

### Semantic Search (Opsiyonel)

Projenin en güçlü özelliği. Klasik keyword arama yerine, sorguyu ve belgeleri vektörlere çevirip **kosinus benzerliği** ile sıralar. Kullanıcı bir hukuki meseleyi serbest cümleyle anlatır (örn. *"İşçinin kidem tazminatına hak kazanma şartları"*), sistem en alakalı kararları döner.

---

## 2. Hızlı Başlangıç (3 dakika)

### Ön gereksinimler

- Python 3.11+ (bu repoda 3.12 kullanılıyor)
- `uv` paket yöneticisi — kurmak için:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

### Adım 1 — Repo zaten klonlu (bu kurulumda)

Repo `/home/z/my-project/research/yargi-mcp` altında. Yeni bir yerde kurmak isterseniz:
```bash
git clone https://github.com/saidsurucu/yargi-mcp.git
cd yargi-mcp
```

### Adım 2 — Bağımlılıkları yükle

```bash
cd /home/z/my-project/research/yargi-mcp
uv sync --extra asgi   # MCP + FastAPI + uvicorn
```

Bu komut tüm bağımlılıkları `.venv` altına kurar (fastmcp, httpx, openai, numpy, fastapi, uvicorn, pydantic, cryptography, pypdf, markitdown, beautifulsoup4).

### Adım 3 — Gemini API Key ayarla

`.env` dosyasını düzenle:
```bash
nano /home/z/my-project/research/yargi-mcp/.env
```

Şu satırı bul ve gerçek key'ini yaz:
```env
GEMINI_API_KEY=AIzaSy...sizin_real_keyiniz
```

> Key almak için: https://aistudio.google.com/app/apikey (ücretsiz tier: 1500 istek/gün)

### Adım 4 — Test et

```bash
python /home/z/my-project/scripts/test_gemini_semantic_search.py
```

5 test çalışır:
1. GeminiEmbedder tekil query embedding
2. GeminiEmbedder batch belge embedding
3. VectorStore + DocumentProcessor ile semantic search pipeline
4. DocumentProcessor'ın gerçek bir Yargıtay karar metnini parçalaması
5. MCP server'da `search_bedesten_semantic` tool'unun yüklendiğinin doğrulanması

Tüm testler "TUM TESTLER BASARILI!" mesajını verirse sistem hazır.

### Adım 5 — Server'ı başlat

**Stdio modu** (Claude Desktop / Cursor için):
```bash
cd /home/z/my-project/research/yargi-mcp
uv run yargi-mcp
# veya: uv run python mcp_server_main.py
```

**HTTP modu** (uzak MCP istemcileri için):
```bash
cd /home/z/my-project/research/yargi-mcp
uv run uvicorn asgi_app:app --host 0.0.0.0 --port 8000
# MCP endpoint: http://localhost:8000/mcp
# Sağlık: http://localhost:8000/health
# Status: http://localhost:8000/status
```

---

## 3. Gemini API Key Alma

Google AI Studio üzerinden **ücretsiz** Gemini API key alınabilir. Bu, OpenRouter middleman olmadan `gemini-embedding-001` (3072 boyut, çok dilli, Türkçe için state-of-the-art) modelini kullanmanın en ucuz yoludur.

### Adımlar

1. **Google AI Studio'ya git:** https://aistudio.google.com/app/apikey
2. Google hesabınla giriş yap
3. **"Create API Key"** butonuna tıkla
4. Yeni bir Google Cloud projesi oluştur (ücretsiz) veya mevcut bir projeyi seç
5. API key'i kopyala (`AIzaSy...` ile başlar)
6. `.env` dosyasına yapıştır:
   ```env
   GEMINI_API_KEY=AIzaSy............
   ```

### Free Tier Limitleri (2026 itibariyle)

- **Embedding API:** 1500 istek/gün (gemini-embedding-001)
- **Batch istek:** her çağrıda 100 metne kadar (bu proje bunu kullanıyor)
- **Bölge:** Global (TR'den erişilebilir)
- **Kredi kartı:** Gerekmez

### Semantic search tüketimi

Her `search_bedesten_semantic` çağrısı:
- 1 query embedding (1 istek)
- ~100 belge için batch embedding (1 istek, batch içinde 100 metin)
- **Toplam: ~2 API isteği** (semantic search başına)

Yani 1500 istek/gün ≈ 750 semantic search sorgusu/gün. Bu, geliştirme ve orta ölçekli kullanım için fazlasıyla yeterli.

### Maliyet kıyaslaması

| Sağlayıcı | Model | Fiyat (1M token) | Free tier |
|-----------|-------|------------------|-----------|
| **Direct Gemini (bu proje)** | gemini-embedding-001 | Ücretsiz | 1500 req/gün |
| OpenRouter | google/gemini-embedding-001 | ~$0.0125 | Yok (ücretli) |
| OrcaRouter | google/gemini-embedding-001 | ~$0.0125 | Yok (ücretli) |
| Local (TEI) | multilingual-e5-large | Ücretsiz | Sınırsız (ama sunucu gerekir) |

---

## 4. Semantic Search Nasıl Çalışır

### Mimari

```
Kullanici sorgusu (serbest cümle)
       │
       ▼
┌─────────────────────────────────────────────┐
│ 1. Bedesten API'den ilk 100 karar cek       │  (keyword ile)
│                                              │
│ 2. Her karar icin tam metni markdown'a cevir │
│    DocumentProcessor ile parcala             │
│    (chunk_size=1500, overlap=300)            │
│                                              │
│ 3. GeminiEmbedder.encode_query(query)        │  → 3072 boyutlu vektor
│    GeminiEmbedder.encode_documents(100 belge)│  → 100x3072 matris
│                                              │
│ 4. VectorStore.search(                       │
│        query_embedding,                      │
│        top_k=10,                             │
│        threshold=0.3                         │
│    )                                         │  → kosinus benzerligi
│                                              │
│ 5. En alakali 10 karari formatla ve don      │
└─────────────────────────────────────────────┘
       │
       ▼
  JSON yanit (id, title, similarity_score, preview, metadata, source_url)
```

### Parametreler

`search_bedesten_semantic` tool'unun 4 parametresi var:

| Parametre | Tip | Zorunlu | Açıklama |
|-----------|-----|---------|---------|
| `initial_keyword` | str | Evet | Bedesten'den ilk sonuçları çekmek için anahtar kelime. `+`, `-`, `AND`, `OR`, `NOT`, `"tam eşleşme"` destekler. |
| `query` | str | Evet | Semantik benzerlik için **detaylı cümle**. Anahtar kelime değil, hukuki meseleyi anlatan tam bir cümle olmalı. |
| `court_types` | List[Enum] | Hayır | Mahkeme türleri. Default: tümü (Yargıtay, Danıştay, Yerel, İstinaf, KYB) |
| `top_k` | int (1-50) | Hayır | Dönecek en iyi kaç sonuç. Default: 10 |

### Örnek kullanım (Claude Desktop'ta)

**Kullanıcı:** "İşçinin işverenin iflası nedeniyle kıdem tazminatına hak kazanıp kazanamayacağına dair emsal kararlar ara"

**Claude şu tool'u çağırır:**
```json
{
  "name": "search_bedesten_semantic",
  "arguments": {
    "initial_keyword": "kidem tazminati iflas",
    "query": "Iscinin isverenin iflas etmesi sebebiyle is sozlesmesinin sona ermesi halinde kidem tazminatina hak kazanıp kazanamayacagi",
    "court_types": ["YARGITAYKARARI", "YERELHUKUK"],
    "top_k": 5
  }
}
```

### Prompt Style Mekanizması

Gemini, OpenRouter/OrcaRouter'tan farklı olarak **prompt prefix kullanmaz** — bunun yerine `taskType` kullanır:

- `RETRIEVAL_QUERY` — sorgu için (GeminiEmbedder.encode_query bunu kullanır)
- `RETRIEVAL_DOCUMENT` — belge için (GeminiEmbedder.encode_documents bunu kullanır)

Diğer sağlayıcılar için:
- `gemini` style: `task: {task} | query: {text}` / `title: {title} | text: {doc}`
- `e5` style: `query: {text}` / `passage: {text}` (Türkçe local için önerilen)
- `raw` style: öneksiz

Yanlış prompt style kullanırsanız retrieval kalitesi sessizce düşer. GeminiEmbedder bu konuyu otomatik halleder.

---

## 5. Tüm MCP Tool'ları (29 adet)

### Bedesten (Birleşik Arama)
- `search_bedesten_unified` — Yargıtay/Danıştay/Yerel/İstinaf/KYB birleşik arama
- `get_bedesten_document_markdown` — Bedesten belgesini Markdown olarak getir
- `search_bedesten_semantic` — **AI embedding'lerle semantik arama** (Gemini/OpenRouter/OrcaRouter/Local)

### Anayasa Mahkemesi
- `search_anayasa_unified` — Norm denetimi + Bireysel başvuru (birleşik)
- `get_anayasa_document_unified` — URL'den AYM kararı getir (sayfalı)

### KİK (Kamu İhale Kurulu)
- `search_kik_v2_decisions` — Karar arama (uyuşmazlık/düzenleyici/mahkeme)
- `get_kik_v2_document_markdown` — Kararın Markdown'u

### GİB (Gelir İdaresi)
- `search_gib_ozelge` — 18.000+ özelge arama (kanunNo filtresi)
- `get_gib_ozelge_document_markdown` — Özelge Markdown'u

### BDDK
- `search_bddk_decisions` — Bankacılık regülasyon kararları
- `get_bddk_document_markdown` — Karar Markdown'u

### BTK
- `search_btk_decisions` — Kurul kararları (no, tarih, birim filtreleri)
- `get_btk_document_markdown` — PDF → Markdown

### KVKK
- `search_kvkk_decisions` — Kişisel veri koruma kararları
- `get_kvkk_document_markdown` — Sayfalı Markdown

### Rekabet Kurumu
- `search_rekabet_kurumu_decisions` — 5 karar türü (birleşme, ihlal vb.)
- `get_rekabet_kurumu_document` — Sayfalı Markdown

### Sayıştay
- `search_sayistay_unified` — Genel Kurul / Temyiz / Daire (birleşik)
- `get_sayistay_document_unified` — Karar Markdown'u

### Sigorta Tahkim
- `search_sigorta_tahkim_decisions` — Hakem Karar Dergisi arama (sayı 1-64)
- `get_sigorta_tahkim_document_markdown` — PDF dergi → Markdown
- `search_within_sigorta_tahkim_issue` — Belirli bir dergi içinde anahtar kelime

### Emsal (UYAP)
- `search_emsal_detailed_decisions` — UYAP emsal karar arama (mahkeme + esas/karar + tarih)
- `get_emsal_document_markdown` — Emsal karar Markdown'u

### Uyuşmazlık Mahkemesi
- `search_uyusmazlik_decisions` — Yetki uyuşmazlığı kararları
- `get_uyusmazlik_document_markdown_from_url` — URL ile karar Markdown'u

### Sağlık Kontrolü
- `check_government_servers_health` — Yargıtay + Bedesten sunucu canlılık kontrolü

### ChatGPT Deep Research
- `search` — Deep Research uyumlu minimal arama (sadece Bedesten)
- `fetch` — Deep Research uyumlu minimal belge getirme

---

## 6. MCP İstemci Konfigürasyon Örnekleri

### Claude Desktop (macOS/Windows/Linux)

`claude_desktop_config.json` konumu:
- **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`
- **Linux:** `~/.config/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "yargi-mcp": {
      "command": "uv",
      "args": ["--directory", "/home/z/my-project/research/yargi-mcp", "run", "yargi-mcp"],
      "env": {
        "GEMINI_API_KEY": "AIzaSy...sizin_real_keyiniz"
      }
    }
  }
}
```

### Cursor

`~/.cursor/mcp.json`:
```json
{
  "mcpServers": {
    "yargi-mcp": {
      "command": "uv",
      "args": ["--directory", "/home/z/my-project/research/yargi-mcp", "run", "yargi-mcp"],
      "env": {
        "GEMINI_API_KEY": "AIzaSy...sizin_real_keyiniz"
      }
    }
  }
}
```

### Gemini CLI

`~/.gemini/settings.json`:
```json
{
  "mcpServers": {
    "yargi-mcp": {
      "command": "uv",
      "args": ["--directory", "/home/z/my-project/research/yargi-mcp", "run", "yargi-mcp"],
      "env": {
        "GEMINI_API_KEY": "AIzaSy...sizin_real_keyiniz"
      }
    }
  }
}
```

### Google Antigravity

`~/.gemini/config/mcp_config.json`:
```json
{
  "mcpServers": {
    "yargi-mcp": {
      "command": "uv",
      "args": ["--directory", "/home/z/my-project/research/yargi-mcp", "run", "yargi-mcp"],
      "env": {
        "GEMINI_API_KEY": "AIzaSy...sizin_real_keyiniz"
      }
    }
  }
}
```

### 5ire

UI üzerinden: `Tools → +Local` →
- **Tool Key:** `yargimcp`
- **Name:** `Yargı MCP`
- **Command:** `uv`
- **Args:** `--directory /home/z/my-project/research/yargi-mcp run yargi-mcp`

### Remote MCP (kurulum gerektirmeden)

Yalnızca karar arama (semantic search yok) için, geliştiricinin sunduğu remote MCP kullanılabilir:
```
https://yargimcp.surucu.dev/mcp
```

Claude Desktop'da: `Settings → Connectors → Add Custom Connector` → URL'yi gir.

> Not: Remote MCP'de semantic search **yapılandırılmamıştır**. Semantic search için self-host şart.

---

## 7. HTTP Server Modu (Streamable HTTP)

Lokal HTTP server olarak çalıştırmak için 2 seçenek:

### Minimal (app.py) — Docker ile uyumlu

```bash
cd /home/z/my-project/research/yargi-mcp
export GEMINI_API_KEY=AIzaSy...
uv run uvicorn app:app --host 0.0.0.0 --port 8000
```

Endpoint'ler:
- `GET /health` → `{"status":"healthy","service":"Yargı MCP Server","version":"0.2.1"}`
- `POST /mcp/` → MCP Streamable HTTP endpoint

### Zengin (asgi_app.py) — Railway ile uyumlu

```bash
cd /home/z/my-project/research/yargi-mcp
export GEMINI_API_KEY=AIzaSy...
uv run uvicorn asgi_app:app --host 0.0.0.0 --port 8000
```

Endpoint'ler:
- `GET /health` → `{"status":"healthy","service":"Yargı MCP Server","version":"0.1.0","tools_count":29}`
- `GET /` → Servis bilgisi + desteklenen veritabanları listesi
- `GET /status` → Operational status + tüm tool listesi
- `GET|POST|HEAD|OPTIONS /mcp` → 308 redirect → `/mcp/`
- `POST /mcp/` → MCP Streamable HTTP endpoint

### Health check örnekleri

```bash
# Health
curl http://localhost:8000/health

# Status (tool listesi)
curl http://localhost:8000/status | python3 -m json.tool

# Semantic search aktif mi?
curl -s http://localhost:8000/status | grep search_bedesten_semantic
```

### İstemciyi HTTP server'a bağlama

Claude Desktop, Cursor vb. için:
```json
{
  "mcpServers": {
    "yargi-mcp-http": {
      "url": "http://localhost:8000/mcp"
    }
  }
}
```

---

## 8. Docker ve Railway Deployment

### Docker (minimal — app.py)

```bash
cd /home/z/my-project/research/yargi-mcp
docker build -t yargi-mcp .
docker run -d \
  -p 8000:8000 \
  -e GEMINI_API_KEY=AIzaSy... \
  --name yargi-mcp \
  yargi-mcp
```

Dockerfile `app:app` kullanır (minimal Starlette + health endpoint).

### Railway (zengin — asgi_app.py)

[railway.json](railway.json) konfigürasyonu:
- Builder: NIXPACKS
- Build: `pip install -e .[asgi]`
- Start: `uvicorn asgi_app:app --host 0.0.0.0 --port $PORT`
- Healthcheck: `/health`, 30s timeout
- Restart: `ON_FAILURE`, max 3 deneme

Environment değişkenleri (Railway Variables):
```
GEMINI_API_KEY=AIzaSy...
ALLOWED_ORIGINS=https://your-frontend.com
LOG_LEVEL=info
```

### Üretim (gunicorn + uvicorn worker)

```bash
pip install -e ".[production]"
export GEMINI_API_KEY=AIzaSy...
gunicorn asgi_app:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```

---

## 9. Hukuki Yapay Zeka Sistemi Mimari Önerisi

Kullanıcının hedefi: **hukuki yapay zeka sistemi geliştirmek**. Bu repoyu bir bileşen olarak kullanarak önerilen mimari:

### Katmanlar

```
┌─────────────────────────────────────────────────────────────┐
│  1. Kullanıcı Arayüzü                                       │
│     - Next.js / React web app                               │
│     - Mobil uygulama (Flutter / React Native)               │
│     - Voice-first arayüz (whisper + TTS)                    │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  2. LLM Orkestrasyon Katmanı                                │
│     - Claude / GPT-4 / Gemini 2.0 Flash (kullanıcı sohbeti) │
│     - Tool routing: hangi MCP tool'u çağıracağına karar ver │
│     - Memory: kullanıcının önceki sorguları                 │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  3. MCP Server Katmanı (BU REPO)                            │
│     - 28 hukuki veri kaynağı tool'u                         │
│     - search_bedesten_semantic (Gemini embedding ile)       │
│     - Streamable HTTP (port 8000)                           │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  4. Veri Katmanı                                            │
│     a) Yukarıdaki MCP tool'ları (canlı veri kaynağı)        │
│     b) Kendi vektor veritabanınız (ChromaDB / Qdrant)       │
│        - Önceden indirilmiş kararlar (birikmeli)            │
│        - Hybrid search: keyword + semantic                  │
│     c) İlişkisel DB (PostgreSQL)                            │
│        - Kullanıcı hesapları                                │
│        - Sorgu geçmişi                                      │
│        - Favori kararlar                                    │
└─────────────────────────────────────────────────────────────┘
```

### Önerilen geliştirme yol haritası

**Aşama 1 (Mevcut — tamamlandı):**
- ✅ Yargı MCP kurulumu
- ✅ Gemini API key entegrasyonu
- ✅ Semantic search aktif

**Aşama 2 (1-2 hafta):**
- [ ] Kendi vektor veritabanınızı kurun (ChromaDB önerilir)
- [ ] `search_bedesten_semantic`'i genişletin: Anayasa, KİK, Rekabet için de semantic search ekleyin
- [ ] İndirilen kararları cache'leyin (Rate limit'e takılmamak için)
- [ ] LLM orkestrasyonu: Gemini 2.0 Flash ile basit sohbet arayüzü

**Aşama 3 (1 ay):**
- [ ] Next.js web arayüzü
- [ ] Kullanıcı kimlik doğrulama (Clerk / Supabase Auth)
- [ ] Sorgu geçmişi ve favoriler
- [ ] PDF yükleme + otomatik özetleme

**Aşama 4 (2-3 ay):**
- [ ] RAG (Retrieval Augmented Generation) pipeline'ı
- [ ] Hukuki metin üretimi (dilekçe, sözleşme, savunma)
- [ ] Çok dilli destek (Almanca, İngilizce — Türk hukukunu yabancılara açmak için)
- [ ] Citation graph: kararlar arası atıf ağı

### Performans optimizasyonu

Mevcut `search_bedesten_semantic` her çağrıda 100 belge için tam embedding üretiyor (in-memory). Bu maliyetli. İyileştirme önerileri:

1. **Kalıcı vector store:** ChromaDB veya Faiss ile embedding'leri disk'e kaydet
2. **Incremental update:** Yeni kararlar geldikçe embedding'leri güncelle, eskileri cache'le
3. **Hybrid search:** Keyword + semantic + metadata filtreleme (tarih, daire, esas no)
4. **Reranking:** İlk 100 sonucu Gemini embedding ile sırala, sonra Cross-Encoder ile ilk 10'u yeniden sırala

### Maliyet optimizasyonu

- **Embedding:** Gemini free tier (1500 req/gün) — başlangıç için yeterli
- **LLM (sohbet):** Gemini 2.0 Flash (ücretsiz tier 15 req/dk) veya Claude Haiku
- **Vector DB:** ChromaDB (self-host, ücretsiz) veya Qdrant Cloud (1GB ücretsiz)
- **Hosting:** Railway (free tier $5/ay kredi) veya Fly.io

---

## 10. Sorun Giderme

### Semantic search aktif olmuyor

**Semptom:** `search_bedesten_semantic` tool'u MCP istemcide görünmüyor.

**Çözüm:**
1. `.env` dosyasında `GEMINI_API_KEY` satırının yorum satırı olmadığından emin ol:
   ```bash
   grep GEMINI_API_KEY /home/z/my-project/research/yargi-mcp/.env
   ```
2. Server'ı yeniden başlat
3. Log'da şu satırı ara:
   ```bash
   grep "Semantic search enabled" /tmp/yargi_*.log
   ```
   `provider=gemini (direct Google API)` görmelisin.

### `OPENROUTER_API_KEY` placeholder bug

`.env` dosyasında `OPENROUTER_API_KEY=sk-or-v1-your_openrouter_api_key_here` gibi placeholder varsa, `is_openrouter_available()` yanlış True döner. Bu durumda provider "openrouter" seçilir ama aslında gerçek key yok.

**Çözüm:** `.env` dosyasındaki placeholder satırları `#` ile yorum satırı yapın (bu kurulumda yapıldı).

### HTTP 429 (Rate limit)

Bedesten API'sinde rate limit'e takılırsanız:
```json
{
  "error": "rate_limit_exceeded",
  "status_code": 429,
  "retry_after": 60,
  "message": "Bedesten API rate limiti. Pro sürümü için: https://yargi.betaspacestudio.com"
}
```

**Çözüm:** 60 saniye bekleyin veya sorgu sıklığını azaltın.

### MCP istemci tool'ları görmüyor

1. Server'ın çalıştığını doğrula: `curl http://localhost:8000/health`
2. Status endpoint'ini kontrol et: `curl http://localhost:8000/status | python3 -m json.tool`
3. İstemcinin config dosyasındaki path'in doğru olduğundan emin ol (mutlak path kullan)
4. İstemciyi tamamen kapatıp yeniden aç

### Gemini API hatası

- **403 Forbidden:** API key geçersiz veya bölge kısıtlı
- **429 Too Many Requests:** Free tier limit aşıldı (1500 req/gün)
- **400 Invalid taskType:** Kod hatası — `taskType` değerini kontrol et

### SSL uyarıları (httpx)

Bazı resmi TC kurumlarının SSL sertifikaları sorunlu. Repo bunu `verify=False` ile geçiştiriyor (Yargıtay client'ında). Üretimde bu güvenlik riski — gerekirse proper CA bundle kullan.

### Token optimizasyonu

Token counting middleware'i (tiktoken) opsiyonel. Yüklü değilse atlanır:
```
[WARN] tiktoken not available — token counting disabled
```

Kurmak için:
```bash
uv add tiktoken
```

---

## Ek Kaynaklar

- **Orijinal repo:** https://github.com/saidsurucu/yargi-mcp
- **Remote MCP (hosted):** https://yargimcp.surucu.dev/mcp
- **Pro sürüm (mevzuat + içtihat):** https://yargi.betaspacestudio.com
- **MCP spesifikasyonu:** https://modelcontextprotocol.io
- **FastMCP framework:** https://gofastmcp.com
- **Gemini API docs:** https://ai.google.dev/gemini-api/docs/embeddings
- **Google AI Studio (API key):** https://aistudio.google.com/app/apikey

---

## Yapılan Değişiklikler (Bu Kurulumda)

Orijinal repoya şu eklemeler yapıldı:

1. **`semantic_search/embedder.py`** — Yeni `GeminiEmbedder` sınıfı eklendi (httpx ile Google'ın REST API'sine doğrudan erişim). `get_embedder()` factory'si güncellendi (öncelik: local → orca → openrouter → gemini). `is_gemini_available()` fonksiyonu eklendi.

2. **`semantic_search/__init__.py`** — `GeminiEmbedder` ve `is_gemini_available` export ediliyor.

3. **`mcp_server_main.py`** — Semantic search log mesajına `gemini` provider durumu eklendi.

4. **`.env.example`** ve **`.env`** — "Option A3: Direct Google Gemini API" bölümü eklendi. Placeholder API key'ler yorum satırı yapıldı (false-positive bug'ı önlemek için).

5. **`/home/z/my-project/scripts/test_gemini_semantic_search.py`** — Uçtan uca test scripti yazıldı (5 test: single embed, batch embed, vector search, document processor, MCP server tool).

Tüm değişiklikler MIT lisansı altında orijinal repo ile uyumludur.
