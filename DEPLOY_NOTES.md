# YargıAI — Yargı MCP Server Deploy

Bu, saidsurucu/yargi-mcp'nin YargıAI projesi için host edilen instance'ıdır.

## Deploy Hedefi
- **Platform**: Koyeb (ücretsiz tier, Docker destekli)
- **Region**: Frankfurt (eu-central-1)
- **URL**: yargi-mcp-<account>.koyeb.app

## Endpoints
- `/mcp/` — MCP server (Streamable HTTP)
- `/health` — Sağlık kontrolü
- `/docs` — FastAPI Swagger UI
- `/openapi.json` — OpenAPI spec

## 13 Hukuk Kaynağı
1. Yargıtay kararları
2. Danıştay kararları
3. Anayasa Mahkemesi (norm denetimi + bireysel başvuru)
4. Emsal kararlar
5. Uyuşmazlık Mahkemesi
6. BDDK kararları
7. BTK kararları
8. KVKK kararları
9. Rekabet Kurumu kararları
10. Sayıştay kararları
11. GİB özelgeleri
12. KİK (Kamu İhale Kurulu) kararları
13. Bedesten (hukuki veritabanı)
14. Sigorta Tahkim Komisyonu

## License
MIT — original work by Said Surucu (https://github.com/saidsurucu/yargi-mcp)
