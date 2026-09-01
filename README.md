# Ragnarok RAG

POC de perguntas e respostas sobre Ragnarok Online com busca híbrida, respostas extrativas e integração com WhatsApp. Não há LLM no fluxo: cada resposta vem literalmente do corpus Markdown local e inclui o arquivo-fonte.

O modo padrão roda offline, sem chave e sem download de modelo. Ele usa embeddings determinísticos por hashing, busca lexical BM25 e um vector store NumPy persistido. ChromaDB e sentence-transformers são upgrades opcionais.

## Início rápido no Windows

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[web]"
.\.venv\Scripts\python -m ragnarok_rag.cli ingest
.\.venv\Scripts\python -m ragnarok_rag.cli serve
```

Abra `http://127.0.0.1:8000`. O simulador “Arquivo de Prontera” usa a mesma camada de serviço dos webhooks.

Para consultar apenas pelo terminal:

```powershell
.\.venv\Scripts\python -m ragnarok_rag.cli ask "o que a carta GTB faz?"
.\.venv\Scripts\python -m ragnarok_rag.cli search "como pegar homúnculo" --top-k 5
.\.venv\Scripts\python -m ragnarok_rag.cli repl
```

Em Linux/macOS, troque `.\.venv\Scripts\python` por `./.venv/bin/python`.

## Como funciona

```text
corpus/*.md
   │  loader + chunker por cabeçalhos
   ▼
embedding hash ──► vector store NumPy
                         │
pergunta ──► vetor + BM25 por corpo + sinais de cabeçalho/tags
                         │
                         ▼
                 resposta extrativa + fonte
                         │
              ┌──────────┼──────────┐
           navegador     Meta     Twilio
```

Os 22 documentos cobrem classes, atributos, elementos, cartas, refino, cidades, MVPs, masmorras, leveling, habilidades, builds, WoE/PvP, economia, itens, companheiros, quests e glossário. Regras de Ragnarok variam entre Renewal, Pre-Renewal, episódios e servidores privados; trate o corpus como uma base de POC e revise o conteúdo antes de uso público.

## Comandos

| Comando | Função |
|---|---|
| `python -m ragnarok_rag.cli ingest` | Recria o índice do corpus |
| `python -m ragnarok_rag.cli ask PERGUNTA` | Resposta extrativa com fonte |
| `python -m ragnarok_rag.cli search PERGUNTA` | Ranking e sinais de debug |
| `python -m ragnarok_rag.cli stats` | Estado do índice |
| `python -m ragnarok_rag.cli repl` | Conversa no terminal |
| `python -m ragnarok_rag.cli serve` | API, simulador e webhooks |

O executável `ragnarok-rag` oferece os mesmos comandos após a instalação editável.

## Endpoints HTTP

- `GET /` — simulador local.
- `GET /health` — disponibilidade do processo e índice.
- `POST /api/chat` — JSON `{"user_id":"demo","message":"o que é WoE?"}`.
- `GET /webhook/meta` — handshake da Meta.
- `POST /webhook/meta` — mensagens de texto da WhatsApp Cloud API.
- `POST /webhook/twilio` — formulário inbound da Twilio; responde TwiML.
- `GET /docs` — OpenAPI interativo do FastAPI.

## Meta WhatsApp Cloud API

1. Copie os nomes de [.env.example](.env.example) para variáveis do processo. O projeto não carrega `.env` automaticamente.
2. Defina `WHATSAPP_VERIFY_TOKEN`, `WHATSAPP_ACCESS_TOKEN`, `WHATSAPP_PHONE_NUMBER_ID` e uma `WHATSAPP_API_VERSION` atualmente suportada.
3. Exponha o servidor HTTPS e configure a callback como `https://SEU_HOST/webhook/meta`.
4. Use o mesmo valor de `WHATSAPP_VERIFY_TOKEN` no painel da Meta.

Exemplo em PowerShell:

```powershell
$env:WHATSAPP_VERIFY_TOKEN="um-token-secreto"
$env:WHATSAPP_ACCESS_TOKEN="token-da-meta"
$env:WHATSAPP_PHONE_NUMBER_ID="123456789"
$env:WHATSAPP_API_VERSION="versao-do-painel"
.\.venv\Scripts\python -m ragnarok_rag.cli serve --host 0.0.0.0
```

A Graph API é versionada e aposenta versões; copie a versão exibida na documentação/painel atual da [WhatsApp Cloud API](https://developers.facebook.com/docs/whatsapp/cloud-api/).

## Twilio WhatsApp

No Sandbox ou remetente habilitado para WhatsApp, configure o webhook de mensagem recebida como `https://SEU_HOST/webhook/twilio`, método `POST`. A Twilio envia `application/x-www-form-urlencoded` com `From` e `Body`; o POC devolve a resposta em TwiML, conforme a [documentação oficial de webhooks](https://www.twilio.com/docs/messaging/guides/webhook-request).

Este é um POC local. Antes de expor em produção, valide a assinatura `X-Twilio-Signature` com o SDK oficial, valide a assinatura da Meta, use armazenamento compartilhado para deduplicação/rate limit e coloque o serviço atrás de HTTPS e observabilidade.

## Backends opcionais

```powershell
.\.venv\Scripts\python -m pip install -e ".[quality]"
$env:RAG_EMBEDDING_BACKEND="local"
$env:RAG_STORE_BACKEND="chroma"
.\.venv\Scripts\python -m ragnarok_rag.cli ingest
```

O modelo multilingual de sentence-transformers é baixado no primeiro uso. Depois disso, a consulta pode rodar offline.

## Testes e avaliação

```powershell
.\.venv\Scripts\python -m pip install -e ".[web,dev]"
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\python -m scripts.evaluate
```

A avaliação força hash + NumPy e verifica um conjunto fixo de perguntas, incluindo WoE, homúnculo, tabela elemental, refino, GTB, Baphomet, zeny e Asura.

## Configuração

Todas as opções têm origem em variáveis de ambiente; consulte [.env.example](.env.example). As principais são `RAG_CORPUS_DIR`, `RAG_INDEX_DIR`, `RAG_EMBEDDING_BACKEND`, `RAG_STORE_BACKEND`, `RAG_TOP_K`, `RAG_MIN_SCORE`, `RAG_WEB_HOST` e `RAG_WEB_PORT`.
