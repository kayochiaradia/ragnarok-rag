"""FastAPI para simulador local, Meta WhatsApp Cloud API e Twilio."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Awaitable, Callable, Protocol
from urllib.parse import parse_qs
from xml.etree.ElementTree import Element, SubElement, tostring

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, PlainTextResponse, Response
from pydantic import BaseModel

from ragnarok_rag.config import CONFIG, Config
from ragnarok_rag.pipeline import get_rag

from .meta import MetaConfigurationError, parse_meta_messages, send_meta_text
from .service import ChatReply, ChatService


STATIC_DIR = Path(__file__).resolve().parent / "static"


class ReplyService(Protocol):
    def reply(self, sender_id: str, message: str) -> ChatReply: ...


MetaSender = Callable[[str, str], Awaitable[str]]


class ChatRequest(BaseModel):
    message: str
    user_id: str = "browser"


def create_app(
    *,
    service: ReplyService | None = None,
    config: Config | None = None,
    meta_sender: MetaSender | None = None,
) -> FastAPI:
    settings = config or CONFIG
    injected_service = service
    app = FastAPI(title="Ragnarok RAG", version="1.0.0")
    app.state.service = service
    app.state.seen_meta_ids = set()

    def current_service() -> ReplyService:
        if app.state.service is None:
            app.state.service = ChatService(
                get_rag(),
                max_query_chars=settings.wa_max_query_chars,
                max_output_chars=settings.wa_max_chars,
                rate_limit_per_min=settings.wa_rate_limit_per_min,
            )
        return app.state.service

    async def default_meta_sender(recipient: str, text: str) -> str:
        return await send_meta_text(
            access_token=settings.wa_access_token,
            phone_number_id=settings.wa_phone_number_id,
            api_version=settings.wa_api_version,
            recipient=recipient,
            text=text,
        )

    send_to_meta = meta_sender or default_meta_sender

    @app.get("/", include_in_schema=False)
    async def simulator() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health")
    async def health() -> dict[str, object]:
        if injected_service is not None:
            ready = True
        else:
            ready = get_rag().is_ready()
        return {"status": "ok", "ready": ready, "channel": "local"}

    @app.post("/api/chat")
    async def local_chat(request: ChatRequest) -> dict[str, object]:
        reply = current_service().reply(request.user_id, request.message)
        return asdict(reply)

    @app.get("/webhook/meta")
    async def verify_meta(request: Request) -> PlainTextResponse:
        query = request.query_params
        valid = (
            query.get("hub.mode") == "subscribe"
            and query.get("hub.verify_token") == settings.wa_verify_token
            and query.get("hub.challenge") is not None
        )
        if not valid:
            raise HTTPException(status_code=403, detail="Falha na verificação do webhook")
        return PlainTextResponse(query["hub.challenge"])

    @app.post("/webhook/meta")
    async def receive_meta(request: Request) -> dict[str, object]:
        payload = await request.json()
        messages = parse_meta_messages(payload)
        processed = 0
        duplicates = 0
        for message in messages:
            if message.message_id in app.state.seen_meta_ids:
                duplicates += 1
                continue
            reply = current_service().reply(message.sender_id, message.text)
            try:
                await send_to_meta(message.sender_id, reply.text)
            except MetaConfigurationError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            except httpx.HTTPError as exc:
                raise HTTPException(status_code=502, detail="Falha ao enviar resposta à Meta") from exc
            app.state.seen_meta_ids.add(message.message_id)
            processed += 1
        return {"received": True, "processed": processed, "duplicates": duplicates}

    @app.post("/webhook/twilio")
    async def receive_twilio(request: Request) -> Response:
        raw = (await request.body()).decode("utf-8", errors="replace")
        fields = parse_qs(raw, keep_blank_values=True)
        sender = fields.get("From", ["twilio"])[0]
        body = fields.get("Body", [""])[0]
        reply = current_service().reply(sender, body)

        root = Element("Response")
        SubElement(root, "Message").text = reply.text
        xml = tostring(root, encoding="utf-8", xml_declaration=True)
        return Response(content=xml, media_type="application/xml")

    return app


app = create_app()
