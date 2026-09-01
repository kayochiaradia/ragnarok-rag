"""Contrato mínimo da Meta WhatsApp Cloud API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


class MetaConfigurationError(RuntimeError):
    """Credenciais obrigatórias da Cloud API não foram configuradas."""


@dataclass(frozen=True)
class InboundMessage:
    message_id: str
    sender_id: str
    text: str


def parse_meta_messages(payload: dict[str, Any]) -> list[InboundMessage]:
    """Extrai mensagens de texto e ignora status e mídias não suportadas."""
    parsed: list[InboundMessage] = []
    for entry in payload.get("entry") or []:
        for change in entry.get("changes") or []:
            value = change.get("value") or {}
            for message in value.get("messages") or []:
                if message.get("type") != "text":
                    continue
                text = (message.get("text") or {}).get("body")
                message_id = message.get("id")
                sender_id = message.get("from")
                if text and message_id and sender_id:
                    parsed.append(
                        InboundMessage(
                            message_id=str(message_id),
                            sender_id=str(sender_id),
                            text=str(text),
                        )
                    )
    return parsed


async def send_meta_text(
    *,
    access_token: str,
    phone_number_id: str,
    api_version: str,
    recipient: str,
    text: str,
    client: httpx.AsyncClient | None = None,
) -> str:
    """Envia texto pela Graph API e retorna o ID da mensagem criada."""
    if not access_token or not phone_number_id or not api_version:
        raise MetaConfigurationError(
            "Configure WHATSAPP_ACCESS_TOKEN, WHATSAPP_PHONE_NUMBER_ID e "
            "WHATSAPP_API_VERSION para usar o webhook da Meta."
        )

    url = f"https://graph.facebook.com/{api_version.strip('/')}/{phone_number_id}/messages"
    headers = {"Authorization": f"Bearer {access_token}"}
    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": recipient,
        "type": "text",
        "text": {"preview_url": False, "body": text},
    }

    async def request(active_client: httpx.AsyncClient) -> str:
        response = await active_client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        messages = data.get("messages") or []
        return str(messages[0].get("id", "")) if messages else ""

    if client is not None:
        return await request(client)
    async with httpx.AsyncClient(timeout=15.0) as owned_client:
        return await request(owned_client)
