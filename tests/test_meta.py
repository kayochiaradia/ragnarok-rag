from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from whatsapp.meta import MetaConfigurationError, parse_meta_messages, send_meta_text


META_PAYLOAD = {
    "object": "whatsapp_business_account",
    "entry": [
        {
            "changes": [
                {
                    "field": "messages",
                    "value": {
                        "messages": [
                            {
                                "from": "5511999999999",
                                "id": "wamid.1",
                                "timestamp": "1788240000",
                                "type": "text",
                                "text": {"body": "o que e woe"},
                            },
                            {
                                "from": "5511999999999",
                                "id": "wamid.2",
                                "type": "image",
                                "image": {"id": "media.1"},
                            },
                        ]
                    },
                }
            ]
        }
    ],
}


def test_parse_meta_messages_extracts_only_text_messages() -> None:
    messages = parse_meta_messages(META_PAYLOAD)

    assert len(messages) == 1
    assert messages[0].message_id == "wamid.1"
    assert messages[0].sender_id == "5511999999999"
    assert messages[0].text == "o que e woe"


def test_parse_meta_messages_ignores_delivery_status_payload() -> None:
    payload = {"entry": [{"changes": [{"value": {"statuses": [{"id": "x"}]}}]}]}

    assert parse_meta_messages(payload) == []


def test_send_meta_text_rejects_missing_live_credentials() -> None:
    with pytest.raises(MetaConfigurationError):
        asyncio.run(
            send_meta_text(
                access_token="",
                phone_number_id="",
                api_version="v21.0",
                recipient="5511999999999",
                text="resposta",
            )
        )


def test_send_meta_text_emits_graph_api_contract() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert str(request.url) == "https://graph.facebook.com/v21.0/123/messages"
        assert request.headers["authorization"] == "Bearer token"
        assert json.loads(request.content) == {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": "5511999999999",
            "type": "text",
            "text": {"preview_url": False, "body": "resposta"},
        }
        return httpx.Response(200, json={"messages": [{"id": "wamid.out"}]})

    async def scenario() -> str:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await send_meta_text(
                access_token="token",
                phone_number_id="123",
                api_version="v21.0",
                recipient="5511999999999",
                text="resposta",
                client=client,
            )

    assert asyncio.run(scenario()) == "wamid.out"
