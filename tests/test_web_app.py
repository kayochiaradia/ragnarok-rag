from __future__ import annotations

from xml.etree import ElementTree

from fastapi.testclient import TestClient

from ragnarok_rag.config import Config
from whatsapp.app import create_app
from whatsapp.service import ChatReply

from .test_meta import META_PAYLOAD


class StubChatService:
    def reply(self, sender_id: str, message: str) -> ChatReply:
        return ChatReply(
            text=f"Resposta para {message}: A & B < C",
            ok=True,
            found=True,
            confidence=0.81,
        )


def app_config(**overrides) -> Config:
    values = {
        "wa_verify_token": "verify-secret",
        "wa_access_token": "",
        "wa_phone_number_id": "",
    }
    values.update(overrides)
    return Config(**values)


def test_root_serves_offline_simulator_and_health_is_ready() -> None:
    client = TestClient(create_app(service=StubChatService(), config=app_config()))

    root = client.get("/")
    health = client.get("/health")

    assert root.status_code == 200
    assert "Arquivo de Prontera" in root.text
    assert health.json() == {"status": "ok", "ready": True, "channel": "local"}


def test_local_chat_returns_channel_neutral_reply() -> None:
    client = TestClient(create_app(service=StubChatService(), config=app_config()))

    response = client.post("/api/chat", json={"user_id": "browser", "message": "gtb"})

    assert response.status_code == 200
    assert response.json() == {
        "text": "Resposta para gtb: A & B < C",
        "ok": True,
        "found": True,
        "confidence": 0.81,
        "error_code": None,
    }


def test_meta_verification_accepts_only_configured_token() -> None:
    client = TestClient(create_app(service=StubChatService(), config=app_config()))
    params = {
        "hub.mode": "subscribe",
        "hub.verify_token": "verify-secret",
        "hub.challenge": "challenge-123",
    }

    accepted = client.get("/webhook/meta", params=params)
    params["hub.verify_token"] = "wrong"
    rejected = client.get("/webhook/meta", params=params)

    assert accepted.status_code == 200
    assert accepted.text == "challenge-123"
    assert rejected.status_code == 403


def test_meta_status_payload_is_acknowledged_without_credentials() -> None:
    client = TestClient(create_app(service=StubChatService(), config=app_config()))
    payload = {"entry": [{"changes": [{"value": {"statuses": [{"id": "x"}]}}]}]}

    response = client.post("/webhook/meta", json=payload)

    assert response.status_code == 200
    assert response.json() == {"received": True, "processed": 0, "duplicates": 0}


def test_meta_text_is_sent_once_and_duplicate_message_is_ignored() -> None:
    sent: list[tuple[str, str]] = []

    async def sender(recipient: str, text: str) -> str:
        sent.append((recipient, text))
        return "wamid.out"

    config = app_config(wa_access_token="token", wa_phone_number_id="123")
    client = TestClient(
        create_app(service=StubChatService(), config=config, meta_sender=sender)
    )

    first = client.post("/webhook/meta", json=META_PAYLOAD)
    second = client.post("/webhook/meta", json=META_PAYLOAD)

    assert first.json() == {"received": True, "processed": 1, "duplicates": 0}
    assert second.json() == {"received": True, "processed": 0, "duplicates": 1}
    assert sent == [("5511999999999", "Resposta para o que e woe: A & B < C")]


def test_twilio_webhook_returns_valid_escaped_twiml() -> None:
    client = TestClient(create_app(service=StubChatService(), config=app_config()))

    response = client.post(
        "/webhook/twilio",
        data={"From": "whatsapp:+5511999999999", "Body": "fogo & terra"},
    )

    root = ElementTree.fromstring(response.content)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/xml")
    assert root.findtext("Message") == "Resposta para fogo & terra: A & B < C"
