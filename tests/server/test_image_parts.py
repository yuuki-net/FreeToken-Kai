"""OpenAI ``image_url`` content parts: data: URLs are decoded and the template sees
``{"type": "image"}`` placeholders; the text-only renderer keeps rejecting them."""

from __future__ import annotations

import base64

import pytest

from freetoken.server.generation import render_messages, render_messages_multimodal

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
DATA_URL = "data:image/png;base64," + base64.b64encode(PNG).decode()


def _msg(*parts):
    return [{"role": "user", "content": list(parts)}]


def test_multimodal_render_keeps_image_parts_and_collects_bytes():
    rendered, images = render_messages_multimodal(
        _msg({"type": "text", "text": "what is this? "},
             {"type": "image_url", "image_url": {"url": DATA_URL}},
             {"type": "text", "text": " answer briefly"})
    )
    assert images == [PNG]
    assert rendered[0]["content"] == [
        {"type": "text", "text": "what is this? "},
        {"type": "image"},
        {"type": "text", "text": " answer briefly"},
    ]


def test_multimodal_render_flattens_text_only_messages_unchanged():
    rendered, images = render_messages_multimodal(
        _msg({"type": "text", "text": "a"}, {"type": "text", "text": "b"})
    )
    assert images == []
    assert rendered[0]["content"] == "ab"


def test_multimodal_render_orders_images_across_messages():
    other = "data:image/jpeg;base64," + base64.b64encode(b"second").decode()
    messages = [
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": DATA_URL}}]},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": other}}]},
    ]
    _, images = render_messages_multimodal(messages)
    assert images == [PNG, b"second"]


@pytest.mark.parametrize(
    "url",
    ["https://example.com/a.png", "data:image/png,raw-not-base64", "data:image/png;base64,", 42],
)
def test_multimodal_render_rejects_non_data_or_malformed_urls(url):
    with pytest.raises(ValueError):
        render_messages_multimodal(_msg({"type": "image_url", "image_url": {"url": url}}))


def test_multimodal_render_accepts_bare_string_image_url():
    _, images = render_messages_multimodal(_msg({"type": "image_url", "image_url": DATA_URL}))
    assert images == [PNG]


def test_text_only_renderer_still_rejects_images():
    with pytest.raises(ValueError, match="text-only"):
        render_messages(_msg({"type": "image_url", "image_url": {"url": DATA_URL}}))


def test_openai_genspec_carries_images():
    from freetoken.server.api_models import ChatCompletionRequest
    from freetoken.server.openai_api import chat_request_to_genspec

    req = ChatCompletionRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "describe"},
                    {"type": "image_url", "image_url": {"url": DATA_URL}},
                ]}
            ],
        }
    )
    spec = chat_request_to_genspec(req, {})
    assert spec.images == [PNG]
    assert spec.messages[0]["content"][1] == {"type": "image"}
