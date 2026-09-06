"""Colour / layout probe for the image path: pure-colour squares and a two-colour image are sent
to the running server and the model's answers printed. Run on the Linux box:

    ~/FreeToken/.venv-dev/bin/python color_probe.py [http://localhost:1919] [model-name]

Correct channel order  -> red/green/blue/yellow named right.
Correct spatial layout -> "left red, right blue" for the split image (M-RoPE h/w axes).
"""

from __future__ import annotations

import base64
import io
import json
import sys
import urllib.request

from PIL import Image

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:1919"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "Qwen3.8-Flash-Next-NVFP4"


def data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def ask(img: Image.Image, question: str, max_tokens: int = 60) -> tuple[str, int]:
    body = {
        "model": MODEL,
        "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": data_url(img)}},
            {"type": "text", "text": question},
        ]}],
    }
    req = urllib.request.Request(f"{BASE}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.load(r)
    msg = out["choices"][0]["message"]
    text = (msg.get("content") or "").strip() or f"(reasoning only) {(msg.get('reasoning_content') or '')[:120]}"
    return text, out.get("usage", {}).get("prompt_tokens", -1)


def solid(rgb):
    return Image.new("RGB", (320, 320), rgb)


probes = [
    ("red    (255,0,0)", solid((255, 0, 0)), "この画像全体の色を日本語の一語で答えてください。"),
    ("green  (0,255,0)", solid((0, 255, 0)), "この画像全体の色を日本語の一語で答えてください。"),
    ("blue   (0,0,255)", solid((0, 0, 255)), "この画像全体の色を日本語の一語で答えてください。"),
    ("yellow (255,255,0)", solid((255, 255, 0)), "この画像全体の色を日本語の一語で答えてください。"),
    ("black text on white", None, "この画像の文字は何色ですか。一語で答えてください。"),
    ("left red / right blue", None, "この画像の左半分と右半分はそれぞれ何色ですか。"),
]

# black bold text on white, like a page title
txt = Image.new("RGB", (640, 160), (255, 255, 255))
try:
    from PIL import ImageDraw
    d = ImageDraw.Draw(txt)
    d.rectangle((40, 60, 600, 100), fill=(0, 0, 0))   # a thick black bar stands in for text
    d.rectangle((40, 110, 300, 125), fill=(0, 0, 0))
except Exception:  # noqa: BLE001
    pass
probes[4] = (probes[4][0], txt, probes[4][2])

split = Image.new("RGB", (640, 320), (0, 0, 255))
split.paste((255, 0, 0), (0, 0, 320, 320))
probes[5] = (probes[5][0], split, probes[5][2])

for label, img, q in probes:
    try:
        answer, ptoks = ask(img, q)
    except Exception as exc:  # noqa: BLE001
        answer, ptoks = f"ERROR {exc}", -1
    print(f"{label:24s} prompt_tokens={ptoks:5d}  ->  {answer}")
