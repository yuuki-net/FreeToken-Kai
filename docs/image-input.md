# Image input over the API

`/v1/chat/completions` accepts OpenAI `image_url` content parts when the served checkpoint
is multimodal and its vision tower is loaded (`FREETOKEN_LOAD_VISION=1`; Gemma 4 today).

```json
{"role": "user", "content": [
  {"type": "text", "text": "What is in this picture?"},
  {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0..."}}
]}
```

* Only `data:` URLs are accepted. The server never fetches a remote URL for a client.
* Image bytes ride the request to the tokenizer worker, which renders the prompt with the
  checkpoint's HF **processor** template (it places one image token per part) and runs the
  processor: placeholder expansion plus `pixel_values` / `image_position_ids`.
* Those tensors reach the scheduler in `UserMsg.mm_inputs`; on admission it runs
  `model.encode_images` and attaches the soft-token embeddings (`mm_embeds`), which the model
  scatters at the placeholder positions during prefill.
* Limits inherited from the offline path: a prompt with images must fit one prefill chunk
  (`--max-extend-tokens`), and multimodal requests bypass the shared prefix cache.
* A text-only checkpoint, a missing vision tower, an undecodable file, or a placeholder /
  feature count mismatch fails **that request** with a 400-class error; text-only requests
  are unaffected.

The Anthropic and Responses adapters remain text-only (image blocks are dropped as before).
