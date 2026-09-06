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

## Qwen3.8-Flash-Next

The NVFP4 checkpoint keeps the vision tower (`model.visual.*`, bf16, ~0.9 GB) and the
Qwen3-VL processor config, so the same request shape works there too, with two differences:

* **The tower runs on the CPU, in the tokenizer worker** (transformers' own
  `Qwen4ExpVisionModel`, float32, loaded on the first request with images). A 12 GB card
  has no room for it, and one image is a few seconds of CPU time. The worker hands the
  scheduler the already-projected soft tokens; nothing image-related touches the GPU beyond
  the scatter. `FT_IMAGE_MAX_PIXELS` (default `1048576`, about 1k soft tokens per image)
  bounds the resolution the processor keeps.
* **M-RoPE.** Image tokens rope at 3-D `(t, h, w)` positions and every token after them at
  `logical + delta` (`delta <= 0`), exactly as the HF model does. FreeToken keeps its logical
  positions for the QSA ring / slab / causal bookkeeping and swaps only the rope lookups: the
  image prompt's single prefill chunk ropes from a per-request cos/sin table indexed by
  logical position (`Batch.rope_cos_sin`; such a prompt is scheduled alone), later tokens
  rope at `Batch.rope_positions` (a static input of the decode graph). Text-only prompts are
  bit-for-bit the old path.

Not covered: video, the Anthropic / Responses adapters (text-only, image blocks are dropped
as before), and speculative decoding with image prompts.
