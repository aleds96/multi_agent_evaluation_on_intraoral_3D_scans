# llm_utils.py

import mimetypes
from google.genai import types

#image come part (nb per dipendenze usiamo adk più vecchio)
def part_from_image_path(path):
    """Converte un file immagine in Part inline_data (ADK 2.8.0)."""
    with open(path, "rb") as f:
        data = f.read()
    mime = mimetypes.guess_type(path)[0] or "image/png"
    return types.Part.from_bytes(data=data, mime_type=mime)

#Invia prompt + una o più immagini a Gemini multimodale via ADK.
async def run_multimodal(runner, session_id, image_paths, prompt):
    parts = [types.Part.from_text(text=prompt)]
    for img in image_paths:
        parts.append(part_from_image_path(img))

    msg = types.Content(role="user", parts=parts)

    last = None
    async for event in runner.run_async(
        user_id="eval_user",
        session_id=session_id,
        new_message=msg,
    ):
        last = event

    return last.content.parts[0].text

#gestione sssessione adk
async def ensure_session(runner, app_name, session_id):
    await runner.session_service.create_session(
        app_name=app_name,
        user_id="eval_user",
        session_id=session_id,
    )
def parse_quality_output(text):
    quality = None
    motivation = ""

    for line in text.splitlines():
        low = line.lower()
        if low.startswith("quality"):
            try:
                quality = int(line.split(":")[1].strip())
            except:
                pass
        if low.startswith("motivation"):
            motivation = line.split(":", 1)[1].strip()

    return quality, motivation
