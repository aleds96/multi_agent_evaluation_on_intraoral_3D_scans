import mimetypes
from google.genai import types
import asyncio
import random
def part_from_image_path(path):
    with open(path, "rb") as f:
        data = f.read()
    mime = mimetypes.guess_type(path)[0] or "image/png"
    return types.Part.from_bytes(data=data, mime_type=mime)

def build_multimodal_prompt(
    text,
    image_paths,
):

    parts = [
        types.Part.from_text(text=text)
    ]

    for img in image_paths:
        parts.append(
            part_from_image_path(img)
        )
    return types.Content(
        role="user",
        parts=parts,
    )
async def run_multimodal(runner, session_id, image_paths, prompt):

    msg =build_multimodal_prompt(text=prompt,image_paths=image_paths)

    last_event = None
    async for event in runner.run_async(
        user_id="eval_user",
        session_id=session_id,
        new_message=msg,
    ):
        last_event = event
    if last_event is None:
        raise RuntimeError("No event returned from workflow.")
    return last_event


#gestione sssessione adk
async def ensure_session(runner, app_name, session_id):
    await runner.session_service.create_session(
        app_name=app_name,
        user_id="eval_user",
        session_id=session_id,
    )


async def safe_run_multimodal(runner, session_id, image_paths, prompt, 
                              max_retries=5, base_delay=5):
    for attempt in range(max_retries):
        try:
            return await run_multimodal(runner, session_id, image_paths, prompt)

        except Exception as e:
            msg = str(e).lower()

            if "resource_exhausted" not in msg and "429" not in msg:
                raise

            delay = base_delay * (2 ** attempt) + random.uniform(0, 2)
            print(f"[WARN] 429 RESOURCE_EXHAUSTED → retry {attempt+1}/{max_retries} in {delay:.1f}s")
            await asyncio.sleep(delay)

    raise RuntimeError("Max retries exceeded for run_multimodal")
