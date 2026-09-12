import mimetypes
from google.genai import types

def part_from_image_path(path):
    with open(path, "rb") as f:
        data = f.read()
    mime = mimetypes.guess_type(path)[0] or "image/png"
    return types.Part.from_bytes(data=data, mime_type=mime)


async def run_multimodal(runner, session_id, image_paths, prompt):
    parts = [types.Part.from_text(text=prompt)]
    for img in image_paths:
        parts.append(part_from_image_path(img))

    msg = types.Content(role="user", parts=parts)

    last_event = None
    async for event in runner.run_async(
        user_id="eval_user",
        session_id=session_id,
        new_message=msg,
    ):
        last_event = event
    return last_event


#gestione sssessione adk
async def ensure_session(runner, app_name, session_id):
    await runner.session_service.create_session(
        app_name=app_name,
        user_id="eval_user",
        session_id=session_id,
    )

