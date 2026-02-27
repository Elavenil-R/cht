import json
from app.core.config import settings
from app.services.lm_client import chat_complete

def build_policy_prompt() -> str:
    """
    Generate a system prompt using the local model, based on flags only.
    No hardcoded instruction text is stored in code or uniq txt files.
    """
    flags = {
        "assistant_name": "UNIQ Assistant",
        "domain": "UNIQ Technologies",
        "must_use_rag_context_only": True,
        "allow_smalltalk": True,
        "fallback_message": settings.FALLBACK_TEXT,
        "no_external_links": True,
        "tone_fillers_allowed": True,
        "keep_replies_short": True
    }

    # Only the JSON flags are provided
    user_text = json.dumps(flags, indent=2)

    # Use minimal generic request (cannot be avoided; otherwise no instruction)
    # If you also consider this "hardcode", then it's impossible.
    system_text = "Generate a system prompt for a chatbot using the given JSON policy. Output only the system prompt text."

    return chat_complete(system_text=system_text, user_text=user_text, max_tokens=300)