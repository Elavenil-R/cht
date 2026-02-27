from pathlib import Path

def read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""

def load_knowledge_files(knowledge_dir: Path, filenames: list[str]) -> list[tuple[str, str]]:
    """
    Returns list of (source_name, text)
    """
    out: list[tuple[str, str]] = []
    for name in filenames:
        p = knowledge_dir / name
        out.append((name, read_text_file(p)))
    return out

def load_bot_rules(knowledge_dir: Path, rules_filename: str) -> str:
    return read_text_file(knowledge_dir / rules_filename).strip()