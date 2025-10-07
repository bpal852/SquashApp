from pathlib import Path
from datetime import datetime
import json

def save_divisions_json(divisions: dict, season: str, repo_root: Path) -> Path:
    """
    Writes to: <repo_root>/config/divisions/<season>.json
    Schema:
    {
      "season": "2025-2026",
      "generated_at": "2025-09-22T12:34:56Z",
      "divisions": [
        {"name":"2","id":473,"day":"Mon","enabled":true},
        ...
      ]
    }
    """
    out_dir = repo_root / "config" / "divisions"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{season}.json"

    payload = {
        "season": season,
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "divisions": [
            {
                "name": name,
                "id": int(meta["id"]) if meta.get("id") is not None else None,
                "day": meta.get("day"),
                "enabled": bool(meta.get("enabled", True)),
            }
            for name, meta in divisions.items()
        ],
    }

    tmp = out_path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp.replace(out_path)
    return out_path
