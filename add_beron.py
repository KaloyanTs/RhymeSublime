import argparse
import csv
import json
import os
from typing import Any, Dict, Optional, Tuple


VOWELS_BG = set(
    [
        "а",
        "е",
        "и",
        "о",
        "у",
        "ъ",
        "ю",
        "я",
        "А",
        "Е",
        "И",
        "О",
        "У",
        "Ъ",
        "Ю",
        "Я",
    ]
)


def load_existing_words(csv_path: str) -> Tuple[int, set]:
    """Return (last_id, existing_names_set) from the CSV if present.

    The CSV format is: id,name,name_stressed
    """
    if not os.path.exists(csv_path):
        return 0, set()

    last_id = 0
    names = set()
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            # Skip header if detected
            if row[0].strip().lower() == "id":
                continue
            try:
                _id = int(row[0])
                last_id = max(last_id, _id)
            except Exception:
                # Non-standard line; skip id update
                pass
            if len(row) >= 2:
                names.add(row[1])
    return last_id, names


def insert_backtick_char(word: str, char_index: int) -> str:
    if char_index < 0 or char_index >= len(word):
        raise ValueError(f"char_index {char_index} out of range for '{word}'")
    return word[: char_index + 1] + "`" + word[char_index + 1 :]


def insert_backtick_syllable(word: str, syllable_index: int) -> str:
    positions = [i for i, ch in enumerate(word) if ch in VOWELS_BG]
    if not positions:
        # No vowels — place at end as fallback
        return word + "`"
    if syllable_index < 0 or syllable_index >= len(positions):
        raise ValueError(
            f"syllable_index {syllable_index} out of range for '{word}' (vowels={len(positions)})"
        )
    pos = positions[syllable_index]
    return insert_backtick_char(word, pos)


def derive_name_and_stressed(
    obj: Dict[str, Any],
    base_key: Optional[str],
    stress_key: Optional[str],
    stress_mode: str,
) -> Optional[Tuple[str, str]]:
    """Try to extract base 'name' and 'name_stressed' from a JSON object.

    - base_key: preferred field for base word (fallbacks: word/name/lemma)
    - stress_key: preferred field for stress information or stressed string
    - stress_mode: 'auto' | 'char' | 'syllable'
    """
    # Resolve base name
    candidate_keys = []
    if base_key:
        candidate_keys.append(base_key)
    candidate_keys.extend(["word", "name", "lemma"])  # common names

    name = None
    for k in candidate_keys:
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            name = v.strip()
            break
    if not name:
        return None

    # If a pre-stressed string exists, prefer it
    stressed_keys = []
    if stress_key:
        stressed_keys.append(stress_key)
    stressed_keys.extend([
        "name_stressed",
        "word_stressed",
        "stressed",
        "accented",
    ])
    for k in stressed_keys:
        v = obj.get(k)
        if isinstance(v, str) and "`" in v:
            return name, v

    # Otherwise, try index-based fields
    index_fields = [
        "stress_index",
        "accent_index",
        "stress_pos",
        "stress",
    ]
    for k in index_fields:
        v = obj.get(k)
        if isinstance(v, int):
            try:
                return name, insert_backtick_char(name, v)
            except Exception:
                pass

    # Try syllable-based fields
    syll_fields = ["stress_syllable", "accent_syllable", "syllable"]
    for k in syll_fields:
        v = obj.get(k)
        if isinstance(v, int):
            try:
                return name, insert_backtick_syllable(name, v)
            except Exception:
                pass

    # If mode is forced, attempt conversion based on provided stress_key value
    if stress_key and stress_key in obj:
        v = obj[stress_key]
        if isinstance(v, str) and "`" in v:
            return name, v
        if isinstance(v, int):
            try:
                if stress_mode == "char":
                    return name, insert_backtick_char(name, v)
                elif stress_mode == "syllable":
                    return name, insert_backtick_syllable(name, v)
            except Exception:
                pass

    # As a last resort, if we find any accent mark commonly used, convert to backtick
    # Common accents: acute (´), grave (`), apostrophe ('), double quote (")
    for k in stressed_keys:
        v = obj.get(k)
        if isinstance(v, str):
            s = v
            if "`" in s:
                return name, s
            if "´" in s or "'" in s or "ˈ" in s:
                # Normalize to backtick: insert after the accented character
                # Find first accent occurrence and replace with backtick at that position
                # Simplistic approach: remove the accent and place a backtick after preceding char
                for i, ch in enumerate(s):
                    if ch in {"´", "'", "ˈ"}:
                        base = s[:i] + s[i + 1 :]
                        # backtick after previous char (i-1)
                        idx = max(0, i - 1)
                        return name, insert_backtick_char(base, idx)

    return None


def process_jsonl(
    input_path: str,
    csv_path: str,
    base_key: Optional[str],
    stress_key: Optional[str],
    stress_mode: str,
    dedup: bool,
) -> Tuple[int, int]:
    """Process input JSONL and append to CSV. Returns (added_count, skipped_count)."""
    last_id, existing_names = load_existing_words(csv_path)
    added = 0
    skipped = 0

    # Open CSV for append (create if not exists)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    csv_file_exists = os.path.exists(csv_path)
    out_f = open(csv_path, "a", encoding="utf-8", newline="")
    writer = csv.writer(out_f)

    # If file doesn't exist, write header
    if not csv_file_exists:
        writer.writerow(["id", "name", "name_stressed"])

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                skipped += 1
                continue

            # Prefer explicit forms list if present (Beron JSONL schema)
            forms = obj.get("forms")
            processed_any = False
            if isinstance(forms, list):
                for fo in forms:
                    if not isinstance(fo, dict):
                        continue
                    name = fo.get("form")
                    si = fo.get("stress_index")

                    # Skip if missing name or stress index
                    if not isinstance(name, str) or not name.strip():
                        continue
                    # stress_index might be string; normalize
                    if si is None:
                        continue
                    if isinstance(si, str):
                        si = si.strip()
                        if si == "":
                            continue
                        try:
                            si = int(si)
                        except Exception:
                            continue
                    elif not isinstance(si, int):
                        continue

                    # Convert from 1-based to 0-based char index
                    char_idx = si - 1
                    try:
                        name_stressed = insert_backtick_char(name, char_idx)
                    except Exception:
                        skipped += 1
                        continue

                    if dedup and name in existing_names:
                        skipped += 1
                        continue

                    last_id += 1
                    if last_id % 100 == 0:
                        print(
                            f"Processing... Added: {added}, Skipped: {skipped}, Last ID: {last_id}",
                            end="\r",
                            flush=True,
                        )
                    writer.writerow([last_id, name, name_stressed])
                    added += 1
                    processed_any = True
                    if dedup:
                        existing_names.add(name)

            # Fallback path: derive from top-level fields
            if not processed_any:
                derived = derive_name_and_stressed(obj, base_key, stress_key, stress_mode)
                if not derived:
                    skipped += 1
                    continue

                name, name_stressed = derived
                if dedup and name in existing_names:
                    skipped += 1
                    continue

                last_id += 1
                if last_id % 100 == 0:
                    print(
                        f"Processing... Added: {added}, Skipped: {skipped}, Last ID: {last_id}",
                        end="\r",
                        flush=True,
                    )
                writer.writerow([last_id, name, name_stressed])
                added += 1
                if dedup:
                    existing_names.add(name)

    out_f.close()
    return added, skipped


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Append words from a Beron JSONL to single_stress.csv with stress marks."
        )
    )
    parser.add_argument(
        "--input",
        default="beron_fast_full.jsonl",
        help="Path to input JSONL file (default: beron_fast_full.jsonl)",
    )
    parser.add_argument(
        "--csv",
        default=os.path.join("bg_dict_csv", "single_stress.csv"),
        help="Path to output CSV (default: bg_dict_csv/single_stress.csv)",
    )
    parser.add_argument(
        "--base-key",
        default=None,
        help="Preferred field name for the base word in JSON objects (optional)",
    )
    parser.add_argument(
        "--stress-key",
        default=None,
        help=(
            "Preferred field name for stress info or pre-stressed string in JSON objects (optional)"
        ),
    )
    parser.add_argument(
        "--stress-mode",
        choices=["auto", "char", "syllable"],
        default="auto",
        help="How to interpret numeric stress: character index or syllable index (default: auto)",
    )
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Do not skip words already present in the CSV (default: dedup enabled)",
    )

    args = parser.parse_args()
    dedup = not args.no_dedup

    added, skipped = process_jsonl(
        input_path=args.input,
        csv_path=args.csv,
        base_key=args.base_key,
        stress_key=args.stress_key,
        stress_mode=args.stress_mode,
        dedup=dedup,
    )

    print(
        f"Done. Added: {added} rows. Skipped: {skipped} lines. Output: {args.csv}"
    )


if __name__ == "__main__":
    main()
