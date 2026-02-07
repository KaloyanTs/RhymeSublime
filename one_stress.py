# keep_single_stress.py
# Usage: python keep_single_stress.py words_combined.csv single_stress.csv

import sys
import pandas as pd

VOWELS_BG = set(list("аеиоуъюяѝАЕИОУЪЮЯЍ"))

def process_row(name, name_stressed):
    """Process each row to normalize stress marking."""
    if pd.isna(name_stressed):
        name_stressed = ""
    
    name = str(name)
    name_stressed = str(name_stressed)
    
    stress_count = name_stressed.count("`")
    
    if stress_count == 1:
        return name_stressed
    
    if stress_count > 1:
        first_backtick = name_stressed.find("`")
        return name_stressed[:first_backtick+1] + name_stressed[first_backtick+1:].replace("`", "")
    
    vowel_count = sum(1 for ch in name if ch in VOWELS_BG)
    if vowel_count == 1:
        for i, ch in enumerate(name):
            if ch in VOWELS_BG:
                return name[:i+1] + "`" + name[i+1:]
    
    return None

def main(inp, out):
    df = pd.read_csv(inp)

    if " name_stressed" not in df.columns:
        raise ValueError("CSV must have a 'name_stressed' column")

    results = []
    stats = {"exact_1": 0, "multi_to_1": 0, "one_vowel": 0, "skipped": 0}
    
    for _, row in df.iterrows():
        processed = process_row(row[" name"], row[" name_stressed"])
        
        if processed is not None:
            stress_count = str(row[" name_stressed"]).count("`")
            if stress_count == 1:
                stats["exact_1"] += 1
            elif stress_count > 1:
                stats["multi_to_1"] += 1
            else:
                stats["one_vowel"] += 1
            
            results.append({
                "id": row["id"],
                " name": row[" name"],
                " name_stressed": processed
            })
        else:
            stats["skipped"] += 1
    
    kept = pd.DataFrame(results)

    print(f"Rows total: {len(df):,}")
    print(f"Rows with exactly 1 stress: {stats['exact_1']:,}")
    print(f"Rows with 1 vowel, no stress (added): {stats['one_vowel']:,}")
    print(f"Rows with multiple stresses (first kept): {stats['multi_to_1']:,}")
    print(f"Total rows kept: {len(kept):,}")
    print(f"Rows skipped: {stats['skipped']:,}")
    
    kept.to_csv(out, index=False)

if __name__ == "__main__":
    main("bg_dict_csv/words.csv", "bg_dict_csv/single_stress.csv")
