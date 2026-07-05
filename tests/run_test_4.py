import requests
import json

BASE_URL = "http://localhost:8000"

def build_baseline():
    res = requests.post(f"{BASE_URL}/baseline/build", json={})
    return res.json()

def get_baseline():
    res = requests.get(f"{BASE_URL}/baseline")
    return res.json()

def jaccard_similarity(list1, list2):
    set1 = set([tuple(x) for x in list1]) if list1 and isinstance(list1[0], list) else set(list1) if list1 else set()
    set2 = set([tuple(x) for x in list2]) if list2 and isinstance(list2[0], list) else set(list2) if list2 else set()
    return len(set1 & set2) / max(len(set1 | set2), 1)

def dict_similarity(d1, d2):
    all_keys = set(d1.keys()) | set(d2.keys())
    score = 0
    total = 0
    for k in all_keys:
        v1 = d1.get(k, 0)
        v2 = d2.get(k, 0)
        score += min(v1, v2)
        total += max(v1, v2)
    return score / total if total else 0

def validate_baseline():
    print("\n=== PHASE 4 VALIDATION START ===\n")
    print("[1] Building baseline run A...")
    build_baseline()
    baseline_a = get_baseline()

    print("[2] Building baseline run B...")
    build_baseline()
    baseline_b = get_baseline()

    freq_a = baseline_a.get("tool_frequency", {})
    freq_b = baseline_b.get("tool_frequency", {})
    freq_score = dict_similarity(freq_a, freq_b)

    seq_a = baseline_a.get("top_sequences", [])
    seq_b = baseline_b.get("top_sequences", [])
    seq_score = jaccard_similarity(seq_a, seq_b)

    trans_a = baseline_a.get("transitions", {})
    trans_b = baseline_b.get("transitions", {})
    trans_score = dict_similarity(
        {str(k): sum(v.values()) for k, v in trans_a.items()},
        {str(k): sum(v.values()) for k, v in trans_b.items()},
    )

    len_a = baseline_a.get("avg_trace_length", 0)
    len_b = baseline_b.get("avg_trace_length", 0)
    if max(len_a, len_b) == 0:
        len_score = 0
    else:
        len_score = 1 - abs(len_a - len_b) / max(len_a, len_b)

    final_score = (freq_score * 0.35 + seq_score * 0.25 + trans_score * 0.25 + len_score * 0.15)

    print("\n=== RESULTS ===")
    print(f"Tool Frequency Similarity : {freq_score:.3f}")
    print(f"Sequence Similarity       : {seq_score:.3f}")
    print(f"Transition Similarity     : {trans_score:.3f}")
    print(f"Trace Length Stability    : {len_score:.3f}")
    print(f"\n🔥 FINAL BASELINE SCORE   : {final_score:.3f}")

    if final_score >= 0.85:
        print("✅ PASS: Baseline is stable and reliable")
    elif final_score >= 0.70:
        print("⚠️ WARNING: Baseline is moderately stable but noisy")
    else:
        print("❌ FAIL: Baseline is unstable \u2014 fix simulation or aggregation")

if __name__ == "__main__":
    validate_baseline()