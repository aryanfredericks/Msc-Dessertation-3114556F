def find_occurrences(doc_text: str, surface: str) -> list[tuple[int, int]]:
    out, L, start = [], len(surface), 0
    if not surface:
        return out
    has_special = any(c in surface for c in "()[].,;")
    doc_lower = doc_text.lower()
    surface_lower = surface.lower()
    while True:
        i = doc_lower.find(surface_lower, start)
        if i == -1:
            break
        j = i + L
        if has_special:
            out.append((i, j))
        else:
            left_ok = i == 0 or not doc_text[i - 1].isalnum()
            right_ok = j == len(doc_text) or not doc_text[j].isalnum()
            if left_ok and right_ok:
                out.append((i, j))
        start = i + 1
    return out