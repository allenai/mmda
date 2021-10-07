def levenshtein(s1: str, s2: str, case_sensitive: bool = True) -> int:
    """See https://en.wikipedia.org/wiki/Levenshtein_distance.

    Args:
        s1 (str): String 1 for comparison
        s2 (str): String 2 for comparison
        case_sensitive (bool): When true compare strings as-is, downcase otherwise

    Returns:
        int: The Levenshtein distance between strings
    """
    if not case_sensitive:
        return levenshtein(s1.lower(), s2.lower(), case_sensitive=True)

    if len(s1) > len(s2):
        return levenshtein(s2, s1)

    v0 = list(range(len(s2) + 1))
    v1 = [0 for _ in range(len(s2) + 1)]

    for i in range(len(s1)):
        v1[0] = i + 1

        for j in range(len(s2)):
            d_ = v0[j + 1] + 1
            i_ = v1[j] + 1

            if s1[i] == s2[j]:
                s_ = v0[j]
            else:
                s_ = v0[j] + 1

            v1[j + 1] = min([d_, i_, s_])

        v0 = v1
        v1 = [0 for _ in range(len(s2) + 1)]

    return v0[len(s2)]
