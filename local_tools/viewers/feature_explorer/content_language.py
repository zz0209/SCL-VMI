def translated_fields(source, translated):
    return [[source[key], value] for key, value in translated.items()]


def finding_translations(source, translated, collections):
    pairs = []
    for key in collections:
        assert len(source[key]) == len(translated[key]), f"Translation count differs: {key}"
        for original, english in zip(source[key], translated[key], strict=True):
            pairs.extend(translated_fields(original, english))
    return pairs
