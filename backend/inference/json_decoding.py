"""JSON decoding with duplicate fields rejected before business validation."""

import json


def load_unique_json(text, *, duplicate_message="duplicate_json_key"):
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(duplicate_message)
            result[key] = value
        return result

    return json.loads(text, object_pairs_hook=unique_object)
