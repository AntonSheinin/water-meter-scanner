"""
    Utilities module for API    
"""

def convert_milvus_result(result) -> dict:
    """
        Convert Milvus result to JSON-serializable format
    """

    converted = {}

    for key, value in result.items():
        if hasattr(value, 'item'):  # numpy scalar
            converted[key] = value.item()

        elif hasattr(value, 'tolist'):  # numpy array
            converted[key] = value.tolist()

        elif isinstance(value, (list, tuple)):
            # Handle lists that might contain numpy objects
            converted[key] = [v.item() if hasattr(v, 'item') else v for v in value]

        else:
            converted[key] = value

    return converted
