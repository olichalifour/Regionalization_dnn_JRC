

def make_json_serializable(history):
    """
    Convert all values in history dict to Python native types.
    """
    serializable = {}
    for key, values in history.items():
        # Convert each item in the list to float
        serializable[key] = [float(v) for v in values]
    return serializable