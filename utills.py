def move_to(obj, device):
    """
    Based on the type move python object
    :param obj: the python object to move to a device, or to move its contents to a device
    :param device: the compute device to move objects to
    :return: python obj
    """
    if isinstance(obj, list):
        return [move_to(x, device) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to(list(obj), device))
    elif isinstance(obj, set):
        return set(move_to(list(obj), device))
    elif isinstance(obj, dict):
        to_ret = dict()
        for key, value in obj.items():
            to_ret[move_to(key, device)] = move_to(value, device)
        return to_ret
    elif hasattr(obj, "to"):
        return obj.to(device)
    else:
        return obj
