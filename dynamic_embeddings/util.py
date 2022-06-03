SLASH_ALTERNATIVE = "@@slash@@"


def get_h5py_safe_name(name: str) -> str:
    return name.replace("/", SLASH_ALTERNATIVE)


def h5py_safe_name_to_original(name: str) -> str:
    return name.replace(SLASH_ALTERNATIVE, "-")
