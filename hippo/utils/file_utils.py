import os
import uuid


def get_temp_filename(prefix="hippo",suffix="",ext=".tmp"):
    ret = f"{prefix}{uuid.uuid4()}{suffix}"
    if ext == "":
        return ret
    if ext.startswith("."):
        ext = ext[1:]
    return f"{ret}.{ext}"

def get_tmp_file(folder="/tmp", prefix="hippo", suffix="", ext=".tmp"):
    return f"{folder}/{get_temp_filename(prefix,suffix,ext)}"

def get_tmp_folder():
    folder_name = get_temp_filename(ext="")
    folder_path = f"/tmp/{folder_name}"
    os.makedirs(folder_path)
    return folder_path