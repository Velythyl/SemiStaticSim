import hashlib
import json

from tqdm import tqdm, trange
import prior
from prior.utils.types import LazyJsonDataset

def get_procthor10k():
    d = prior.load_dataset("procthor-10k")
    d = LazyJsonDataset(d.train.data+d.test.data+d.val.data, "traintestval", "train")

    #def get_sha_for_index(i):
    #    return bytes_to_unique_str(d.data[i])

    #d.get_sha_for_index = get_sha_for_index
    return d

def get_uuid(id_with_vbar):
    return id_with_vbar.split("|")[0].lower().strip()
    import hashlib
    import uuid

    m = hashlib.md5()
    m.update(id_with_vbar.encode('utf-8'))
    new_uuid = uuid.UUID(m.hexdigest(), version=4)
    new_uuid = new_uuid.hex.replace("-", "")
    return id_with_vbar.split("|")[0].lower() + f"-{new_uuid[:4]}-{new_uuid[4:8]}"

def get_objects_list_single_p10k(scene):
    objs = []
    for obj in scene["objects"]:
        o = {
            "id": get_uuid(obj["id"]),
        }

        if "children" in obj:
            children = obj["children"]
            children = [get_uuid(x["id"]) for x in children]

            o["children"] = children


        objs.append(o)
    return objs

#def bytes_to_unique_str(data: bytes) -> str:
#    # Compute SHA-256 hash of the bytes
#    hash_obj = hashlib.sha256(data)
#    return hash_obj.hexdigest()  # Returns a hex string

def get_objects_list_p10k(d, num_todo=100):
    objlist = {}
    for i in trange(len(d)):
        objlist[i] = {"scene_objects" :get_objects_list_single_p10k(d[i])}

    with open("objlist.json", "w") as f:
        f.write(json.dumps(objlist, indent=2))

if __name__ == "__main__":
    d = get_procthor10k()
    get_objects_list_p10k(d)

    exit()

    objlist = []
    for i in tqdm(d):
        get_objects_list_single_p10k(i)
    exit()
    objlists = list()
    print(get_objects_list_single_p10k(d[0]))

    i=0