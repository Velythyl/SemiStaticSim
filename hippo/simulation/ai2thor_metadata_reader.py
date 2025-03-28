
def get_robot_inventory(controller, agent_id):
    inventory = controller.last_event.events[agent_id].metadata["inventoryObjects"]
    inventory = [o["objectId"] for o in inventory]

    return inventory

def get_object_list_from_controller(controller):
    objects = controller.last_event.metadata["objects"]

    held_objects = {} # id: robot
    for i, robot_event in enumerate(controller.last_event.events):
        robot_metadata = robot_event.metadata

        inventory = robot_metadata["inventoryObjects"]

        robot_dict = {
            "assetId": f"", # makes robot not considered as runtime object
            "objectId": f"robot{i+1}",
            "position": robot_metadata["agent"]['position'],
            "rotation": robot_metadata["agent"]['rotation'],
            "inventory": inventory
        }
        for obj in inventory:
            held_objects[obj["objectId"]] = f"robot{i+1}"


        objects.append(robot_dict)

    for obj_dict in objects:
        if obj_dict["objectId"] in held_objects:
            obj_dict["heldBy"] = held_objects[obj_dict["objectId"]]
        else:
            obj_dict["heldBy"] = None

    return objects
