from SMARTLLM.smartllm.utils.get_controller import get_sim

get_sim


def parse_floorplan(args_floor_plan, args_custom_task=None):
    scene = resolve_scene_id(args_floor_plan)

    if not scene.endswith(".json"):
        task_path = f"./data/{args.test_set}/{scene}.json"
        return parse_ai2thor_plan(scene, task_path)

    # custom task is of format:
    # "<text description of the task 1>", "<text description of the task 2>,... [[<int id of robot>, <int id of robot>, ...], [<int for robot>, <int id of robot>, ...]]

    if args_custom_task is None:
        task_path = args_floor_plan.split(".json")[0] + "_TASK.json"
        with open(task_path, "r") as f:
            task_descriptions = f.read()
    else:
        assert isinstance(args_custom_task, str)
        task_descriptions = args_custom_task

    def parse_task_string(input_string):
        # Pattern to match the tasks and robot assignments
        pattern = r'^(.*?)\s*\[\[?([\d,\s]*)\]\]?$'
        match = re.match(pattern, input_string)

        if not match:
            raise ValueError("Invalid input format")

        task_descriptions_part = match.group(1).strip()
        robots_part = match.group(2)

        # Split task descriptions (handling quoted strings if needed)
        # This simple split works for your examples, but may need adjustment for more complex cases
        task_descriptions = [desc.strip() for desc in task_descriptions_part.split(',')]
        task_descriptions = [desc.strip('"').strip("'").strip() for desc in task_descriptions if desc.strip()]

        # Parse robots assignments
        robots = []
        # Check if it's a simple case like "[1,2]"
        if '[' not in robots_part and ']' not in robots_part:
            robot_ids = [int(id_str.strip()) for id_str in robots_part.split(',') if id_str.strip()]
            robots.append(robot_ids)
        else:
            # Handle nested lists like "[[1,2], [3,4]]"
            robot_groups = re.findall(r'\[([\d,\s]*)\]', robots_part)
            for group in robot_groups:
                robot_ids = [int(id_str.strip()) for id_str in group.split(',') if id_str.strip()]
                robots.append(robot_ids)

        # Special case: if we have one task but multiple robot assignments (unlikely but possible)
        if len(task_descriptions) == 1 and len(robots) > 1:
            # We'll assume each robot assignment is for the single task
            pass
        elif len(task_descriptions) != len(robots):
            raise ValueError(
                f"Mismatch between number of tasks ({len(task_descriptions)}) and robot assignments ({len(robots)})")

        return task_descriptions, robots

    tasks, robots_for_tasks = parse_task_string(task_descriptions)

    available_robots = []
    for robots_list in robots_for_tasks:
        task_robots = []
        for i, r_id in enumerate(robots_list):
            rob = robots.robots[r_id - 1]
            # rename the robot
            rob['name'] = 'robot' + str(i + 1)
            task_robots.append(rob)
        available_robots.append(task_robots)

    return SceneTask(test_tasks=tasks, robots_test_tasks=robots_for_tasks, gt_test_tasks=None, trans_cnt_tasks=None,
                     max_trans_cnt_tasks=None, scene=scene, available_robots=available_robots, scene_name=args_floor_plan)