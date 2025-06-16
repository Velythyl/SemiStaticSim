import os
import subprocess
import uuid

from hippo.utils.subproc import run_subproc


def convert(input_folder, target_uuid, target_folder):

    retcode = run_subproc(f"mv {input_folder}/model.glb {input_folder}/{target_uuid}.glb", shell=True)
    assert retcode.success

    user_uid = os.getuid()
    try:
        CONTAINER_NAME = f"{uuid.uuid4().hex[:8]}"
        def timeout_cleanup():
            print("Conversion timed out! Make sure the timeout was long enough!")
            run_subproc(f'podman kill $(docker ps -q --filter "name={CONTAINER_NAME}")', shell=True)

        run_subproc(f'podman pull velythyl/objathorconvert',
                              shell=True,
                              timeout=60 * 3,
                              raise_timeout_exception=True,
                              timeout_cleanup_func=timeout_cleanup
                              )

        retcode = run_subproc(f'podman run --name {CONTAINER_NAME} -v {input_folder}:/input -v {target_folder}:/output -u {user_uid}  velythyl/objathorconvert --uids={target_uuid} --glb_paths=/input/{target_uuid}.glb',
                              shell=True,
                              timeout=60 * 3,
                              raise_timeout_exception=True,
                              timeout_cleanup_func=timeout_cleanup
                              )
        assert retcode.success
        return True
    except subprocess.TimeoutExpired:
        return False
