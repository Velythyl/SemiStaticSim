import os

from hippo.utils.subproc import run_subproc


def convert(input_folder, target_uuid, target_folder):

    retcode = run_subproc(f"mv {input_folder}/model.glb {input_folder}/{target_uuid}.glb", shell=True)
    assert retcode.success

    user_uid = os.getuid()
    retcode = run_subproc(f'docker run -v {input_folder}:/input -v {target_folder}:/output -u {user_uid}  velythyl/objathorconvert --uids={target_uuid} --glb_paths=/input/{target_uuid}.glb', shell=True)
    assert retcode.success
