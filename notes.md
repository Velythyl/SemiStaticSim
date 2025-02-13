# installation progress

this is done
```
conda create --name holodeck python=3.10
conda activate holodeck
pip install -r requirements.txt
pip install --extra-index-url https://ai2thor-pypi.allenai.org ai2thor==0+8524eadda94df0ab2dbb2ef5a577e4d37c712897
```

this is in progress

```
python -m objathor.dataset.download_holodeck_base_data --version 2023_09_23 # done
python -m objathor.dataset.download_assets --version 2023_09_23 # WIP
python -m objathor.dataset.download_annotations --version 2023_09_23    # not started
python -m objathor.dataset.download_features --version 2023_09_23       # not started
```



---

args.model = Holodeck

1. can start from prior generation (gen variants) : --original_scene