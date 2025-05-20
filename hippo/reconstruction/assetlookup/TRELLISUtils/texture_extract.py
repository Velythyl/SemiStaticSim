import io

import trimesh
from PIL import Image


def extract_textures_with_trimesh(glb_path, output_prefix="texture"):
    # Load the GLB file
    scene = trimesh.load(glb_path)

    if not isinstance(scene, trimesh.Scene):
        print("File doesn't contain a scene with materials")
        return

    # Iterate through materials
    for name, material in scene.graph.materials.items():
        # Albedo/Diffuse texture
        if hasattr(material, 'baseColorTexture'):
            img = Image.open(io.BytesIO(material.baseColorTexture))
            img.convert('RGB').save(f"{output_prefix}_albedo.jpg", quality=95)

        # Emission texture
        if hasattr(material, 'emissiveTexture'):
            img = Image.open(io.BytesIO(material.emissiveTexture))
            img.convert('RGB').save(f"{output_prefix}_emission.jpg", quality=95)

        # Normal texture
        if hasattr(material, 'normalTexture'):
            img = Image.open(io.BytesIO(material.normalTexture))
            img.save(f"{output_prefix}_normal.jpg", quality=95)


# Example usage
extract_textures_with_trimesh("model.glb", "my_model")