import os
from glob import glob

from setuptools import find_packages, setup

package_name = "perception_models"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test", "legacy"]),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        ("share/" + package_name, ["package.xml", "README.md"]),
        (
            os.path.join("share", package_name, "dora"),
            glob("dora/*"),
        ),
        (
            os.path.join("lib", package_name),
            [
                "scripts/export_clipseg_vision_onnx.py",
                "scripts/build_clipseg_vision_engine.sh",
                "scripts/export_clipseg_film_onnx.py",
                "scripts/build_clipseg_film_engine.sh",
                "scripts/build_all_engines.sh",
                "scripts/trt_common.sh",
                "scripts/bench_clipseg.py",
                "scripts/export_nanosam_mask_decoder_batched_onnx.py",
                "scripts/build_nanosam_mask_decoder_batched_engine.sh",
                "scripts/build_nanosam_512_pair.sh",
                "scripts/build_nanosam_image_encoder_batched.sh",
                "scripts/export_yolo_world.py",
                "scripts/bench_yolo_world.py",
                "scripts/install_deps.sh",
                "scripts/dora_rgb_source",
                "scripts/dora_clipseg_encoder",
                "scripts/dora_seg_refine",
                "scripts/dora_seg_viz",
                "scripts/dora_yolo_world",
            ],
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="hound",
    maintainer_email="hound@todo.todo",
    description="CLIPSeg + NanoSAM + YOLO-World backends, Dora nodes, and TRT engines.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "dora_rgb_source = perception_models.dora_rgb_source:main",
            "dora_clipseg_encoder = perception_models.dora_clipseg_encoder:main",
            "dora_seg_refine = perception_models.dora_seg_refine:main",
            "dora_seg_viz = perception_models.dora_seg_viz:main",
            "dora_yolo_world = perception_models.dora_yolo_world:main",
        ],
    },
)
