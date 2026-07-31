from setuptools import find_packages, setup

package_name = "perception_models"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        ("share/" + package_name, ["package.xml", "README.md"]),
        (
            "lib/" + package_name,
            [
                "scripts/export_clipseg_vision_onnx.py",
                "scripts/build_clipseg_vision_engine.sh",
                "scripts/build_all_engines.sh",
                "scripts/bench_clipseg.py",
                "scripts/export_nanosam_mask_decoder_batched_onnx.py",
                "scripts/build_nanosam_mask_decoder_batched_engine.sh",
                "scripts/build_nanosam_512_pair.sh",
                "scripts/export_yolo_world.py",
                "scripts/bench_yolo_world.py",
            ],
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="hound",
    maintainer_email="hound@todo.todo",
    description="CLIPSeg + NanoSAM backends and TRT engines for HOUND perception.",
    license="Apache-2.0",
)
