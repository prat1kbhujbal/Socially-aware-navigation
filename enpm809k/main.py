import tarfile
from pathlib import Path
from glob import glob
import click
from wget import download


IMAGE_URL = (
    "https://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1.tar.gz"
)


def get_dataset():
    # Check if we already have the dataset downloaded
    if len(glob("./pose_data/images/*.jpg")) == 0:
        print("Dataset not found locally, pulling it...")
        pose_data = Path(__file__).parent.parent / "pose_data"
        images = pose_data / "images"
        if not pose_data.exists():
            pose_data.mkdir()
            if not images.exists():
                images.mkdir()
        temp_path = (
            Path(
                __file__,
            ).parent.parent
            / "temp.tar.gz"
        ).as_posix()
        download(IMAGE_URL, temp_path)
        tar = tarfile.open(temp_path)
        tar.extractall(images)


@click.command
def main():
    # Ensure dataset is available
    get_dataset()


if __name__ == "__main__":
    main()
