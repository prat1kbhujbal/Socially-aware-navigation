import tarfile
import shutil
from glob import glob
from pathlib import Path
from wget import download
from json import load, dump


class dataset:

    def __init__(self):
        self.image_url = (
            "https://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1.tar.gz"
        )
        self.allow_all = False
        self.allowed_categories = ["bicycling", "running", "walking"]
        # r hip, l hip, head, r shoulder, l shoulder
        self.allow_joints = [2, 3, 9, 12, 13]
        self.data_dir = Path(__file__).parent.parent / "pose_data"

    def get_dataset(self):
        # Check if we already have the dataset downloaded
        if len(glob("./pose_data/images/*.jpg")) == 0:
            print("Dataset not found locally, pulling it...")
            pose_data = self.data_dir
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
            download(self.image_url, temp_path)
            tar = tarfile.open(temp_path)
            tar.extractall(images)

    def filter_images(self):
        p = (Path(self.data_dir) / "annotations.json").resolve()
        annotations = load(p.open())
        valid_imgs = []
        for i in range(len(annotations["act"])):
            if self.allow_all or annotations["act"][i]["cat_name"] in self.allowed_categories:
                valid_imgs.append(i)
        print(f"Indexed {len(valid_imgs)} images with valid categories")

        # Parse it to only images with a single person visible
        annorect_imgs = []

        for i in range(len(valid_imgs)):
            i_img = valid_imgs[i]
            annolist = annotations["annolist"][i_img]
            annorects = annolist["annorect"]
            if isinstance(annorects, dict):  # Only one annorect (one person)
                if "annopoints" in annorects.keys():
                    point = annorects["annopoints"]["point"]
                    # Check that every joint we care about is annotated
                    all_present = True
                    for j in self.allow_joints:
                        all_present &= j in [p["id"] for p in point]
                    if all_present:
                        joints = []
                        for j in self.allow_joints:
                            joints.append([[p["x"], p["y"]]
                                          for p in point if p["id"] == j][0])
                        annorect_imgs.append({"id": i_img, "joints": joints})

        print(f"Indexed {len(annorect_imgs)} images with valid annotations")

        # Copy those images to the valid dir

        annotation_output = []

        for idx, i in enumerate(annorect_imgs):
            img_name = annotations["annolist"][i["id"]]["image"]["name"]
            joints = i["joints"]
            annotation_output.append(joints)
            shutil.copy(
                self.data_dir +
                "/images/" +
                img_name,
                self.data_dir +
                "/images/single/" +
                f"{idx}.jpg")

        dump(
            annotation_output,
            Path(
                self.data_dir +
                "/images/single/annotations.json").open("w"))
