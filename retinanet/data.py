from pathlib import Path
from xml.etree import ElementTree as ET
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset
import torch

class RaccoonDataset(Dataset):
    def __init__(self, root_dir: Path, image_size: tuple[int, int]=(224, 224)):
        self.root_dir = root_dir
        self.image_dir = root_dir / "images"
        self.annotation_dir = root_dir / "annotations"

        self.image_files = list(self.image_dir.glob("*.jpg"))

        self.target_image_size = image_size
        # Scaling to fixed size is required for batching, unless using an
        # advanced collate function that can handle variable size images.
        self.transform = T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.class2idx = {"raccoon": 1} # 1-based indexing, 0 is background
        self.idx2class = {v: k for k, v in self.class2idx.items()}

    def __len__(self):
        return len(self.image_files)
    
    def _parse_xml_annotation(self, annotation_path: Path):
        """ Parse a PASCAL-VOC style XML file. Extracting bounding boxes and 
        labels.

        Args:
            annotation_path (Path): Path to the XML annotation file.
        Returns: (boxes, labels): A tuple of bounding boxes and labels.
            boxes (Tensor): Bounding boxes in (xmin, ymin, xmax, ymax) format.
            labels (Tensor): Corresponding labels for the bounding boxes.
        """
        tree = ET.parse(annotation_path)
        root = tree.getroot()

        boxes = []
        labels = []

        for obj in root.findall("object"):
            label = obj.find("name").text
            if label not in self.class2idx:
                continue
            
            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class2idx[label])
        
        if not boxes:
            return (
                torch.empty((0, 4), dtype=torch.float32), 
                torch.empty((0,), dtype=torch.long)
            )

        return (
            torch.tensor(boxes, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.long)
        )
    
    def __getitem__(self, index: int):
        image_path = self.image_files[index]
        annotation_path = self.annotation_dir / f"{image_path.stem}.xml"

        image = Image.open(image_path).convert("RGB")
        image_w, image_h = image.size
        
        if not annotation_path.exists():
            # Raccoon dataset has all images annotated, but just in case
            # as best practice.
            boxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.long)
        else:
            boxes, labels = self._parse_xml_annotation(annotation_path)
        
        # Scale boxes to match resized image, if any boxes exist
        if boxes.numel() > 0:
            scale_x = self.target_image_size[0] / image_w
            scale_y = self.target_image_size[1] / image_h
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale_x
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale_y

        img = self.transform(image)
        return img, {"boxes": boxes, "labels": labels}
    

def collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    # Note: in this implementation, to keep the code simple,
    # all images are resized to the same size in the dataset class,
    # so we can stack them directly.
    # In a real application, that goes beyond the scope of this project,
    # you would want to implement padding to the largest image in the batch,
    # or use a more advanced collate function that can handle variable size 
    # images, so that the model can be trained on images of different sizes.

    return torch.stack(images, dim=0), targets