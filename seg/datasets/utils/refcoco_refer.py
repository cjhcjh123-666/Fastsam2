"""
REFER API for loading RefCOCO/RefCOCO+/RefCOCOg/RefCLEF datasets.

This interface provides access to four datasets:
1) refclef
2) refcoco
3) refcoco+
4) refcocog
split by unc and google/umd

Adapted from OMG-Seg-main project.
"""

import itertools
import json
import os.path as osp
import pickle
import sys
import time
from collections import defaultdict

import numpy as np
from pycocotools import mask


class REFER:
    """REFER API class for loading referring expression datasets."""
    
    def __init__(self, data_root, dataset="refcoco", splitBy="unc"):
        """
        Initialize REFER API.
        
        Args:
            data_root: Root directory containing refclef, refcoco, refcoco+ and refcocog
            dataset: Dataset name ('refcoco', 'refcoco+', 'refcocog', 'refclef')
            splitBy: Split type ('unc' for refcoco/refcoco+/refclef, 'umd' or 'google' for refcocog)
        """
        print("loading dataset %s into memory..." % dataset)
        self.ROOT_DIR = osp.abspath(osp.dirname(__file__))
        self.DATA_DIR = osp.join(data_root, dataset)
        
        # Set image directory based on dataset
        # Note: IMAGE_DIR is only used as a reference, actual image loading
        # will be handled by BaseDetDataset using data_prefix
        if dataset in ["refcoco", "refcoco+", "refcocog"]:
            # Default image directory structure for RefCOCO datasets
            # The actual path will be handled by data_prefix in dataset config
            self.IMAGE_DIR = None  # Will be set by data_prefix
        elif dataset == "refclef":
            # Default image directory for RefCLEF
            self.IMAGE_DIR = None  # Will be set by data_prefix
        else:
            raise ValueError("No refer dataset is called [%s]" % dataset)

        self.dataset = dataset
        self.splitBy = splitBy

        # load refs from data/dataset/refs(splitBy).p
        tic = time.time()
        ref_file = osp.join(self.DATA_DIR, "refs(" + splitBy + ").p")
        if not osp.exists(ref_file):
            raise FileNotFoundError(f"Ref file not found: {ref_file}")
        
        print("ref_file: ", ref_file)
        self.data = {}
        self.data["dataset"] = dataset
        with open(ref_file, "rb") as f:
            self.data["refs"] = pickle.load(f)

        # load annotations from data/dataset/instances.json
        instances_file = osp.join(self.DATA_DIR, "instances.json")
        if not osp.exists(instances_file):
            raise FileNotFoundError(f"Instances file not found: {instances_file}")
        
        with open(instances_file, "r") as f:
            instances = json.load(f)
        self.data["images"] = instances["images"]
        self.data["annotations"] = instances["annotations"]
        self.data["categories"] = instances["categories"]

        # create index
        self.createIndex()
        print("DONE (t=%.2fs)" % (time.time() - tic))

    def createIndex(self):
        """Create indexing mappings for efficient data access."""
        print("creating index...")
        # fetch info from instances
        Anns, Imgs, Cats, imgToAnns = {}, {}, {}, {}
        for ann in self.data["annotations"]:
            Anns[ann["id"]] = ann
            if ann["image_id"] not in imgToAnns:
                imgToAnns[ann["image_id"]] = []
            imgToAnns[ann["image_id"]].append(ann)
        for img in self.data["images"]:
            Imgs[img["id"]] = img
        for cat in self.data["categories"]:
            Cats[cat["id"]] = cat["name"]

        # fetch info from refs
        Refs, imgToRefs, refToAnn, annToRef, catToRefs = {}, {}, {}, {}, {}
        Sents, sentToRef, sentToTokens = {}, {}, {}
        for ref in self.data["refs"]:
            # ids
            ref_id = ref["ref_id"]
            ann_id = ref["ann_id"]
            category_id = ref["category_id"]
            image_id = ref["image_id"]

            # add mapping related to ref
            Refs[ref_id] = ref
            if image_id not in imgToRefs:
                imgToRefs[image_id] = []
            imgToRefs[image_id].append(ref)
            
            if category_id not in catToRefs:
                catToRefs[category_id] = []
            catToRefs[category_id].append(ref)
            
            refToAnn[ref_id] = Anns[ann_id]
            annToRef[ann_id] = ref

            # add mapping of sent
            for sent in ref["sentences"]:
                Sents[sent["sent_id"]] = sent
                sentToRef[sent["sent_id"]] = ref
                sentToTokens[sent["sent_id"]] = sent.get("tokens", [])

        # create class members
        self.Refs = Refs
        self.Anns = Anns
        self.Imgs = Imgs
        self.Cats = Cats
        self.Sents = Sents
        self.imgToRefs = imgToRefs
        self.imgToAnns = imgToAnns
        self.refToAnn = refToAnn
        self.annToRef = annToRef
        self.catToRefs = catToRefs
        self.sentToRef = sentToRef
        self.sentToTokens = sentToTokens
        print("index created.")

    def getRefIds(self, image_ids=[], cat_ids=[], ref_ids=[], split=""):
        """Get ref ids that satisfy given filter conditions."""
        image_ids = image_ids if isinstance(image_ids, list) else [image_ids]
        cat_ids = cat_ids if isinstance(cat_ids, list) else [cat_ids]
        ref_ids = ref_ids if isinstance(ref_ids, list) else [ref_ids]

        if len(image_ids) == len(cat_ids) == len(ref_ids) == len(split) == 0:
            refs = self.data["refs"]
        else:
            if len(image_ids) > 0:
                refs = []
                for img_id in image_ids:
                    if img_id in self.imgToRefs:
                        refs.extend(self.imgToRefs[img_id])
            else:
                refs = self.data["refs"]
            
            if len(cat_ids) > 0:
                refs = [ref for ref in refs if ref["category_id"] in cat_ids]
            if len(ref_ids) > 0:
                refs = [ref for ref in refs if ref["ref_id"] in ref_ids]
            if len(split) > 0:
                if split in ["testA", "testB", "testC"]:
                    refs = [ref for ref in refs if split[-1] in ref["split"]]
                elif split in ["testAB", "testBC", "testAC"]:
                    refs = [ref for ref in refs if ref["split"] == split]
                elif split == "test":
                    refs = [ref for ref in refs if "test" in ref["split"]]
                elif split == "train" or split == "val":
                    refs = [ref for ref in refs if ref["split"] == split]
                else:
                    raise ValueError("No such split [%s]" % split)
        ref_ids = [ref["ref_id"] for ref in refs]
        return ref_ids

    def getAnnIds(self, image_ids=[], cat_ids=[], ref_ids=[]):
        """Get ann ids that satisfy given filter conditions."""
        image_ids = image_ids if isinstance(image_ids, list) else [image_ids]
        cat_ids = cat_ids if isinstance(cat_ids, list) else [cat_ids]
        ref_ids = ref_ids if isinstance(ref_ids, list) else [ref_ids]

        if len(image_ids) == len(cat_ids) == len(ref_ids) == 0:
            ann_ids = [ann["id"] for ann in self.data["annotations"]]
        else:
            if len(image_ids) > 0:
                lists = [
                    self.imgToAnns[image_id]
                    for image_id in image_ids
                    if image_id in self.imgToAnns
                ]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.data["annotations"]
            if len(cat_ids) > 0:
                anns = [ann for ann in anns if ann["category_id"] in cat_ids]
            ann_ids = [ann["id"] for ann in anns]
            if len(ref_ids) > 0:
                ids = set(ann_ids).intersection(
                    set([self.Refs[ref_id]["ann_id"] for ref_id in ref_ids])
                )
                ann_ids = list(ids)
        return ann_ids

    def getImgIds(self, ref_ids=[]):
        """Get image ids that satisfy given filter conditions."""
        ref_ids = ref_ids if isinstance(ref_ids, list) else [ref_ids]

        if len(ref_ids) > 0:
            image_ids = list(set([self.Refs[ref_id]["image_id"] for ref_id in ref_ids]))
        else:
            image_ids = list(self.Imgs.keys())
        return image_ids

    def getCatIds(self):
        """Get all category ids."""
        return list(self.Cats.keys())

    def loadRefs(self, ref_ids=[]):
        """Load refs with the specified ref ids."""
        if isinstance(ref_ids, list):
            return [self.Refs[ref_id] for ref_id in ref_ids]
        elif isinstance(ref_ids, int):
            return [self.Refs[ref_ids]]
        else:
            return []

    def loadAnns(self, ann_ids=[]):
        """Load anns with the specified ann ids."""
        if isinstance(ann_ids, list):
            return [self.Anns[ann_id] for ann_id in ann_ids]
        elif isinstance(ann_ids, int):
            return [self.Anns[ann_ids]]
        else:
            return []

    def loadImgs(self, image_ids=[]):
        """Load images with the specified image ids."""
        if isinstance(image_ids, list):
            return [self.Imgs[image_id] for image_id in image_ids]
        elif isinstance(image_ids, int):
            return [self.Imgs[image_ids]]
        else:
            return []

    def loadCats(self, cat_ids=[]):
        """Load category names with the specified category ids."""
        if isinstance(cat_ids, list):
            return [self.Cats[cat_id] for cat_id in cat_ids]
        elif isinstance(cat_ids, int):
            return [self.Cats[cat_ids]]
        else:
            return []

    def getRefBox(self, ref_id):
        """Get ref's bounding box [x, y, w, h] given the ref_id."""
        ref = self.Refs[ref_id]
        ann = self.refToAnn[ref_id]
        return ann["bbox"]  # [x, y, w, h]

    def getMask(self, ref):
        """
        Get mask and area of the referred object given ref.
        
        Returns:
            dict: {'mask': mask_array, 'area': area}
        """
        ann = self.refToAnn[ref["ref_id"]]
        image = self.Imgs[ref["image_id"]]
        
        if len(ann["segmentation"]) == 0:
            m = np.zeros((image["height"], image["width"])).astype(np.uint8)
            return {"mask": m, "area": 0}
        
        rle = None
        if isinstance(ann["segmentation"][0], list):  # polygon
            rle = mask.frPyObjects(ann["segmentation"], image["height"], image["width"])
        else:  # RLE
            rle = ann["segmentation"]
            # Ensure counts are bytes
            if isinstance(rle, list):
                for i in range(len(rle)):
                    if isinstance(rle[i], dict) and not isinstance(rle[i]["counts"], bytes):
                        rle[i]["counts"] = rle[i]["counts"].encode()
            elif isinstance(rle, dict) and not isinstance(rle["counts"], bytes):
                rle["counts"] = rle["counts"].encode()
        
        m = mask.decode(rle)
        if len(m.shape) == 3:
            m = np.sum(m, axis=2)  # sometimes there are multiple binary maps
        m = (m > 0).astype(np.uint8)  # convert to binary mask
        
        # compute area
        if isinstance(rle, list):
            area = sum([mask.area(r) for r in rle])
        else:
            area = mask.area(rle)
        
        return {"mask": m, "area": area}