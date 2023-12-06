"""

Benchmarking different ways of indexing and searching over Boxes

RTree
    INFO:root:RTreeIndexer took 0.23895001411437988 seconds to index 10000 boxes)
    INFO:root:RTreeIndexer took 188.41757702827454 seconds for 1000 queries
    
    INFO:root:RTreeIndexer took 0.24275708198547363 seconds to index 10000 boxes)
    INFO:root:RTreeIndexer took 1.1544108390808105 seconds for 1000 queries

KDTree
    Stopped Implementation was kind of tricky.

Numpy array
    INFO:root:NumpyIndexer took 0.0027608871459960938 seconds to index 10000 boxes)
    INFO:root:NumpyIndexer took 187.8270788192749 seconds for 1000 queries

    INFO:root:NumpyIndexer took 0.0034339427947998047 seconds to index 10000 boxes)
    INFO:root:NumpyIndexer took 0.07767200469970703 seconds for 1000 queries

"""

import logging
import time
from typing import List, Set, Tuple

import numpy as np
from rtree import index
from sklearn.neighbors import KDTree

from mmda.types.box import Box

logging.basicConfig(level=logging.INFO)

# create random boxes
boxes = [
    Box(
        l=np.random.rand(),
        t=np.random.rand(),
        w=np.random.rand(),
        h=np.random.rand(),
        page=0,
    )
    for _ in range(10000)
]

# create random boxes for querying
queries = [
    Box(
        l=np.random.rand(),
        t=np.random.rand(),
        w=np.random.rand(),
        h=np.random.rand(),
        page=0,
    )
    for _ in range(1000)
]


class RTreeIndexer:
    def __init__(self, boxes: List[Box]):
        self.boxes = boxes
        self.rtree = index.Index(interleaved=True)
        for i, b in enumerate(boxes):
            x1, y1, x2, y2 = b.coordinates
            self.rtree.insert(i, (x1, y1, x2, y2))

    def find(self, query: Box) -> List[int]:
        x1, y1, x2, y2 = query.coordinates
        box_ids = self.rtree.intersection((x1, y1, x2, y2))
        return list(box_ids)


class NumpyIndexer:
    def __init__(self, boxes: List[Box]):
        self.boxes = boxes
        self.np_boxes_x1 = np.array([b.l for b in boxes])
        self.np_boxes_y1 = np.array([b.t for b in boxes])
        self.np_boxes_x2 = np.array([b.l + b.w for b in boxes])
        self.np_boxes_y2 = np.array([b.t + b.h for b in boxes])

    def find(self, query: Box) -> List[int]:
        x1, y1, x2, y2 = query.coordinates
        mask = (
            (self.np_boxes_x1 <= x2)
            & (self.np_boxes_x2 >= x1)
            & (self.np_boxes_y1 <= y2)
            & (self.np_boxes_y2 >= y1)
        )
        return np.where(mask)[0].tolist()


def bulk_query(indexer, boxes, queries, is_validate: bool = True):
    for q in queries:
        found = indexer.find(q)
        if is_validate:
            for i in range(len(boxes)):
                if i in found:
                    assert boxes[i].is_overlap(q)
                else:
                    assert not boxes[i].is_overlap(q)


def benchmark_rtree(boxes, queries, is_validate: bool = True):
    # indexing time
    start = time.time()
    logging.info("Starting benchmarking")
    rtree_indexer = RTreeIndexer(boxes)
    end = time.time()
    logging.info(
        f"RTreeIndexer took {end - start} seconds to index {len(boxes)} boxes)"
    )

    # searching time
    start = time.time()
    logging.info("Starting benchmarking")
    bulk_query(rtree_indexer, boxes, queries, is_validate=is_validate)

    end = time.time()
    logging.info(f"RTreeIndexer took {end - start} seconds for {len(queries)} queries")


def benchmark_numpy(boxes, queries, is_validate: bool = True):
    # indexing time
    start = time.time()
    logging.info("Starting benchmarking")
    numpy_indexer = NumpyIndexer(boxes)
    end = time.time()
    logging.info(
        f"NumpyIndexer took {end - start} seconds to index {len(boxes)} boxes)"
    )

    # searching time
    start = time.time()
    logging.info("Starting benchmarking")
    bulk_query(numpy_indexer, boxes, queries, is_validate=is_validate)

    end = time.time()
    logging.info(f"NumpyIndexer took {end - start} seconds for {len(queries)} queries")


benchmark_rtree(boxes=boxes, queries=queries)
benchmark_numpy(boxes=boxes, queries=queries)

benchmark_rtree(boxes=boxes, queries=queries, is_validate=False)
benchmark_numpy(boxes=boxes, queries=queries, is_validate=False)
