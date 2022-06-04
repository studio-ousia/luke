from typing import Iterator, Tuple
import h5py
import numpy as np


class HyperlinkDataset:
    SLASH_ALTERNATIVE = "@@slash@@"

    def __init__(self, file_path: str, mode: str):
        self.file_path = file_path
        self.hdf = None
        self.mode = mode

    def __enter__(self):
        self.hdf = h5py.File(self.file_path, self.mode)
        return self

    def __exit__(self, *args):
        self.hdf.close()

    def get_h5py_safe_name(self, name: str) -> str:
        return name.replace("/", self.SLASH_ALTERNATIVE)

    def h5py_safe_name_to_original(self, name: str) -> str:
        return name.replace(self.SLASH_ALTERNATIVE, "/")

    def add_entity_data(self, entity_name: str, word_ids: np.ndarray, entity_position_ids: np.ndarray):
        entity_name = self.get_h5py_safe_name(entity_name)
        self.hdf.create_group(entity_name)
        self.hdf.create_dataset(f"/{entity_name}/word_ids", data=word_ids, dtype="int32")
        self.hdf.create_dataset(f"/{entity_name}/entity_position_ids", data=entity_position_ids, dtype="int16")

    def generate_entity_data(self) -> Iterator[Tuple[str, np.ndarray, np.ndarray]]:
        for entity_name in self.hdf.keys():
            word_ids = np.array(self.hdf[f"/{entity_name}/word_ids"])
            entity_position_ids = np.array(self.hdf[f"/{entity_name}/entity_position_ids"])
            entity_name = self.h5py_safe_name_to_original(entity_name)
            yield entity_name, word_ids, entity_position_ids
