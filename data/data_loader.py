import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm


class FlowDataset(Dataset):
    def __init__(self, smile_dataset, preprocessor):
        self.smile_dataset = smile_dataset
        self.preprocessor = preprocessor

        if hasattr(smile_dataset, "descriptors") and smile_dataset.descriptors:
            self.preprocessor.descriptor_keys = smile_dataset.descriptors
            self.preprocessor.num_descriptors = len(smile_dataset.descriptors)

        self._process_molecules()

    def _process_molecules(self):        
        molecules = self.smile_dataset.data["mol"].values
        smiles_data = self.smile_dataset.data[self.smile_dataset.smile_col].values
        
        N = len(molecules)
        sample_atom, sample_adj, sample_desc = self.preprocessor.preprocess(molecules[0])
        
        max_atoms = sample_atom.shape[0]
        num_atom_features = sample_atom.shape[1]
        num_bond_types = sample_adj.shape[0]
        num_descriptors = sample_desc.shape[0]
              
        self.atom_matrices = np.zeros((N, max_atoms, num_atom_features), dtype=np.float32)
        self.adj_matrices = np.zeros((N, num_bond_types, max_atoms, max_atoms), dtype=np.float32)
        self.desc_arrays = np.zeros((N, num_descriptors), dtype=np.float32)
        
        self.atom_matrices[0] = sample_atom
        self.adj_matrices[0] = sample_adj
        self.desc_arrays[0] = sample_desc
        
        for i in tqdm(range(1, N), desc="Processing molecules", unit="mol", initial=1, total=N):
            atom_matrix, adjacency_matrix, descriptor_array = self.preprocessor.preprocess(molecules[i])
            
            self.atom_matrices[i] = atom_matrix
            self.adj_matrices[i] = adjacency_matrix
            self.desc_arrays[i] = descriptor_array
        
        self.smiles = smiles_data.tolist()

    def __len__(self):
        return self.atom_matrices.shape[0]

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
            
        atom_mat = torch.from_numpy(self.atom_matrices[idx]).float()
        adj_mat = torch.from_numpy(self.adj_matrices[idx]).float()
        desc_arr = torch.from_numpy(self.desc_arrays[idx]).float()
        
        return {
            "atom_matrix": atom_mat,
            "adjacency_matrix": adj_mat,
            "descriptor_array": desc_arr,
            "smiles": self.smiles[idx],
        }

    def get_feature_dimensions(self):
        if len(self) == 0:
            return None
            
        return {
            "max_atoms": self.atom_matrices.shape[1],
            "num_atom_features": self.atom_matrices.shape[2],
            "num_bond_types": self.adj_matrices.shape[1],
            "num_descriptors": self.desc_arrays.shape[1],
            "descriptor_names": getattr(self.preprocessor, 'descriptor_keys', []),
        }

    def get_flow_inputs(self, idx):
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
            
        atom_mat = torch.from_numpy(self.atom_matrices[idx]).float()
        adj_mat = torch.from_numpy(self.adj_matrices[idx]).float()
        desc_arr = torch.from_numpy(self.desc_arrays[idx]).float()

        return {
            "molecular_flow": {
                "atom_matrix": atom_mat,
                "adjacency_matrix": adj_mat,
            },
            "property_flow": {
                "descriptor_array": desc_arr,
            },
            "smiles": self.smiles[idx],
        }

    def save(self, filepath, compress=False):
        save_data = {
            "atom_matrices": self.atom_matrices,
            "adjacency_matrices": self.adj_matrices,
            "descriptor_arrays": self.desc_arrays,
            "smiles": self.smiles,
            "preprocessor_config": {
                "descriptor_keys": getattr(self.preprocessor, 'descriptor_keys', []),
                "num_descriptors": getattr(self.preprocessor, 'num_descriptors', 0),
            }
        }
        
        with tqdm(total=1, desc="Saving dataset", unit="file") as pbar:
            if compress:
                torch.save(save_data, filepath, _use_new_zipfile_serialization=True)
            else:
                torch.save(save_data, filepath)
            pbar.update(1)

    @classmethod
    def load(cls, filepath, preprocessor=None, verbose=True):
        
        with tqdm(total=1, desc="Loading dataset", unit="file") as pbar:
            data = torch.load(filepath, map_location='cpu')
            pbar.update(1)
        
        dataset = cls.__new__(cls)
        
        if preprocessor is not None:
            dataset.preprocessor = preprocessor
        else:
            class MinimalPreprocessor:
                def __init__(self, config):
                    self.descriptor_keys = config.get('descriptor_keys', [])
                    self.num_descriptors = config.get('num_descriptors', 0)
            
            dataset.preprocessor = MinimalPreprocessor(data.get('preprocessor_config', {}))
        
        dataset.smile_dataset = None
        dataset.atom_matrices = data["atom_matrices"]
        dataset.adj_matrices = data["adjacency_matrices"]
        dataset.desc_arrays = data["descriptor_arrays"]
        dataset.smiles = data["smiles"]
        
        return dataset