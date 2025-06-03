"""
Preprocessor for both molecules and their descriptors/properties.
This molecular arrays(atom, bond array) code preperocessor adapted from chainer_chemistry\dataset\preprocessors
"""

from rdkit import Chem
import numpy as np

class MolecularPreprocessor:
    def __init__(self, add_hs=False, kekulize=False):
        self.num_bond_types = 4
        self.bond_map = {'SINGLE': 0, 'DOUBLE': 1, 'TRIPLE': 2, 'AROMATIC': 3}

        self.add_hs = add_hs
        self.kekulize = kekulize
        self.loaded = False


    def load(self, max_atoms, atomic_numbers, descriptor_keys=None, add_hs=False, kekulize=False):
        self.max_atoms = max_atoms
        self.atomic_numbers = [0] + list(atomic_numbers)
        self.num_atom_types = len(self.atomic_numbers)
        self.atom_to_idx = {num: idx for idx, num in enumerate(self.atomic_numbers)}
        
        self.descriptor_keys = descriptor_keys or []
        self.num_descriptors = len(self.descriptor_keys)
        
        self.add_hs = add_hs
        self.kekulize = kekulize

        self.loaded = True

    def canonicalize(self, mol):
        if mol is None:
            return None, None
            
        try:
            canonical_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
            canonical_mol = Chem.MolFromSmiles(canonical_smiles)
            
            if canonical_mol is None:
                return None, None
            
            if self.add_hs:
                canonical_mol = Chem.AddHs(canonical_mol)
            if self.kekulize:
                Chem.Kekulize(canonical_mol)
                
            return canonical_mol, canonical_smiles
        except Exception:
            return None, None

    def process_molecule(self, mol):
        if mol is None:
            return None, None
            
        n_atoms = mol.GetNumAtoms()
        if n_atoms > self.max_atoms:
            return None, None

        # Create atom feature matrix
        atom_feats = np.zeros((self.max_atoms, self.num_atom_types), dtype=np.float32)
        for i, atom in enumerate(mol.GetAtoms()):
            atomic_num = atom.GetAtomicNum()
            idx = self.atom_to_idx.get(atomic_num, 0)  # Use index 0 for unknown atoms
            atom_feats[i, idx] = 1.0
        
        # Create bond adjacency matrices (one per bond type)
        adj = np.zeros((self.num_bond_types, self.max_atoms, self.max_atoms), dtype=np.float32)
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_type = str(bond.GetBondType())
            if bond_type in self.bond_map:
                bond_idx = self.bond_map[bond_type]
                adj[bond_idx, i, j] = adj[bond_idx, j, i] = 1.0

        return atom_feats, adj

    def extract_descriptors(self, mol):
        if mol is None or not self.descriptor_keys:
            return np.zeros(self.num_descriptors, dtype=np.float32)
        
        descriptor_values = np.zeros(self.num_descriptors, dtype=np.float32)
        
        for i, key in enumerate(self.descriptor_keys):
            try:
                if mol.HasProp(key):
                    value = float(mol.GetProp(key))
                    descriptor_values[i] = value
                else:
                    descriptor_values[i] = 0.0
            except (ValueError, TypeError):
                descriptor_values[i] = 0.0
        return descriptor_values

    def preprocess(self, mol):
        if mol is None:
            return None, None, None
        
        # Canonicalize molecule first
        mol, _ = self.canonicalize(mol)
        
        atom_matrix, adjacency_matrix = self.process_molecule(mol)
        if atom_matrix is None:
            return None, None, None
        
        descriptor_array = self.extract_descriptors(mol)
        
        return atom_matrix, adjacency_matrix, descriptor_array