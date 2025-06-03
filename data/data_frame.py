import pandas as pd
from rdkit import Chem
from tqdm import tqdm
from data.utils import get_descriptors

class SmileDataset:
    def __init__(self, path, sep=',', index_col=None, smile_col='smiles', compute_stats=True):
        self.data = pd.read_csv(path, sep=sep, index_col=index_col)
        self.smile_col = smile_col

        if smile_col not in self.data.columns:
            raise ValueError(f"Column '{smile_col}' not found.")

        # Clean and convert SMILES to molecules
        self.data = self.data.dropna(subset=[smile_col]).reset_index(drop=True)
        tqdm.pandas(desc="Converting SMILES to molecules")
        self.data['mol'] = self.data[smile_col].progress_apply(Chem.MolFromSmiles)
        self.data = self.data[self.data['mol'].notnull()].reset_index(drop=True)
        
        self.descriptors = []
        
        if compute_stats:
            self._compute_stats()
        else:
            self.atomic_numbers = []
            self.max_atoms = -1

    def _compute_stats(self):
        atom_counts = []
        all_atomic_numbers = []
        
        for mol in tqdm(self.data['mol'], desc="Analyzing molecular properties"):
            atom_counts.append(mol.GetNumAtoms())
            atomic_nums = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
            all_atomic_numbers.extend(atomic_nums)

        atom_set = set(all_atomic_numbers)
        atom_set.add(0)  # Add padding token
        
        self.atomic_numbers = sorted(atom_set)
        self.max_atoms = max(atom_counts) if atom_counts else 0

    def add_properties(self, rdkit_descriptors, dataframe_properties=None):
        self.descriptors = rdkit_descriptors.copy()
        if dataframe_properties:
            self.descriptors.extend(dataframe_properties)
        
        def set_properties(row):
            mol = row['mol']
            
            # Calculate RDKit descriptors
            desc_values = get_descriptors(mol, rdkit_descriptors)
            
            # Set properties on molecule
            for key, value in desc_values.items():
                mol.SetProp(key, str(value))
            
            # Add DataFrame properties to molecule
            if dataframe_properties:
                for prop in dataframe_properties:
                    if prop in row and not pd.isna(row[prop]):
                        mol.SetProp(prop, str(row[prop]))
                        desc_values[prop] = float(row[prop])
            
            return mol, desc_values

        tqdm.pandas(desc="Adding molecular properties")
        self.data[['mol', 'descriptors']] = self.data.progress_apply(
            set_properties, axis=1, result_type='expand'
        )

    def get_statistics(self):
        if not hasattr(self, 'max_atoms'):
            print("Statistics not computed. Use compute_stats=True in constructor.")
            return {}
        
        return {
            'num_molecules': len(self.data),
            'max_atoms': self.max_atoms,
            'atomic_numbers': self.atomic_numbers
        }
    
    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.data):
            raise IndexError("Index out of bounds")
        
        row = self.data.iloc[idx]
        return {
            'smiles': row[self.smile_col],
            'mol': row['mol'],
            'descriptors': row.get('descriptors', {})
        }
        
    def __len__(self):
        return len(self.data)