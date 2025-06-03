def get_descriptors(mol, properties):
    from rdkit.Chem import Descriptors
    result = {}
    for prop in properties:
        if hasattr(Descriptors, prop):
            result[prop] = getattr(Descriptors, prop)(mol)
        else:
            raise ValueError(f"Descriptor {prop} not found in RDKit Descriptors.")
    return result