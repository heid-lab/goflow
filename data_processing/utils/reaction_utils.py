#!/usr/bin/env python3

from typing import List, Tuple
import numpy as np
from rdkit import Chem 

def get_atom_index_by_mapnum(mol, mapnum):
    for atom in mol.GetAtoms():
        if atom.GetAtomMapNum() == mapnum:
            return atom.GetIdx()
    return None

def unmap_smarts(smarts_str):
    mol = Chem.MolFromSmarts(smarts_str)
    if mol is None:
        raise ValueError(f"Could not parse the given SMARTS string: {smarts_str}")
    for atom in mol.GetAtoms():
        atom.ClearProp('molAtomMapNumber')
    unmapped_smarts = Chem.MolToSmarts(mol)
    return unmapped_smarts

def unmap_smiles(smiles_str):
    params = Chem.SmilesParserParams()
    params.removeHs = False
    params.sanitize = False
    mol = Chem.MolFromSmiles(smiles_str, params=params)
    if mol is None:
        raise ValueError(f"Could not parse the given SMILES string: {smiles_str}")
    for atom in mol.GetAtoms():
        atom.ClearProp('molAtomMapNumber')
    # CHANGED: Use MolToSmiles to return a proper SMILES string
    unmapped_smiles = Chem.MolToSmiles(mol)
    return unmapped_smiles

def smarts2smarts(smarts_str):
    mol = Chem.MolFromSmarts(smarts_str)
    if mol is None:
        raise ValueError(f"Could not parse the given SMARTS string: {smarts_str}")
    new_smarts_str = Chem.MolToSmarts(mol)
    return new_smarts_str

def smiles2smiles(smiles_str):
    params = Chem.SmilesParserParams()
    params.removeHs = False
    params.sanitize = False
    mol = Chem.MolFromSmiles(smiles_str, params=params)
    if mol is None:
        raise ValueError(f"Could not parse the given SMILES string: {smiles_str}")
    # CHANGED: Use MolToSmiles for proper SMILES output
    new_smiles_str = Chem.MolToSmiles(mol)
    return new_smiles_str

def bondtypes(atom):
    return sorted(b.GetBondType() for b in atom.GetBonds())

def neighbors(atom):
    return sorted(n.GetAtomMapNum() for n in atom.GetNeighbors())

def neighbors_and_bondtypes(atom):
    return neighbors(atom) + bondtypes(atom)

def remove_atoms_from_rxn(mr, mp, r_p_atoms2remove_N_2):
    r_p_smiles_new_2 = []
    for mol_i, mol in enumerate([mr, mp]):
        editable_mol = Chem.EditableMol(mol)
        for i in sorted(r_p_atoms2remove_N_2[:, mol_i], reverse=True):
            editable_mol.RemoveAtom(int(i))
        r_p_smiles_new_2.append(Chem.MolToSmiles(editable_mol.GetMol(), allHsExplicit=True))
    return r_p_smiles_new_2

def get_small_rxn_smiles_and_mapnums2remove(r_smiles, p_smiles) -> Tuple[str, List[int]]:
    params = Chem.SmilesParserParams()
    params.removeHs = False
    params.sanitize = False
    mr = Chem.MolFromSmiles(r_smiles, params=params)
    mp = Chem.MolFromSmiles(p_smiles, params=params)

    p_map2i = {a.GetAtomMapNum(): a.GetIdx() for a in mp.GetAtoms()}

    def r2p_atom(ar):
        return mp.GetAtomWithIdx(p_map2i[ar.GetAtomMapNum()])

    r_p_atoms2remove_N_2 = np.array([(ar.GetIdx(), p_map2i[ar.GetAtomMapNum()]) for ar in mr.GetAtoms() 
                                      if neighbors_and_bondtypes(ar) == neighbors_and_bondtypes(r2p_atom(ar))
                                     ])
    if len(r_p_atoms2remove_N_2) == 0:
        return ['>>'.join([r_smiles, p_smiles]), []]

    r_p_smiles_new_2 = remove_atoms_from_rxn(mr, mp, r_p_atoms2remove_N_2)

    mapnums2remove = sorted([ar.GetAtomMapNum()-1 for ar in mr.GetAtoms() 
                              if neighbors_and_bondtypes(ar) == neighbors_and_bondtypes(r2p_atom(ar))
                             ])

    return ['>>'.join(r_p_smiles_new_2), mapnums2remove]