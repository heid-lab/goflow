#!/usr/bin/env python3

import os
import io
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image

def get_atom_index_by_mapnum(mol, mapnum):
    for atom in mol.GetAtoms():
        if atom.GetAtomMapNum() == mapnum:
            return atom.GetIdx()
    return None

def calc_largest_2D_dim(mol):
    conf = mol.GetConformer()
    xs, ys = [], []
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        xs.append(pos.x)
        ys.append(pos.y)
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    width = xmax - xmin
    height = ymax - ymin
    return max(width, height)

def draw_molecule_to_png(
    mol,
    highlightAtoms=None,
    highlightBonds=None,
    image_size=400,
    fixedBondLength=30.0,
    fontSize=10.0,
    label=""
):
    mol.RemoveAllConformers()
    rdDepictor.Compute2DCoords(mol)
    
    drawer = rdMolDraw2D.MolDraw2DCairo(image_size, image_size)
    opts = drawer.drawOptions()
    opts.prepareMolsBeforeDrawing = True
    opts.fixedBondLength = fixedBondLength
    opts.fixedFontSize = round(fontSize)

    drawer.DrawMolecule(
        mol,
        highlightAtoms=highlightAtoms,
        highlightBonds=highlightBonds,
        legend=label
    )
    drawer.FinishDrawing()
    return drawer.GetDrawingText()

def draw_two_molecules_side_by_side(
    r_mol,
    p_mol,
    labelLeft="Reactant",
    labelRight="Product",
    highlightAtomsLeft=None,
    highlightBondsLeft=None,
    highlightAtomsRight=None,
    highlightBondsRight=None,
    square_size=400,
    baseFontSize=10.0,
    out_path="combined.png"
):
    r_mol.RemoveAllConformers()
    p_mol.RemoveAllConformers()
    rdDepictor.Compute2DCoords(r_mol)
    rdDepictor.Compute2DCoords(p_mol)
    
    r_dim = calc_largest_2D_dim(r_mol)
    p_dim = calc_largest_2D_dim(p_mol)
    largest_dim = max(r_dim, p_dim)
    
    if largest_dim > 0:
        fixedBondLength = 0.8 * square_size / largest_dim
    else:
        fixedBondLength = 30.0
    
    bondLengthScale = fixedBondLength / 30.0
    finalFontSize = baseFontSize * bondLengthScale
    
    r_png_data = draw_molecule_to_png(
        r_mol,
        highlightAtoms=highlightAtomsLeft,
        highlightBonds=highlightBondsLeft,
        image_size=square_size,
        fixedBondLength=fixedBondLength,
        fontSize=finalFontSize,
        label=labelLeft
    )
    p_png_data = draw_molecule_to_png(
        p_mol,
        highlightAtoms=highlightAtomsRight,
        highlightBonds=highlightBondsRight,
        image_size=square_size,
        fixedBondLength=fixedBondLength,
        fontSize=finalFontSize,
        label=labelRight
    )
    
    r_img = Image.open(io.BytesIO(r_png_data))
    p_img = Image.open(io.BytesIO(p_png_data))
    
    combined_width = r_img.width + p_img.width
    combined_height = max(r_img.height, p_img.height)
    combined_img = Image.new("RGB", (combined_width, combined_height), color=(255, 255, 255))
    combined_img.paste(r_img, (0, 0))
    combined_img.paste(p_img, (r_img.width, 0))
    
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)  # CHANGED: handle empty dirname
    combined_img.save(out_path)

def get_matching_atoms(template_mol, main_mol):
    atom_map_nums = []
    atom_idxs = []
    for atom in template_mol.GetAtoms():
        mapnum = atom.GetAtomMapNum()
        atom_map_nums.append(mapnum)
        atom_idxs.append(get_atom_index_by_mapnum(main_mol, mapnum))
    return atom_map_nums, atom_idxs

def get_matching_bonds(template_mol, main_mol, matched_atom_idxs):
    bond_idxs = []
    for bond in template_mol.GetBonds():
        aid1 = matched_atom_idxs[bond.GetBeginAtomIdx()]
        aid2 = matched_atom_idxs[bond.GetEndAtomIdx()]
        main_bond = main_mol.GetBondBetweenAtoms(aid1, aid2)
        if main_bond is not None:
            bond_idxs.append(main_bond.GetIdx())
    return bond_idxs