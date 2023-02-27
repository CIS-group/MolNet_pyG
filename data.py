import os
import logging

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MolFromSmiles
from rdkit.Chem import rdmolops
from torch_geometric.data import Data
from itertools import permutations
import pickle
from tqdm import tqdm


class Parser(object):
    def __init__(self):
        self.data = [mol for mol in Chem.SDMolSupplier(".data/bace_cla.sdf")]
        self.atom_type =  ['C', 'O', 'N', 'S', 'Cl', 'F', 'Br', 'P', 'I', 'Si', 'B', 'Na', 'Sn', 'Se', 'other']

    def parse_dataset(self):
        total_data = []
        def _one_hot(x, allowable_set):
            if x not in allowable_set:
                x = allowable_set[-1]
            temp = list(map(lambda s: x == s, allowable_set))
            return [1 if i else 0 for i in temp]
        hydrogen_donor = Chem.MolFromSmarts("[$([N;!H0;v3,v4&+1]),$([O,S;H1;+0]),n&H1&+0]")
        hydrogen_acceptor = Chem.MolFromSmarts(
            "[$([O,S;H1;v2;!$(*-*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=[O,N,P,S])]),n&H0&+0,$([o,s;+0;!$([o,s]:n);!$([o,s]:c:n)])]")
        acidic = Chem.MolFromSmarts("[$([C,S](=[O,S,P])-[O;H1,-1])]")
        basic = Chem.MolFromSmarts(
            "[#7;+,$([N;H2&+0][$([C,a]);!$([C,a](=O))]),$([N;H1&+0]([$([C,a]);!$([C,a](=O))])[$([C,a]);!$([C,a](=O))]),$([N;H0&+0]([C;!$(C(=O))])([C;!$(C(=O))])[C;!$(C(=O))])]")
        total_target = []
        for kdx, mol in (enumerate(self.data)):
            if mol is not None:
                if "active" in mol.GetPropNames() and kdx == 0:
        #             total_target.append(int(mol.GetProp("active")))
        # total_target = np.array(total_target)
        # idx_inactive = np.squeeze(np.argwhere(total_target == 0))  # 825
        # idx_active = np.squeeze(np.argwhere(total_target == 1))  # 653 
        # np.random.seed(100)  # 37
        # np.random.shuffle(idx_inactive)
        # np.random.seed(100)  # 37
        # np.random.shuffle(idx_active)
        # result = {}
        # result["active"] = idx_active
        # result["inactive"] = idx_inactive
        # with open("./data/index.pkl", "wb") as f:
        #     pickle.dump(result, f)
                    scalar_feature = []
                    pos = []
                    N = mol.GetNumAtoms()
                    ring = mol.GetRingInfo()
                    hydrogen_donor_match = sum(mol.GetSubstructMatches(hydrogen_donor), ())
                    hydrogen_acceptor_match = sum(mol.GetSubstructMatches(hydrogen_acceptor), ())
                    acidic_match = sum(mol.GetSubstructMatches(acidic), ())
                    basic_match = sum(mol.GetSubstructMatches(basic), ())
                    adj = np.array(rdmolops.GetAdjacencyMatrix(mol), dtype=float)
                    conf = mol.GetConformer()
                    dist = np.array(rdmolops.Get3DDistanceMatrix(mol))
                    for atom_idx in range(N):
                        # make atom feature
                        atom = mol.GetAtomWithIdx(atom_idx)
                        atom_feature = []
                        atom_feature += _one_hot(atom.GetSymbol(), self.atom_type)
                        # atom_feature += [atom.GetAtomicNum()]
                        atom_feature += _one_hot(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6])
                        atom_feature += _one_hot(atom.GetFormalCharge(), [-3, -2, -1, 0, 1, 2, 3])
                        atom_feature += _one_hot(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6])
                        atom_feature += [atom.GetIsAromatic()]
                        atom_feature += _one_hot(atom.GetHybridization(), [Chem.rdchem.HybridizationType.SP,
                                                                    Chem.rdchem.HybridizationType.SP2,
                                                                    Chem.rdchem.HybridizationType.SP3,
                                                                    Chem.rdchem.HybridizationType.SP3D,
                                                                    Chem.rdchem.HybridizationType.SP3D2])
                        atom_feature += [ring.IsAtomInRingOfSize(atom_idx, 3),
                                    ring.IsAtomInRingOfSize(atom_idx, 4),
                                    ring.IsAtomInRingOfSize(atom_idx, 5),
                                    ring.IsAtomInRingOfSize(atom_idx, 6),
                                    ring.IsAtomInRingOfSize(atom_idx, 7),
                                    ring.IsAtomInRingOfSize(atom_idx, 8)]
                        atom_feature += _one_hot(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
                        atom_feature += [atom_idx in acidic_match, atom_idx in basic_match]
                        atom_feature += [atom_idx in hydrogen_donor_match, atom_idx in hydrogen_acceptor_match]
                        scalar_feature.append(atom_feature)

                        # get atom position
                        atom_pos = conf.GetAtomPosition(atom_idx)
                        pos += [list(atom_pos)]

                    # make edge index for B matrix
                    row_B, col_B, edge_attr = [], [], []
                    for i in range(N):
                        for j in range(N):
                            bond = mol.GetBondBetweenAtoms(i, j)
                            if bond is not None:
                                row_B += [i]
                                col_B += [j]
                                bond_type = bond.GetBondType()
                                if bond_type == Chem.rdchem.BondType.SINGLE:
                                    edge_attr += [1]
                                elif bond_type == Chem.rdchem.BondType.DOUBLE:
                                    edge_attr += [2]
                                elif bond_type == Chem.rdchem.BondType.TRIPLE:
                                    edge_attr += [3]
                                elif bond_type == Chem.rdchem.BondType.AROMATIC:
                                    edge_attr += [1.5]
                    # add selfloop for B
                    # for i in range(N):
                        row_B += [i]
                        col_B += [i]
                        edge_attr += [1]

                    m = np.zeros([N, N, 4], dtype=int)
                    for i in range(mol.GetNumAtoms()):
                        for j in range(mol.GetNumAtoms()):
                            bond = mol.GetBondBetweenAtoms(i, j)
                            if bond is not None:
                                bond_type = bond.GetBondType()
                                # bond.GetBeginAtom(), bond.GetEndAtom()
                                
                                m[i, j] = np.squeeze(np.array([
                                    bond_type == Chem.rdchem.BondType.SINGLE, bond_type == Chem.rdchem.BondType.DOUBLE,
                                    bond_type == Chem.rdchem.BondType.TRIPLE, bond_type == Chem.rdchem.BondType.AROMATIC]))

                    # adj based degree of bond
                    bond_single = np.where(m[:, :, 0] == 1,
                                        np.full_like(adj, 1), np.zeros_like(adj))
                    bond_double = np.where(m[:, :, 1] == 1,
                                        np.full_like(adj, 2), np.zeros_like(adj))
                    bond_triple = np.where(m[:, :, 2] == 1,
                                        np.full_like(adj, 3), np.zeros_like(adj))
                    bond_aromatic = np.where(m[:, :, 3] == 1,
                                            np.full_like(adj, 1.5), np.zeros_like(adj))

                    adjms = bond_single + bond_double + bond_triple + bond_aromatic
                    adjms += np.eye(N)
                    degree = np.array(adjms.sum(1))
                    print(degree)
                    deg_inv_sqrt = np.power(degree, -0.5)
                    deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.
                    deg_inv_sqrt = np.diag(deg_inv_sqrt)
                    adjms = np.matmul(np.matmul(deg_inv_sqrt, adjms), deg_inv_sqrt)

                    row_A_, col_A_ = [], []
                    for i in range(N):
                        for j in range(N):
                            if dist[i][j] < 5 and adj[i][j] == 0 and i != j:
                                row_A_ += [i]
                                col_A_ += [j]

                    near = (lambda x: np.where(x < 5,
                                                    np.ones_like(np.sum(x, axis=-1)),
                                                    np.zeros_like(np.sum(x, axis=-1))))(dist)
                    near_not_bond = (
                        lambda x: np.subtract(x[0], np.where(x[1] > np.zeros_like(x[1]), np.ones_like(x[1]), np.zeros_like(x[1]))))(
                        [near, adjms])

                    count=0
                    for i in range(N):
                        for j in range(N):
                            if near_not_bond[i][j] !=0:
                                count+=1
                    print(count)
                    print(len(row_A_))

                    edge_index_A_ = torch.tensor([row_A_, col_A_], dtype=torch.long)          
                    edge_index_B = torch.tensor([row_B, col_B], dtype=torch.long)
                    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
                    y = torch.tensor(float(mol.GetProp('active')), dtype=torch.float).view(-1, 1)
                    pos = torch.tensor(pos).to(torch.float)
                    scalar = torch.tensor(scalar_feature).to(torch.float)

                    data = Data(scalar=scalar, pos=pos, edge_index_A_= edge_index_A_, edge_index_B=edge_index_B, edge_attr=edge_attr, y=y)
                    total_data.append(data)
                torch.save(total_data, "./data/bace_cla.pt")



temp = Parser()
temp.parse_dataset()
