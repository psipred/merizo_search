three_to_single_aa = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    'UNK': 'X', 'ASH': 'D', 'GLH': 'E', 'HID': 'H', 'HIE': 'H',
    'HIP': 'H', 'HSD': 'H', 'HSE': 'H', 'LYN': 'K',
}

single_to_three_aa = {v: k for k, v in three_to_single_aa.items()}
