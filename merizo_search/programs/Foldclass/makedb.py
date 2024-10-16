#!/usr/bin/env python

from __future__ import print_function

import os
import argparse
import pickle
import logging

import numpy as np
import torch

from .nndef_fold_egnn_embed import FoldClassNet
from .constants import three_to_single_aa

def network_setup(device: str) -> FoldClassNet:
    """
    Set up the FoldClassNet neural network.

    Args:
        device (str): Device to run the network on.

    Returns:
        FoldClassNet: Initialized FoldClassNet network.
    """
    
    network = FoldClassNet(128).to(device).eval().to(torch.device(device))

    scriptdir = os.path.dirname(os.path.realpath(__file__))
    network.load_state_dict(torch.load(scriptdir + '/FINAL_foldclass_model.pt', map_location=lambda storage, loc: storage), strict=False)
    
    return network

def run_createdb(pdb_files: str, out_db: str, device: str) -> None:
    """
    Create a Foldclass database from a directory of PDB files.

    Args:
        pdb_files (str): Path to directory containing PDB files.
        out_db (str): Output prefix for the created Foldclass database.
        device (str): Device to run the neural network on.

    Returns:
        None
    """
    
    pdb_files = [os.path.join(pdb_files, f) for f in os.listdir(pdb_files) if f.endswith('.pdb')]
    pdb_files.sort() # os.listdir() returns entries in arbitrary order; this keeps things consistent between runs
    logging.info(f"{len(pdb_files)} PDB files found in model directory. Will generate Foldclass database..")

    network = network_setup(device=device)

    targets = []
    tvecs = []
    pdbs_read = 0
    
    for pdb in pdb_files:
        with open(pdb, 'r') as targpdbfile:
            coords = []
            seq = []
            n = 0
            for line in targpdbfile:
                if line[:4] == 'ATOM' and line[12:16] == ' CA ':
                    pdb_fields = [line[:6], line[6:11], line[12:16], line[17:20], line[21], line[22:26], line[30:38], line[38:46], line[46:54]]
                    coords.append(np.array([float(pdb_fields[6]), float(pdb_fields[7]), float(pdb_fields[8])]))
                    seq.append(three_to_single_aa.get(pdb_fields[3], 'X'))

        ca_coords_t = np.array(coords, dtype=np.float32)[:2000]
        seq_t = ''.join(seq[:2000])

        if len(ca_coords_t) == 0 or len(seq_t) == 0:
            logging.warning('No CA atoms read from PDB file '+pdb+'; skipping.')
            continue

        inputs = torch.from_numpy(ca_coords_t).unsqueeze(0).to(device)

        with torch.no_grad():
            tvec = network(inputs)
            tvecs.append(tvec.cpu())
            targets.append((pdb, ca_coords_t, seq_t))
            pdbs_read += 1
            
    logging.info(f"Output database contains {pdbs_read} PDBs.")

    search_tensor = torch.cat(tvecs, dim=0)
    out_db_name = out_db + '.pt'
    torch.save(search_tensor, out_db_name)

    out_db_index = out_db + '.index'
    with open(out_db_index, 'wb') as targfile:
        pickle.dump(targets, targfile)
       
    logging.info(f"Saved Foldclass database to {out_db_name}")
    logging.info(f"Saved Foldclass index file to {out_db_index}")

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Merizo createdb calls the createdb function of Foldclass to embed a directory of pdb files into a Foldclass database.")
    parser.add_argument('input_dir', type=str, help='Directory containing pdb files. Will read all .pdb files in this directory.')
    parser.add_argument('out_db', type=str, help='Output prefix for the created Foldclass database.')
    parser.add_argument('-d', '--device', type=str, default='cuda', required=False, help="Decive to use when creating Foldclass embeddings. (default: cuda)")
    args = parser.parse_args()
    
    run_createdb(pdb_files=args.input_dir, out_db=args.out_db, device=args.device)
