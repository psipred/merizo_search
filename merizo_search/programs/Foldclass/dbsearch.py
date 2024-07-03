#!/usr/bin/env python

import os
import sys
import argparse
import pickle
import logging

import torch
import torch.nn.functional as F

from .nndef_fold_egnn_embed import FoldClassNet
from .utils import (
    read_pdb, 
    write_pdb, 
    run_tmalign,
)

logger = logging.getLogger(__name__)

def network_setup(threads: int, device: str) -> FoldClassNet:
    if threads > 0:
        torch.set_num_threads(threads)
    
    device = torch.device(device)
    network = FoldClassNet(128).eval().to(device)

    scriptdir = os.path.dirname(os.path.realpath(__file__))
    network.load_state_dict(torch.load(scriptdir + '/FINAL_foldclass_model.pt', map_location=lambda storage, loc: storage), strict=False)
    
    return network, device


def read_database(db_name: str, device: str):
    target_db = torch.load(db_name + '.pt').to(device)
    
    with open(db_name + '.index', 'rb') as targfile:
        target_index = pickle.load(targfile)

    assert len(target_index) == target_db.size(0)

    target_lengths = torch.tensor([len(target[2]) for target in target_index], dtype=torch.float, device=device)
    
    return {'database': target_db, 'index': target_index, 'lengths': target_lengths}


def search_query_against_db(query_dict, target_dict, mincov, topk):
    mask = (len(query_dict['seq']) >= target_dict['lengths'] * mincov).float()
    
    scores = F.cosine_similarity(target_dict['database'], query_dict['embedding'], dim = -1) * mask
    top_scores, top_indices = torch.topk(scores, topk, dim=0)
    
    return {'scores': top_scores, 'indices': top_indices}
    
    
def dbsearch(query, target_dict: dict, tmp: str, network: FoldClassNet, topk: int, mincov: float, mincos: float, mintm: float, fastmode: bool, device: torch.device, inputs_are_ca: bool=False, pdb_chain: str="A"):
    
    with torch.no_grad():
        if inputs_are_ca:
            # Query is dict of coords and sequence
            query_dict = query
        else:
            # Read coords and seq from PDB file
            query_dict = read_pdb(pdbfile=query, pdb_chain=pdb_chain)
            
        query_input = torch.from_numpy(query_dict['coords']).unsqueeze(0).to(device)
        query_dict['embedding'] = network(query_input)
        
        result_dict = search_query_against_db(
            query_dict=query_dict, target_dict=target_dict, mincov=mincov, topk=topk)

        results = {}
        for i in range(min(topk, result_dict['scores'].size(0))):
            if result_dict['scores'][i] >= mincos:
                target_name, target_coords, target_seq = target_dict['index'][result_dict['indices'][i]]

                query_fn = write_pdb(tmp, query_dict['coords'], query_dict['seq'])
                target_fn = write_pdb(tmp, target_coords, target_seq)
                
                tm_output = run_tmalign(query_fn, target_fn, options='-fast' if fastmode else None)
                max_tm = max(tm_output['qtm'], tm_output['ttm'])
                
                if tm_output['len_ali'] >= len(target_seq) * mincov and max_tm >= mintm:
                    results[i] = {
                        'query': os.path.basename(query_dict['name']).replace('.pdb',''), 
                        'target': os.path.basename(target_name).replace('.pdb',''), 
                        'score': result_dict['scores'][i],
                        'q_len': len(query_dict['seq']), 
                        't_len': len(target_seq), 
                        'tmalign_output': tm_output,
                        'dom_str': query_dict['dom_str'] if 'dom_str' in query_dict.keys() else None,
                        'dom_conf': query_dict['dom_conf'] if 'dom_conf' in query_dict.keys() else None,
                        'dom_plddt': query_dict['dom_plddt'] if 'dom_plddt' in query_dict.keys() else None,
                    }

        return results

def run_dbsearch(inputs: list[str], db_name: str, tmp: str, device: torch.device, topk: int, fastmode: bool, 
                 threads: int, mincos: float, mintm: float, mincov: float, inputs_are_ca: bool=False, pdb_chain: str="A") -> None:
    
    if len(inputs) == 0:
        logging.error("No inputs were provided!")
        sys.exit(1)
    
    if not os.path.exists(tmp):
        os.mkdir(tmp)

    # Set up the network 
    network, device = network_setup(threads=threads, device=device)
    
    # Read the database file
    target_db = read_database(db_name=db_name, device=device)

    # Search the input against the database
    search_results = []
    for pdb in inputs:
        results = dbsearch(
            query=pdb, 
            target_dict=target_db, 
            tmp=tmp, 
            network=network, 
            mincov=mincov, 
            mincos=mincos,
            mintm=mintm,
            topk=topk, 
            fastmode=fastmode, 
            device=device, 
            inputs_are_ca=inputs_are_ca,
            pdb_chain=pdb_chain
        )
        
        search_results.append(results)
        
    return search_results

            
            # if tmalign_output:
            #     tmalign_out = extract_tmalign_values(tmalign_output)
            #     print(tmalign_out)
            #     exit()
                
            #     # if aligned_length is not None:
            #     # if aligned_length >= len(target_seq) * mincov and max(tm_scores) >= mintm:
                #     bn_q = os.path.basename(input).replace('.pdb', '')
                #     bn_t = os.path.basename(target_name).replace('.pdb', '')
                
                

            #     line = "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.6f}\t{:.4f}".format(
            #         bn_q, bn_t, aligned_length, rmsd, seq_identity, len(query_dict['seq']),
            #         len(target_seq), result_dict['scores'][i].item(), max(tm_scores)
            #     )
                
            #     print(line)

            
            # if result_dict['scores'][i] >= mincos:
            #     target_name, target_coords, target_seq = target_dict['index'][result_dict['indices'][i]]

                # tmalign_output = run_tmalign(fname_q, fname_t, options=tmopts)
                # if tmalign_output:
                #     aligned_length, rmsd, seq_identity, tm_scores, alignment_lines = extract_tmalign_values(tmalign_output)
                #     if aligned_length is not None:
                #         if aligned_length >= len(seq_t) * mincov and max(tm_scores) >= mintm:
                #             bn_q = os.path.basename(fname_q).replace('.pdb', '')
                #             bn_t = os.path.basename(fname_t).replace('.pdb', '')
                            
                #             line = "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.6f}\t{:.4f}\n".format(bn_q, bn_t, aligned_length, rmsd, seq_identity, len(seq_q), len(seq_t), top_scores[i].item(), max(tm_scores))
                #             print(line)
                                
            # processed_list.append(fname_q)
            
    # with open(log, 'w+') as fn:
    #     for line in processed_list:
    #         fn.write(os.path.basename(line) + '\n')


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputs', type=str, required=False, default=None, help='File containing list of PDB paths to run on.')
    parser.add_argument('-o', '--outfile', type=str, required=True, help='File to write results to.')
    parser.add_argument('-n', '--dbname', type=str, required=True)
    parser.add_argument('-k', '--topk', type=int, default=1, required=False)
    parser.add_argument('-f', '--fastmode', action='store_true', required=False)
    parser.add_argument('-t', '--threads', type=int, default=-1, required=False)
    parser.add_argument('-s', '--mincos', type=float, default=0.5, required=False)
    parser.add_argument('-m', '--mintm', type=float, default=0.5, required=False)
    parser.add_argument('-c', '--mincov', type=float, default=0.7, required=False)
    parser.add_argument('-ca', '--ca_coords', type=list, default=None, help='List of CA coordinates with shape list([[N, 3]]')
    parser.add_argument('-d', '--device', type=str, default='cpu', required=False)
    parser.add_argument("--pdb_chain", type=str, dest="pdb_chain", default="A", help="Select which PDB Chain you are analysing. Defaut is chain A")
    args = parser.parse_args()

    with open(args.inputs, 'r') as fn:
        inputs = [line.rstrip('\n') for line in fn.readlines()]

    device = torch.device(args.device)
    run_dbsearch(inputs, device, args.topk, args.dbname, args.fastmode, args.threads, args.mincos, args.mintm, args.mincov, args.pdb_chain)
