#!/usr/bin/env python

import os
import sys
import argparse
import pickle
import logging
import re
import mmap

import numpy as np
import torch
import torch.nn.functional as F

from .nndef_fold_egnn_embed import FoldClassNet
from .utils import (
    read_pdb, 
    write_pdb, 
    run_tmalign,
)

from .dbutil import (
    read_dbinfo, 
    retrieve_names_by_idx,
    retrieve_start_end_by_idx,
    retrieve_bytes,
    coord_conv,
    db_iterator,
    db_memmap,
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
    # first determine if we're using a faiss db or a pt one
    if os.path.exists(db_name + '.pt'):
        target_db = torch.load(db_name + '.pt').to(device)

        with open(db_name + '.index', 'rb') as targfile:
            target_index = pickle.load(targfile)

        assert len(target_index) == target_db.size(0)

        target_lengths = torch.tensor([len(target[2]) for target in target_index], dtype=torch.float, device=device)
        mdfn = db_name + '.metadata'
        mifn = mdfn + '.index'
        if not os.path.exists(mdfn) or not os.path.exists(mifn):
            mdfn = mifn = None

        return {'database': target_db, 'index': target_index, 'lengths': target_lengths, 'faiss': False, 'mdfn': mdfn, 'mifn': mifn}
    elif os.path.exists(db_name + '.json'):
        # faiss db; db reading is done elsewhere
        target_db = db_name+'.json'
        
        return {'database': target_db, 'faiss': True}
    else:
        logger.error(db_name,'is not a valid db or the path basename is incorrect; neither', db_name+'.pt', 'nor', db_name+'.json', 'were found.')
        sys.exit(1)


def search_query_against_db(query_dict, target_dict, mincov, topk, score_corrections=None):
    mask = (len(query_dict['seq']) >= target_dict['lengths'] * mincov).float()
    # modify mask with score corrections
    scores = F.cosine_similarity(target_dict['database'], query_dict['embedding'], dim = -1) * mask
    top_scores, top_indices = torch.topk(scores, topk, dim=0)
    
    return {'scores': top_scores, 'indices': top_indices}
    
    
def dbsearch(query, target_dict: dict, tmp: str, network: FoldClassNet, 
             topk: int, mincov: float, mincos: float, mintm: float, 
             fastmode: bool, device: torch.device, inputs_are_ca: bool=False, 
             pdb_chain: str="A", skip_tmalign=False, score_corrections=None):
    
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
            query_dict=query_dict, target_dict=target_dict, mincov=mincov, topk=topk, score_corrections=score_corrections)
        
        if target_dict['mdfn'] is not None and target_dict['mifn'] is not None:
            mifname = target_dict['mifn']
            mdfname = target_dict['mdfn']

            mif = open(mifname, 'rb')
            mdf = open(mdfname, 'rb')
            mimm = mmap.mmap(mif.fileno(), 0, access=mmap.ACCESS_READ)
            mdmm = mmap.mmap(mdf.fileno(), 0, access=mmap.ACCESS_READ)
        else:
            mimm = mdmm = None
            metadata = '{ }'

        results = {}
        all_results = {}
        if skip_tmalign:
            for i in range(min(topk, result_dict['scores'].size(0))):
                if mimm is not None:
                    startend = retrieve_start_end_by_idx(idx=[result_dict['indices'][i]], mm=mimm)

                    for start, end in startend:
                        metadata = retrieve_bytes(start, end, mm=mdmm, typeconv=lambda x: x.decode('ascii'))
                target_name, target_coords, target_seq = target_dict['index'][result_dict['indices'][i]]
                if result_dict['scores'][i] >= mincos:
                    results[i] = {
                        'query': os.path.basename(query_dict['name']).replace('.pdb',''), 
                        'target': os.path.basename(target_name).replace('.pdb',''), 
                        'score': result_dict['scores'][i],
                        'q_len': len(query_dict['seq']), 
                        't_len': len(target_seq), 
                        'tmalign_output': None,
                        'dom_str': query_dict['dom_str'] if 'dom_str' in query_dict.keys() else None,
                        'dom_conf': query_dict['dom_conf'] if 'dom_conf' in query_dict.keys() else None,
                        'dom_plddt': query_dict['dom_plddt'] if 'dom_plddt' in query_dict.keys() else None,
                        'dbindex': result_dict['indices'][i],
                        'metadata': metadata,
                    }
                # commented this out for parity with the faiss version.
                # else:
                #     all_results[i] = {
                #     'query': os.path.basename(query_dict['name']).replace('.pdb',''), 
                #     'target': os.path.basename(target_name).replace('.pdb',''), 
                #     'score': result_dict['scores'][i],
                #     'q_len': len(query_dict['seq']), 
                #     't_len': len(target_seq), 
                #     'tmalign_output': None,
                #     'dom_str': query_dict['dom_str'] if 'dom_str' in query_dict.keys() else None,
                #     'dom_conf': query_dict['dom_conf'] if 'dom_conf' in query_dict.keys() else None,
                #     'dom_plddt': query_dict['dom_plddt'] if 'dom_plddt' in query_dict.keys() else None,
                #     'dbindex': result_dict['indices'][i],
                #     'metadata': metadata,
                #     }                         
        else: # not skip_tmalign
            for i in range(min(topk, result_dict['scores'].size(0))):
                if result_dict['scores'][i] >= mincos:
                    target_name, target_coords, target_seq = target_dict['index'][result_dict['indices'][i]]

                    query_fn = write_pdb(tmp, query_dict['coords'], query_dict['seq'])
                    target_fn = write_pdb(tmp, target_coords, target_seq)
                    
                    tm_output = run_tmalign(query_fn, target_fn, options='-fast' if fastmode else None)
                    max_tm = max(tm_output['qtm'], tm_output['ttm'])
                    
                    if tm_output['len_ali'] >= len(target_seq) * mincov:
                        if mimm is not None:
                            startend = retrieve_start_end_by_idx(idx=[result_dict['indices'][i]], mm=mimm)

                            for start, end in startend:
                                metadata = retrieve_bytes(start, end, mm=mdmm, typeconv=lambda x: x.decode('ascii'))
                        if max_tm >= mintm:
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
                                'dbindex': result_dict['indices'][i],
                                'metadata': metadata,
                            }
                        else:
                            all_results[i] = {
                            'query': os.path.basename(query_dict['name']).replace('.pdb',''), 
                            'target': os.path.basename(target_name).replace('.pdb',''), 
                            'score': result_dict['scores'][i],
                            'q_len': len(query_dict['seq']), 
                            't_len': len(target_seq), 
                            'tmalign_output': tm_output,
                            'dom_str': query_dict['dom_str'] if 'dom_str' in query_dict.keys() else None,
                            'dom_conf': query_dict['dom_conf'] if 'dom_conf' in query_dict.keys() else None,
                            'dom_plddt': query_dict['dom_plddt'] if 'dom_plddt' in query_dict.keys() else None,
                            'dbindex': result_dict['indices'][i],
                            'metadata': metadata,
                            }         
        # end if/else skip_tmalign                
        return results, all_results


def dbsearch_faiss(queries: list[dict], target_dict: dict, tmp: str, network: FoldClassNet, 
                topk: int, mincov: float, mincos: float, mintm: float, fastmode: bool,
                device: torch.device, inputs_are_ca: bool=False, 
                search_batchsize:int=262144, search_type='IP', pdb_chain:str='A', 
                skip_tmalign=False, score_corrections=None):

    
    import faiss
    # from faiss.contrib.exhaustive_search import knn_ground_truth

    def knn_exact_faiss(xq, db_iterator, k, metric_type=faiss.METRIC_INNER_PRODUCT, device=torch.device('cpu')):
        """Computes the exact KNN search results for a dataset that possibly
        does not fit in RAM but for which we have an iterator that
        returns it block by block.
        SMK modified version of faiss.contrib.exhaustive_search.knn_ground_truth()
        """
        import time
        logger.info("knn_exact_faiss queries size %s k=%d" % (xq.shape, k))
        t0 = time.time()
        nq, d = xq.shape
        keep_max = faiss.is_similarity_metric(metric_type)
        rh = faiss.ResultHeap(nq, k, keep_max=keep_max)

        index = faiss.IndexFlat(d, metric_type)

        if device != torch.device('cpu') and faiss.get_num_gpus():
            logger.info('running on %d GPU(s)' % faiss.get_num_gpus())
            index = faiss.index_cpu_to_all_gpus(index)

        # compute scores by blocks, and add to heaps
        i0 = 0
        for xbi in db_iterator:
            ni = xbi.shape[0]
            index.add(xbi)
            D, I = index.search(xq, k)
            I += i0
            ## optionally modify D with corrections here; keep track of which D is for which query chain/domain/hit chain
            rh.add_result(D, I)
            index.reset()
            i0 += ni
            logger.info("%d DB elements, %.3f s" % (i0, time.time() - t0))

        rh.finalize()
        logger.info("kNN time: %.3f s (%d vectors)" % (time.time() - t0, i0))

        return rh.D, rh.I


    if len(queries) == 0:
        logger.error("No inputs were provided!")
        sys.exit(1)
    
    if not os.path.exists(tmp):
        os.mkdir(tmp)

    nq = len(queries)

    dbinfofname = target_dict['database']
    dbinfo = read_dbinfo(dbinfofname)
    db_dir = os.path.dirname(dbinfofname)
    
    if search_type=='IP':
        dbfname = os.path.join(db_dir, dbinfo['dbfname_IP'])
        mt=faiss.METRIC_INNER_PRODUCT
    # elif search_type == 'L2':
    #     dbfname = os.path.join(db_dir, dbinfo['dbfname_L2'])
    #     mt=faiss.METRIC_L2
    else:
        logging.error('Invalid/unsupported faiss search type: '+search_type+"\n\tOnly \'IP\' is currently supported.")
        sys.exit(1)
        
    # memory-map the db
    dbmm = db_memmap(filename=dbfname, shape=(dbinfo['DB_SIZE'], dbinfo['DB_DIM']))
    dbi = db_iterator(dbmm, search_batchsize)

    logger.info('DB iterator using batchsize of '+str(search_batchsize))

    query_dicts=[]
    if pdb_chain:
        pdb_chain = pdb_chain.rstrip(",")
        pdb_chains = pdb_chain.split(",")
    else:
        pdb_chains = ["A"] * nq

    with torch.no_grad():
        query_embeddings = torch.zeros(size=(nq,128), dtype=torch.float32, device=device)
        query_lengths = np.zeros(shape=(nq), dtype='int')
        for idx, i in enumerate(range(nq)):
            if inputs_are_ca:
            # Query is dict of coords and sequence
                query_dicts.append(queries[i])
            else:
            # Read coords and seq from PDB file
                query_dicts.append(read_pdb(pdbfile=queries[i], pdb_chain=pdb_chains[idx]))
            
            query_lengths[i] = len(query_dicts[i]['seq'])
            query_input = torch.from_numpy(query_dicts[i]['coords']).unsqueeze(0).to(device)
            #query_dict['embedding'] = \
            query_embeddings[i,:] = network(query_input)
        
    if search_type=='IP':
        query_embeddings = F.normalize(query_embeddings)

    # search using db iterator
    # TODO: SMK this strategy does not allow us to easily implement length/coverage filtering as we have for the pytorch version.
    # faiss IDSelector objects are applied to the db so can't be specified per query in a batch.
    # This needs some thought; simple (but very slow) solution is to just run each query individually and modify knn_exact_faiss().
    # Another way is to apply mincov filter in post for both versions, like for mintm and mincos.

    if device==torch.device('mps'):
        logging.info("Faiss supports CUDA GPUs only, not Apple MPS; falling back to CPU search.")
        device = torch.device('cpu')
    
    D, I = knn_exact_faiss(query_embeddings.cpu(), dbi, topk, metric_type=mt, device=device)

    if faiss.is_similarity_metric(mt):
        D_mask = np.where(D>=mincos)
    else:
        D_mask = np.where(D<=mincos) # we use `mincos` even for L2 searches but here it's a max distance
    
    hit_indices = I[D_mask] # already flatttened
    # logger.info('Hit indices: '+str(hit_indices))
    hit_distances = D[D_mask]
    query_indices = D_mask[0] # which queries the filtered indices belong to
 
    # hit_indices = I.flatten() # already flatttened
    # logger.info('Hit indices: '+str(hit_indices))
    # hit_distances = D.flatten()
    # query_indices = np.repeat(np.arange(nq), topk) # which queries the filtered indices belong to

    n_hits_all_queries = len(query_indices)
    
    # retrieve hit names (and seqlen), coords and seqs
    hit_ids = [] # list of str
    hit_seqs = [] # list of str
    hit_coords = [] # list of np.ndarray
    # hit_lengths = [] # compute below
    hit_metadata = []

    index_names_fname = os.path.join(db_dir, dbinfo['db_names_f'])
    # db_domlengths_fname = os.path.join(db_dir, dbinfo['db_domlengths_f'])
    sifname = os.path.join(db_dir, dbinfo['sif'])
    sdfname = os.path.join(db_dir, dbinfo['sdf'])
    cifname = os.path.join(db_dir, dbinfo['cif'])
    cdfname = os.path.join(db_dir, dbinfo['cdf'])
    logger.info('Retrieve domain hits...')
    with open(index_names_fname, 'rb') as idf:
        idmm = mmap.mmap(idf.fileno(), 0, access=mmap.ACCESS_READ)
        hit_ids = retrieve_names_by_idx(idx=hit_indices, mm=idmm, use_sorting=False)

    # with open(db_domlengths_fname, 'rb']) as dlf:
    #     dlmm = mmap.mmap(dlf.fileno(), 0, access=mmap.ACCESS_READ)
    #     hit_lengths.append( retrieve_domlen_by_idx(idx=hit_indices, mm=dlmm, use_sorting=False) )
    
    with open(sifname, 'rb') as sif, open(sdfname, 'rb') as sdf:
        simm = mmap.mmap(sif.fileno(), 0, access=mmap.ACCESS_READ)
        sdmm = mmap.mmap(sdf.fileno(), 0, access=mmap.ACCESS_READ)

        startend = retrieve_start_end_by_idx(idx=hit_indices, mm=simm)
        
        for start, end in startend:
            hit_seqs.append(retrieve_bytes(start, end, mm=sdmm, typeconv=lambda x: x.decode('ascii')))
    if not skip_tmalign:    
        with open(cifname, 'rb') as cif, open(cdfname, 'rb') as cdf:
            cimm = mmap.mmap(cif.fileno(), 0, access=mmap.ACCESS_READ)
            cdmm = mmap.mmap(cdf.fileno(), 0, access=mmap.ACCESS_READ)

            startend = retrieve_start_end_by_idx(idx=hit_indices, mm=cimm)
            for start, end in startend:
                hit_coords.append(retrieve_bytes(start, end, mm=cdmm, typeconv=coord_conv))

    if 'mif' and 'mdf' in dbinfo.keys():
        mifname = os.path.join(db_dir, dbinfo['mif'])
        mdfname = os.path.join(db_dir, dbinfo['mdf'])

        with open(mifname, 'rb') as mif, open(mdfname, 'rb') as mdf:
            mimm = mmap.mmap(mif.fileno(), 0, access=mmap.ACCESS_READ)
            mdmm = mmap.mmap(mdf.fileno(), 0, access=mmap.ACCESS_READ)

        startend = retrieve_start_end_by_idx(idx=hit_indices, mm=mimm)

        for start, end in startend:
            hit_metadata.append(retrieve_bytes(start, end, mm=mdmm, typeconv=lambda x: x.decode('ascii')))
    else:
        hit_metadata = ['{ }'] * n_hits_all_queries

    hit_lengths = list(map(len, hit_seqs))
    n_queries_with_hits = np.max(query_indices) + 1

    # itertools.repeat makes references, not copies!
    # results = list(repeat(dict(), n_queries_with_hits))
    # results_counts = list(repeat(0, n_queries_with_hits))
    # this also creates references:
    # results = [dict()] * n_queries_with_hits 

    results = [ dict() for _ in range(n_queries_with_hits) ]
    all_results = [ dict() for _ in range(n_queries_with_hits) ]
    
    # structure of 'results': list of dict{0: result_dict1, 1:result_dict2, ...}
    
    results_counts = [0] * n_queries_with_hits

    if not skip_tmalign:
        logger.info("TM-align top hits...")
    n_tm_exclude = 0
    for i in range(n_hits_all_queries):
        qi = query_indices[i]
        if skip_tmalign:
            target_name = hit_ids[i]
            results[qi][ results_counts[qi] ] = {
                'query': os.path.basename(query_dicts[qi]['name']).replace('.pdb',''), 
                'target': os.path.basename(target_name).replace('.pdb',''), 
                'score': hit_distances[i],
                'q_len': len(query_dicts[qi]['seq']), 
                't_len': hit_lengths[i], 
                'tmalign_output': None,
                'dom_str': query_dicts[qi]['dom_str'] if 'dom_str' in query_dicts[qi].keys() else None,
                'dom_conf': query_dicts[qi]['dom_conf'] if 'dom_conf' in query_dicts[qi].keys() else None,
                'dom_plddt': query_dicts[qi]['dom_plddt'] if 'dom_plddt' in query_dicts[qi].keys() else None,
                'dbindex': hit_indices[i],
                'metadata': hit_metadata[i],
            }
            results_counts[qi] += 1
            # no option to output insignificant results when skip_tmalign==True            
        else:

            target_name, target_coords, target_seq = hit_ids[i], hit_coords[i], hit_seqs[i]
                
            query_fn = write_pdb(tmp, query_dicts[qi]['coords'], query_dicts[qi]['seq'], name=os.path.basename(query_dicts[qi]['name']))
            target_fn = write_pdb(tmp, target_coords, target_seq, name=target_name)
                
            tm_output = run_tmalign(query_fn, target_fn, options='-fast' if fastmode else None, keep_pdbs=False)
            max_tm = max(tm_output['qtm'], tm_output['ttm'])
            
            # if tm_output['len_ali'] >= len(target_seq) * mincov and max_tm >= mintm:
            if max_tm >= mintm:
                results[qi][ results_counts[qi] ] = {
                    'query': os.path.basename(query_dicts[qi]['name']).replace('.pdb',''), 
                    'target': os.path.basename(target_name).replace('.pdb',''), 
                    'score': hit_distances[i],
                    'q_len': len(query_dicts[qi]['seq']), 
                    't_len': hit_lengths[i], 
                    'tmalign_output': tm_output,
                    'dom_str': query_dicts[qi]['dom_str'] if 'dom_str' in query_dicts[qi].keys() else None,
                    'dom_conf': query_dicts[qi]['dom_conf'] if 'dom_conf' in query_dicts[qi].keys() else None,
                    'dom_plddt': query_dicts[qi]['dom_plddt'] if 'dom_plddt' in query_dicts[qi].keys() else None,
                    'dbindex': hit_indices[i],
                    'metadata': hit_metadata[i],
                }
                results_counts[qi] += 1
            else:
                all_results[qi][ n_tm_exclude ] = {
                    'query': os.path.basename(query_dicts[qi]['name']).replace('.pdb',''), 
                    'target': os.path.basename(target_name).replace('.pdb',''), 
                    'score': hit_distances[i],
                    'q_len': len(query_dicts[qi]['seq']), 
                    't_len': hit_lengths[i], 
                    'tmalign_output': tm_output,
                    'dom_str': query_dicts[qi]['dom_str'] if 'dom_str' in query_dicts[qi].keys() else None,
                    'dom_conf': query_dicts[qi]['dom_conf'] if 'dom_conf' in query_dicts[qi].keys() else None,
                    'dom_plddt': query_dicts[qi]['dom_plddt'] if 'dom_plddt' in query_dicts[qi].keys() else None,
                    'dbindex': hit_indices[i],
                    'metadata': hit_metadata[i],
                }
                n_tm_exclude +=1

    if n_tm_exclude > 0:
        logger.info('Excluded '+str(n_tm_exclude)+' hits (across all query domains) by TM-score threshold(>='+str(mintm)+')')

    return results, all_results # NB this is the list of hits for *all* query domains, unlike dbsearch(), which returns results for one query domain.


def run_dbsearch(inputs: list[str], db_name: str, tmp: str, device: torch.device, topk: int, fastmode: bool, 
                 threads: int, mincos: float, mintm: float, mincov: float, inputs_are_ca: bool=False, 
                 search_batchsize:int=262144, search_type='IP', pdb_chain: str=None, skip_tmalign:bool=False) -> None:

    
    if len(inputs) == 0:
        logger.error("No inputs were provided!")
        sys.exit(1)
    
    if not os.path.exists(tmp):
        os.mkdir(tmp)

    # Set up the network 
    network, device = network_setup(threads=threads, device=device)
    
    # Read the database file
    target_db = read_database(db_name=db_name, device=device)

    # Search the input against the database
    search_results = []
    all_search_results = []
    # SMK we do things differently when searching against a faiss db
    if target_db['faiss']:
        if search_batchsize < 1:
            logger.error("search_batchsize must be >= 1.")
            sys.exit(1)
        search_results, all_search_results = dbsearch_faiss(queries=inputs,
                              tmp=tmp, 
                              network=network, 
                              mincov=mincov, 
                              mincos=mincos,
                              mintm=mintm,
                              topk=topk, 
                              fastmode=fastmode, 
                              device=device, 
                              inputs_are_ca=inputs_are_ca,
                              pdb_chain=pdb_chain,
                              target_dict=target_db,
                              search_batchsize=search_batchsize,
                              search_type=search_type,
                              skip_tmalign=skip_tmalign
                            )
    else:
        if pdb_chain:
            pdb_chain = pdb_chain.rstrip(",")
            pdb_chains = pdb_chain.split(",")

            if len(inputs) != len(pdb_chains):
                if len(pdb_chains) == 1:
                    pdb_chains = [pdb_chains] * len(inputs)  
                else:
                    logger.error('Number of specified chain IDs not equal to number of input PDB files.')
                    sys.exit(1)
        else:
            pdb_chains = ["A"] * len(inputs)

        for idx, pdb in enumerate(inputs):
            results, all_results = dbsearch(
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
                pdb_chain=pdb_chains[idx],
                skip_tmalign=skip_tmalign
            )

            search_results.append(results)
            all_search_results.append(all_results)
        
    return search_results, all_search_results


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
    parser.add_argument("--pdb_chain", type=str, dest="pdb_chain", default="A", help="Select which PDB Chain you are analysing. Chain 'A' will be analysed if not provided")
    parser.add_argument('--search_batchsize', type=int, default=2097152, required=False)
    parser.add_argument('--search_metric', type=str, default='IP', required=False, help='For searches against Faiss databases, the search metric to use. Ignored otherwise. Currently only \'IP\' (cosine similarity) is supported')
    args = parser.parse_args()

    with open(args.inputs, 'r') as fn:
        inputs = [line.rstrip() for line in fn.readlines()]

    device = torch.device(args.device)
    run_dbsearch(inputs, device, args.topk, args.dbname, args.fastmode, args.threads, args.mincos, args.mintm, args.mincov, args.pdb_chain, args.search_batchsize, args.search_metric)
