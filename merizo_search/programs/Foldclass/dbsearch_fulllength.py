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

from .dbsearch import *


logger = logging.getLogger(__name__)

def full_length_search(queries:list[dict], # each dict has at least 'coords', 'seq', 'name'
                        search_results,
                       target_db,
                       tmp: str,
                       device: torch.device,
                       topk: int=1,
                       fastmode: bool=False, 
                       threads: int=-1,
                       mincos: float=0.5,
                       mintm: float=0.5,
                       mincov: float=0.0,
                       inputs_are_ca: bool=False,
                       search_batchsize:int=262144,
                       search_type='IP',
                       inputs_from_easy_search=False,
                       mode="basic"
                       ):
    # from pprint import pprint # for testing only

    """
    in easy-search: domain names are suffixed with _merizo_01 etc.
    in search: could be anything.
    So, in 'search' mode, we have to treat all queries as coming from one chain.
        in easy-search mode we can map domains to the original queries, so can do it that way.
    """
    if len(queries) == 1: # regardless of the state of inputs_from_easy_search
        logger.warning("cannot execute full-length search with only one query domain.")
        return None
    
    force_expansion = True
    if mode == "basic":
        force_expansion = False

    # extract potential chains for full-length matching
    logger.info('Start full-length search...')
    all_query_domains = list() # merizo-given names for the query domains in dbsearch().
    all_hit_domains = list()
    all_hit_indices = list()
    qd_coords_seqs = dict()
    for qdd in queries:
        qd_coords_seqs[os.path.basename(qdd['name'])] = {"coords": qdd['coords'], "seq":qdd['seq'] }

    # if the same chain id is found for all query domains, then that is a full-length hit. no further work needed.
    # If there are no such ids (or user forces it), then we need to do extra work.

    for hitdict in search_results:
        all_query_domains.extend([ i['query'] for i in hitdict.values() ])        
        all_hit_domains.extend([ i['target'] for i in hitdict.values() ])
        all_hit_indices.extend([ int(i['dbindex']) for i in hitdict.values() ])

    
    assert len(all_query_domains) == len(all_hit_domains) == len(all_hit_indices) 

    if inputs_from_easy_search:
        all_query_chains = [ re.sub("_merizo_[0-9]*$", "", x) for x in all_query_domains ]
    else:
        # treat all query structures as single domains comaing from a single chain
        all_query_chains = [ 'A' for _ in all_query_domains ]

    # if target_db['faiss']:
    #     # AFDB/TED
    #     domid2chainid_fn = lambda x : os.path.basename(x).split('-')[1] # given 'AF-Q96PD2-F1_blahblah' return 'Q96PD2'
    # else:
    #     # cath
    domid2chainid_fn = lambda x : re.sub("[0-9]*$", "", os.path.basename(x).rstrip('.pdb')) # given 'cath-dompdb/2pi4A04.pdb' return '2pi4A'

    all_hit_chains = [ domid2chainid_fn(x) for x in all_hit_domains ]

    #hit_index = np.asarray(list(zip(all_query_chains, all_query_domains, all_hit_chains, all_hit_domains, all_hit_indices)))
    hit_index = dict()    # keys are QUERY CHAINS. Values: { 'query_domain1':[{hit_dict1}, {hit_dict2}, ...],
                          #                                  'query_domain2':[{hit_dict1}, {hit_dict2}, ...]
                          #                                }
    for i in range(len(all_query_chains)):
        qc = all_query_chains[i]
        qd = all_query_domains[i]
        hc = all_hit_chains[i]
        hd = all_hit_domains[i]
        hi = all_hit_indices[i]
        
        if qc not in hit_index.keys():
            hit_index[qc] = dict()        
        if qd not in hit_index[qc].keys():
            hit_index[qc][qd] = dict()
            # look up qd in queries to get coords and seq
            hit_index[qc][qd]['coords'] = qd_coords_seqs[qd]['coords']
            hit_index[qc][qd]['seq'] = qd_coords_seqs[qd]['seq']
            hit_index[qc][qd]['hits'] = list()
        hit_index[qc][qd]['hits'].append({'hc': hc, 'hd': hd, 'hi': hi})
    
    for qc in hit_index.keys():
        # all results should be determined *per query chain*
        # find hit chains common to all query domains for this query chain.
        num_query_domains = len(list(hit_index[qc].keys()))
        
        if num_query_domains == 0:
            logger.info('Query chain '+qc+':no domains detected, skipping full-length search.')
            continue

        if num_query_domains == 1:
            logger.info('Query chain '+qc+': Only one detected domain, so full-length hits are same as those in the per-domain search results.')
            continue
        
        hit_chains_per_query_domain =dict()
        
        for qd in hit_index[qc].keys():
            hit_chains_per_query_domain[qd] = set( [ v['hc'] for v in hit_index[qc][qd]['hits'] ] )
            
        intersection = set.intersection( *hit_chains_per_query_domain.values() )
        
        if len(intersection) == 0 and not force_expansion:
            logger.info("Query chain "+ qc +": No full-length hits found with k = "+str(topk)+", maybe try again with a higher value of -k or enable --extra-full-length.")
            #logger.info("   NB: a high setting of -k *and* --extra-full-length will greatly increase runtime for full-length search.")
        else:
            nint = len(intersection)
            logger.info("Query chain " + qc + ": "+ str(nint) + " full-length hits found in top-k hit lists.")
            # TODO: write list to file?
            
        
    if force_expansion:
        logger.info("Starting expanded full-length search...")
        if target_db['faiss']:
            # AFDB/TED
            dbinfofname = target_db['database']
            dbinfo = read_dbinfo(dbinfofname) # a dict 
            db_dir = os.path.dirname(dbinfofname)
            index_names_fname = os.path.join(db_dir, dbinfo['db_names_f'])
            # db_domlengths_fname = os.path.join(db_dir, dbinfo['db_domlengths_f'])
            sifname = os.path.join(db_dir, dbinfo['sif'])
            sdfname = os.path.join(db_dir, dbinfo['sdf'])
            cifname = os.path.join(db_dir, dbinfo['cif'])
            cdfname = os.path.join(db_dir, dbinfo['cdf'])
        else:
            # cath
            dbindex = target_db['index'] # already parsed pickle
            
        # construct db_indices_to_extract by checking db ids in the vicinity of the hit indices.
        all_db_indices_to_extract = dict()
            
        for qc in hit_index.keys():
            if len(hit_index[qc].keys()) < 2:
                logger.info("Skipping query chain "+qc+" as it has only one detected domain.")
                continue

            logger.info("Start expanded full-length search for query chain "+qc)
            db_indices_to_extract = []
            nqd = len(hit_index.keys())
            for qd in hit_index[qc].keys():
                for hit in hit_index[qc][qd]['hits']:
                    anchor_index = hit['hi']
                    anchor_chain = hit['hc'] # chain name we are looking for
                    anchor_domain = hit['hd'] # not sure we need this yet
                    
                    curr_idx_list = [] # indices to extract
                    
                    # left
                    cur_i = anchor_index
                    while( domid2chainid_fn(dbindex[cur_i-1][0]) == anchor_chain):
                        curr_idx_list.append(cur_i)
                        cur_i-=1
                    # right
                    cur_i = anchor_index
                    while( domid2chainid_fn(dbindex[cur_i+1][0]) == anchor_chain):
                        curr_idx_list.append(cur_i)
                        cur_i+=1
                        
                    #curr_idx_list.sort() # dunno if needed
                    nhd = len(curr_idx_list)

                    if nhd >= nqd : # don't bother if the hit chain is single-domain or has fewer domains than the query
                        db_indices_to_extract.extend(curr_idx_list)
                    
                # end for hit in hit_index[qc][qd]['hits']
            # end for qd in hit_index[qc].keys()
            
            if len(db_indices_to_extract) == 0:
                logger.info("Query chain " + qc + ": all per-domain hits are single-domain entries in the database. Extended full-length search not possible for this chain.")
                continue
            
            # get entries from the index
            db_indices_to_extract = list(set(db_indices_to_extract))
            all_db_indices_to_extract[qc] = db_indices_to_extract
            if mode == 'exhaustive_tmalign':
                if target_db['faiss']:
                    extract_ids = []
                    extract_coords = []
                    extract_seqs = []

                    with open(index_names_fname, 'rb') as idf:
                        idmm = mmap.mmap(idf.fileno(), 0, access=mmap.ACCESS_READ)
                        extract_ids = retrieve_names_by_idx(idx=db_indices_to_extract, mm=idmm, use_sorting=False)

                    with open(sifname, 'rb') as sif, open(sdfname, 'rb') as sdf:
                        simm = mmap.mmap(sif.fileno(), 0, access=mmap.ACCESS_READ)
                        sdmm = mmap.mmap(sdf.fileno(), 0, access=mmap.ACCESS_READ)

                        startend = retrieve_start_end_by_idx(idx=db_indices_to_extract, mm=simm)
                        for start, end in startend:
                            extract_seqs.append(retrieve_bytes(start, end, mm=sdmm, typeconv=lambda x: x.decode('ascii')))
                    
                    with open(cifname, 'rb') as cif, open(cdfname, 'rb') as cdf:
                        cimm = mmap.mmap(cif.fileno(), 0, access=mmap.ACCESS_READ)
                        cdmm = mmap.mmap(cdf.fileno(), 0, access=mmap.ACCESS_READ)

                        startend = retrieve_start_end_by_idx(idx=db_indices_to_extract, mm=cimm)
                        for start, end in startend:
                            extract_coords.append(retrieve_bytes(start, end, mm=cdmm, typeconv=coord_conv))
                
                    index_entries_to_align = zip(extract_ids, extract_coords, extract_seqs)
                else: # not faiss db
                    index_entries_to_align = [ dbindex[i] for i in db_indices_to_extract ] # each elem is a tuple (name:str, coords:np.array with shape (N,3), sequence:str)
                
                # now, write out all the pdbs we need for alignment. first queries, then targets
               
                for qd in hit_index[qc].keys():
                    hit_index[qc][qd]['fname'] = write_pdb(tmp, hit_index[qc][qd]['coords'], hit_index[qc][qd]['seq'], name="FSQUERY-"+qd)
                
                target_fnames = list()
                for entry in index_entries_to_align:
                    target_fnames.append( write_pdb(tmp, entry[1], entry[2], name='FSTARGET-'+entry[0]) )
                return
                # for each query domain in the current query chain, run tm-align against each of these index entries
                for qf in [ x for x in ... ]:
                    
                    for tf in target_fnames:
                        tm_output = run_tmalign(qf, tf, options='-fast' if fastmode else None, keep_pdbs=True)
                        max_tm = max(tm_output['qtm'], tm_output['ttm'])
                        
            elif mode == "embscore": # bonus to scores mode
                # we have a list of hits per query chain. 
                # since we search the db in batch, just collect all the db refs/indices for the current chain

                pass
            else:
                logger.error("Unrecognised mode: "+ mode)
                sys.exit(1)


