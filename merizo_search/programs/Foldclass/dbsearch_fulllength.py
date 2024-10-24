#!/usr/bin/env python

import os
import sys
import shutil
import logging
import re
import mmap
from itertools import repeat, product
from multiprocessing import Pool, cpu_count

import numpy as np
import torch

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
)

from .dbsearch import *

domain_field_set_separator = ','
domain_field_separator = ':'

logger = logging.getLogger(__name__)

def domid2chainid_fn(x):
    # given 'cath-dompdb/2pi4A04.pdb' return '2pi4A'
    # given 'xxx/AF-Q93009-F1-model_v4_TED02.pdb' return 'AF-Q93009-F1-model_v4'
    return re.sub("[0-9]{2}$", "", os.path.basename(x).rstrip('.pdb')).rstrip('_').removesuffix('_TED')


def run_tmalign2(args):
    x, y, options, keep_pdbs = args
    return run_tmalign(x, y, options, keep_pdbs)


def parallel_fill_tmalign_array(qfnames:list[str], 
                                tfnames:list[str], 
                                ncpu:int=-1, 
                                mintm:float=0.5, 
                                options:str=None, 
                                keep_pdbs:bool=False
                                ):

    nrow = len(qfnames)
    ncol = len(tfnames)

    if ncpu <= 0: # wiseguy eh
        ncpu = min(nrow*ncol, cpu_count())
    
    # Create lists of all (i, j) combinations
    tm_args = [(qfname, tfname, options, keep_pdbs) for qfname in qfnames for tfname in tfnames]

    with Pool(ncpu) as pool:
        results = pool.map(run_tmalign2, tm_args)

    tmalign_scores = [ max(d['qtm'], d['ttm']) for d in results ]
    tmalign_scores = np.asarray(tmalign_scores).reshape((nrow,ncol))

    tmalign_scores[tmalign_scores < mintm] = 0.0

    return tmalign_scores


def tmalign_submatrix_to_hits(mtx: np.array, qc:str, hc:str, qds: list[str], hds: list[dict] ):
    result = list()
    nqd, nhd = mtx.shape

    assert len(qds) == nqd
    assert len(hds) == nhd

    # filter out obvious no-hit cases
    # if any row is all zero, there is no multi-domain hit in this submatrix.
    empty_rows = np.where(~mtx.any(axis=1))[0]
    if len(empty_rows) > 0:
        return result
    
    # The number of columns with at least one nonzero entry has to be >= nqd.
    nonzero_cols = np.where(mtx.any(axis=0))[0]
    if len(nonzero_cols) < nqd:
        return result
    
    # these are also indices into `qds` and `hds`
    nz_row_i, nz_col_i = np.nonzero(mtx)

    # hds, but just reorganised into a format for convenient enumeration
    hd_lists = [list() for _ in range(nqd)]
    hd_indices = [list() for _ in range(nqd)]

    for i,j in zip(nz_row_i, nz_col_i):
        hd_lists[i].append(hds[j])
        hd_indices[i].append(j)
        
    # finally, enumerate all valid paths through these lists
    hd_paths = product(*hd_indices)

    # each element of hd_paths is a tuple with `nqd` entries. 
    # Use these to produce the final result for this hc.

    # match_cat values: 
    # 0 : bag-of-domains match
    # 1 : gapped alignment (order preserved), both interstitial and end gaps OK
    # 2 : gapped alignment (order preserved), end gaps OK, but no interstitial gaps
    # 3 : exact MDA match; nqd==nhd, order preserved, no gaps at all

    for path in [ list(p) for p in hd_paths]:
        # a path is not valid if it has a repeated element 
        # (multiple qd matching the same hd)
        if len(set(path)) != nqd:
            continue

        match_cat = 0 
        if sorted(path) == path:
            if nqd == nhd:
                match_cat = 3
            elif np.all(np.diff(path) == 1):
                match_cat = 2
            else:
                match_cat = 1
        
        # create domain-wise matching info
        match_info = list()
        match_metadata = list()
        for qdi in range(len(path)): # always goes from 0 to nqd-1
            curr_hit = hds[ path[qdi] ]
            match_info.append(
                domain_field_separator.join([ qds[qdi], curr_hit['hd'], str(mtx[qdi, path[qdi] ]) ])
                )
            match_metadata.append(curr_hit['hm'])
        
        result.append(
            (qc, nqd, hc, nhd, match_cat,  
             domain_field_set_separator.join(match_info),
             '['+domain_field_set_separator.join(match_metadata)+']', # JSON array
            )
            )
    
    return result

    
def multi_domain_search(queries:list, # if list[str], treat as filenames, if list[dict], each dict has at least 'coords', 'seq', 'name'
                       search_results,
                       db_name,
                       tmp_root: str,
                       device: torch.device,
                    #    topk: int=1,
                       fastmode: bool=False, 
                       threads: int=-1,
                    #    mincos: float=0.5,
                       mintm: float=0.5,
                    #    mincov: float=0.0,
                    #    inputs_are_ca: bool=False,
                    #    search_batchsize:int=262144,
                    #    search_type='IP',
                       inputs_from_easy_search=False,
                       mode="exhaustive_tmalign",
                       pdb_chain=None
                       ):
    """
    TODO list:
    - implement embscore mode
    - refactor for clarity and test
    """

    """
    in easy-search: domain names are suffixed with _merizo_01 etc.
    in search: could be anything.
    So, in 'search' mode, we have to treat all queries as coming from one chain.
        in 'easy-search' mode, we map domains to the original query chains.
    """
    nq = len(queries)
    if nq == 1: # regardless of the state of inputs_from_easy_search
        logger.warning("Cannot execute multi-domain search with only one query domain.")
        return None
    
    if not inputs_from_easy_search:
        if pdb_chain:
            pdb_chain = pdb_chain.rstrip(",")
            pdb_chains = pdb_chain.split(",")
        else:
            pdb_chains = ["A"] * nq
        query_dicts = list()
        for i in range(nq):
            # Read coords and seq from PDB file
            query_dicts.append(read_pdb(pdbfile=queries[i], pdb_chain=pdb_chains[i]))
            
        queries = query_dicts

    # extract potential chains for multi-domain matching
    logger.info('Start multi-domain search...')
    all_query_domains = list() # merizo-given names for the query domains in dbsearch().

    qd_coords_seqs = dict()
    for qdd in queries:
        qd_coords_seqs[os.path.basename(qdd['name'])] = {"coords": qdd['coords'], "seq":qdd['seq'] }
        all_query_domains.append(os.path.basename(qdd['name']).removesuffix(".pdb"))

    if inputs_from_easy_search:
        all_query_chains = [ re.sub("_merizo_[0-9]*$", "", x) for x in all_query_domains ]
    else:
        # treat all query structures as single domains comaing from a single chain
        all_query_chains = [ 'A' for _ in all_query_domains ]

    initial_hit_index = dict()
    ihi_hit_chain_info = dict()   # ihi = initial hit index

    query_dom2chain_lookup = dict()
    for qc,qd in zip(all_query_chains, all_query_domains):
        if qc not in initial_hit_index.keys():
            initial_hit_index[qc] = dict()
        if qd not in initial_hit_index[qc].keys():
            initial_hit_index[qc][qd] = dict()
        
        query_dom2chain_lookup[qd] = qc

        initial_hit_index[qc][qd]['qcoords'] = qd_coords_seqs[qd]['coords']
        initial_hit_index[qc][qd]['qseq'] = qd_coords_seqs[qd]['seq']
        initial_hit_index[qc][qd]['hits'] = list() # this will hold per-hit-domain information

    for hitdict in search_results:
        for hit in hitdict.values():
            qd = hit['query']
            qc = query_dom2chain_lookup[qd]
            hd = hit['target']
            hc = domid2chainid_fn(hd)
            hi = int(hit['dbindex'])
            initial_hit_index[qc][qd]['hits'].append({'hc': hc, 'hd': hd, 'hi': hi})

    qcs = list(initial_hit_index.keys())

    for qc in qcs:
        # all results should be determined *per query chain*
        num_query_domains = len(list(initial_hit_index[qc].keys()))
        
        if num_query_domains == 0:
            logger.info('Query chain '+qc+':no domains detected, skipping multi-domain search.')
            del initial_hit_index[qc]
            continue

        if num_query_domains == 1:
            logger.info('Query chain '+qc+': Only one detected domain, so multi-domain hits are same as those in the per-domain search results.')
            del initial_hit_index[qc]
            continue
        
        # This check is actually meaningless, e.g. for repeat proteins
        """
        # hit_chains_per_query_domain = dict()
        
        # for qd in initial_hit_index[qc].keys():
        #     hit_chains_per_query_domain[qd] = set( [ v['hc'] for v in initial_hit_index[qc][qd]['hits'] ] )
            
        # full_length_hit_chains = set.intersection( *hit_chains_per_query_domain.values() )
        
        # if len(full_length_hit_chains) == 0:
        #     logger.info("Query chain "+ qc +": No multi-domain hits found with k = "+str(topk)+", maybe try again with a higher value of -k.")
        # else:
        #     nint = len(full_length_hit_chains)
        #     logger.info("Query chain " + qc + ": "+ str(nint) + " multi-domain hits found in top-k hit lists.")
        """
                
    target_db = read_database(db_name=db_name, device=device)
    # construct db_indices_to_extract by checking db ids in the vicinity of the hit indices.
    all_db_indices_to_extract = dict()

    if target_db['faiss']:
        # AFDB/TED
        dbinfofname = target_db['database']
        dbinfo = read_dbinfo(dbinfofname)
        db_dir = os.path.dirname(dbinfofname)
        index_names_fname = os.path.join(db_dir, dbinfo['db_names_f'])
        # db_domlengths_fname = os.path.join(db_dir, dbinfo['db_domlengths_f'])
        sifname = os.path.join(db_dir, dbinfo['sif'])
        sdfname = os.path.join(db_dir, dbinfo['sdf'])
        cifname = os.path.join(db_dir, dbinfo['cif'])
        cdfname = os.path.join(db_dir, dbinfo['cdf'])
        if 'mif' and 'mdf' in dbinfo.keys():
            mifname = os.path.join(db_dir, dbinfo['mif'])
            mdfname = os.path.join(db_dir, dbinfo['mdf'])

            mif = open(mifname, 'rb')
            mdf = open(mdfname, 'rb')
            mimm = mmap.mmap(mif.fileno(), 0, access=mmap.ACCESS_READ)
            mdmm = mmap.mmap(mdf.fileno(), 0, access=mmap.ACCESS_READ)
        else:
            mimm = mdmm = None
            metadata = '{ }'
    else:
        # pytorch db
        dbindex = target_db['index'] # already parsed pickle
        if target_db['mdfn'] is not None and target_db['mifn'] is not None:
            mifname = target_db['mifn']
            mdfname = target_db['mdfn']

            mif = open(mifname, 'rb')
            mdf = open(mdfname, 'rb')
            mimm = mmap.mmap(mif.fileno(), 0, access=mmap.ACCESS_READ)
            mdmm = mmap.mmap(mdf.fileno(), 0, access=mmap.ACCESS_READ)
        else:
            mimm = mdmm = None
            metadata = '{ }'

    final_mda_results = list()
    qcs = list(initial_hit_index.keys())

    for qc in qcs:
        if len(initial_hit_index[qc].keys()) < 2:
            logger.info("Skipping query chain "+qc+" as it has only one detected domain.")
            del initial_hit_index[qc]
            continue

        logger.info("Build potential target list for query chain "+qc)
        db_indices_to_extract = []
        nqd = len(initial_hit_index[qc].keys()) # number of domains in current qc
        for qd in initial_hit_index[qc].keys():
            for hit in initial_hit_index[qc][qd]['hits']:
                anchor_index = hit['hi']
                anchor_chain = hit['hc'] # chain name we are looking for
                
                curr_idx_list = [] # indices to extract
                db_entry_is_multidomain = False
                if target_db['faiss']:
                    with open(index_names_fname, 'rb') as idf:
                        idmm = mmap.mmap(idf.fileno(), 0, access=mmap.ACCESS_READ)  
                        # left
                        cur_i = anchor_index
                        while( domid2chainid_fn(retrieve_names_by_idx([cur_i-1], idmm)[0]) == anchor_chain):
                            curr_idx_list.append(cur_i-1)
                            cur_i-=1
                            db_entry_is_multidomain = True
                        # right
                        cur_i = anchor_index
                        while( domid2chainid_fn(retrieve_names_by_idx([cur_i+1], idmm)[0]) == anchor_chain):
                            curr_idx_list.append(cur_i+1)
                            cur_i+=1
                            db_entry_is_multidomain = True
                else: # not faiss db
                    # left
                    cur_i = anchor_index
                    while( domid2chainid_fn(dbindex[cur_i-1][0]) == anchor_chain):
                        curr_idx_list.append(cur_i-1)
                        cur_i-=1
                        db_entry_is_multidomain = True
                    # right
                    cur_i = anchor_index
                    while( domid2chainid_fn(dbindex[cur_i+1][0]) == anchor_chain):
                        curr_idx_list.append(cur_i+1)
                        cur_i+=1
                        db_entry_is_multidomain = True
                
                if db_entry_is_multidomain:
                    curr_idx_list.append(anchor_index)
                
                nhd = len(curr_idx_list)

                if nhd >= nqd : # don't bother if the hit chain has fewer domains than the query
                    curr_idx_list.sort()
                    db_indices_to_extract.extend(curr_idx_list)
                    
                    ihi_hit_chain_info[anchor_chain] = {'nhd':nhd, 
                                                'all_db_indices': curr_idx_list,
                                                # 'metadata_cat': ';'.join([m for m in metadata])
                                                }
                
            # end for hit in hit_index[qc][qd]['hits']
        # end for qd in hit_index[qc].keys()
        
        if len(db_indices_to_extract) == 0:
            logger.info("Query chain " + qc + ": Chains for all per-domain hits in the database have fewer domains than the query. Multi-domain search not possible for this chain.")
            logger.info("Maybe try increasing -k, or use segmented domain structures as queries to `search` with `--multi_domain_search` enabled.")
            del initial_hit_index[qc]
            continue
        
        db_indices_to_extract = list(set(db_indices_to_extract))
        all_db_indices_to_extract[qc] = db_indices_to_extract

    if mode == 'exhaustive_tmalign':

        for qc in initial_hit_index.keys():

            tmp = os.path.join(tmp_root, 'MD_search_structures_'+qc)
            os.makedirs(tmp, exist_ok=True)

            nqd = len(initial_hit_index[qc].keys()) # number of domains in current qc
            if target_db['faiss']:
                extract_ids = []
                extract_coords = []
                extract_seqs = []

                with open(index_names_fname, 'rb') as idf:
                    idmm = mmap.mmap(idf.fileno(), 0, access=mmap.ACCESS_READ)
                    extract_ids = retrieve_names_by_idx(idx=all_db_indices_to_extract[qc], mm=idmm, use_sorting=False)

                with open(sifname, 'rb') as sif, open(sdfname, 'rb') as sdf:
                    simm = mmap.mmap(sif.fileno(), 0, access=mmap.ACCESS_READ)
                    sdmm = mmap.mmap(sdf.fileno(), 0, access=mmap.ACCESS_READ)

                    startend = retrieve_start_end_by_idx(idx=all_db_indices_to_extract[qc], mm=simm)
                    for start, end in startend:
                        extract_seqs.append(retrieve_bytes(start, end, mm=sdmm, typeconv=lambda x: x.decode('ascii')))
                
                with open(cifname, 'rb') as cif, open(cdfname, 'rb') as cdf:
                    cimm = mmap.mmap(cif.fileno(), 0, access=mmap.ACCESS_READ)
                    cdmm = mmap.mmap(cdf.fileno(), 0, access=mmap.ACCESS_READ)

                    startend = retrieve_start_end_by_idx(idx=all_db_indices_to_extract[qc], mm=cimm)
                    for start, end in startend:
                        extract_coords.append(retrieve_bytes(start, end, mm=cdmm, typeconv=coord_conv))

                if mimm is not None:
                    startend = retrieve_start_end_by_idx(idx=all_db_indices_to_extract[qc], mm=mimm)
                    metadata = list()
                    for start, end in startend:
                        metadata.append(retrieve_bytes(start, end, mm=mdmm, typeconv=lambda x: x.decode('ascii')))
                else:
                    metadata = repeat(metadata) # repeat('{ }')
            
                index_entries_to_align = list(zip(extract_ids, extract_coords, extract_seqs, all_db_indices_to_extract[qc], metadata)) # leaving it as a zip generator means we can only iterate over it once
            else: # not faiss db
                index_entries_to_align1 = [ dbindex[i] for i in all_db_indices_to_extract[qc] ] # each elem is a tuple (name:str, coords:np.array with shape (N,3), sequence:str)
                index_entries_to_align = []
                for i, entry in zip(all_db_indices_to_extract[qc], index_entries_to_align1):
                    if mimm is not None:
                        startend = retrieve_start_end_by_idx(idx=[i], mm=mimm)

                        for start, end in startend:
                            metadata = retrieve_bytes(start, end, mm=mdmm, typeconv=lambda x: x.decode('ascii'))

                    index_entries_to_align.append( (os.path.basename(entry[0]).replace('.pdb',''), entry[1], entry[2], i, metadata) )
                del index_entries_to_align1
            
            # now, write out all the pdbs we need for alignment. first queries, then targets

            for qd in initial_hit_index[qc].keys():
                initial_hit_index[qc][qd]['qfname'] = write_pdb(tmp, initial_hit_index[qc][qd]['qcoords'], 
                                                                initial_hit_index[qc][qd]['qseq'], 
                                                                name="FSQUERY-"+qd
                                                                )
            
            target_fnames = list()
            for entry in index_entries_to_align:
                target_fnames.append( write_pdb(tmp, entry[1], entry[2], name='FSTARGET-'+entry[0]) )

            # for each query domain in the current query chain, run tm-align against each of these index entries
            # exh_results = dict() # keys are qd, not qc.
            nqd = len(initial_hit_index[qc])
            # n_targets_to_align = len(target_fnames)

            # parallel version
            logger.info("Run TM-align for all query-hit combinations, query domains from query chain "+qc+"...")
            tmscore_mtx = parallel_fill_tmalign_array(qfnames=[ initial_hit_index[qc][qd]['qfname'] for qd in initial_hit_index[qc].keys() ], 
                                                        tfnames=target_fnames, 
                                                        ncpu=threads, 
                                                        mintm=mintm, 
                                                        options='-fast' if fastmode else None, 
                                                        keep_pdbs=True # this is sent to individual TM-align runs; we delete the files after this step
                                                        )
            # serial version
            """
            # logger.info('start serial tmalign...')
            # tmscore_mtx2 = np.zeros(shape=(nqd, n_targets_to_align))
            # for qi, qd in enumerate(initial_hit_index[qc].keys()):
            #     exh_results[qd] = list() # each element is a dict with keys: hd, hc, maxtm
            #     qfn = initial_hit_index[qc][qd]['qfname']
            #     for ti, tfn in enumerate(target_fnames):
            #         tm_output = run_tmalign(qfn, tfn, options='-fast' if fastmode else None, keep_pdbs=True)
            #         max_tm = max(tm_output['qtm'], tm_output['ttm'])
            #         if max_tm >= mintm:
            #             # retain hit in hitlist for this query domain.
            #             hd = index_entries_to_align[ti][0]
            #             hi = index_entries_to_align[ti][3]
            #             exh_results[qd].append({'hd': hd,
            #                                     'hc': domid2chainid_fn(hd),
            #                                     'hi': hi,
            #                                     'maxtm': max_tm,
            #                                     # 'metadata': metadata
            #                                     })
            #             tmscore_mtx2[qi, ti] = max_tm
            
            # logger.info(str(np.array_equal(tmscore_mtx, tmscore_mtx2)))
            """
            shutil.rmtree(tmp)
            # find sets of column indices in tmscore_mtx corresponding to each hc
            hd_names = np.asarray([ x[0] for x in index_entries_to_align ])
            hc_per_hd = np.asarray([ domid2chainid_fn(x) for x in hd_names ])
            hit_info = list()

            for c in range(len(index_entries_to_align)):
                hd = index_entries_to_align[c][0]
                hc = domid2chainid_fn(hd)
                hi = index_entries_to_align[c][3]
                hm = index_entries_to_align[c][4] # metadata
                hit_info.append(
                    {
                        'hd': hd, # domain name (str)
                        'hc': hc, # chain name (str)
                        'hi': hi, # db index (int64)
                        'hm': hm, # metadata (str)
                    }
                )
            # hit_info = np.asarray(hit_info)
            qds = list(initial_hit_index[qc].keys()) 
            for hc in np.unique(hc_per_hd):
                mtx_col_i = np.where(hc_per_hd == hc)[0]
                hc_hit_info = [ hit_info[i] for i in mtx_col_i ]
                subresult = tmalign_submatrix_to_hits(tmscore_mtx[:,mtx_col_i], 
                                                        qc=qc, 
                                                        hc=hc, 
                                                        qds=qds, 
                                                        hds=hc_hit_info
                                                        )
                if len(subresult) > 0:
                    final_mda_results.extend( subresult )
            logger.info("Finished multi-domain search for query chain "+qc+".")
        return final_mda_results                           
    # elif mode == "embscore": # bonus to scores mode
    #     # we have a list of hits (hc and hd) per qc and qd.
    #     # We also know how many domains are in each hc as we've collected their indices.
    #     # Compute the score corrections for all domains in each hc.
    #     # Score corrections per hd are based on how many domains from that hc match the qd's in the qc.
    #     # Must be careful to apply these corrections per query and not let them 'bleed over' 
    #     # or accumulate as a result of also matching qd's in other qc's.
        
    #     if target_db['faiss']:
    #         # since we search the db in batch, just collect all the db refs/indices for the current chain
    #         # Apply score corrections per query and per db iterator batch; a single ResultHeap can still be used                    
    #         pass
    #     else: # not faiss db
    #         pass
    else:
        logger.error("Unrecognised multi-domain search mode: "+ mode)
        sys.exit(1)

