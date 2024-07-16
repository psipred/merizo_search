#!/usr/bin/env python

import os
import sys
import json
import numpy as np
import mmap

def name_conv(l):
    return [ x.decode().rstrip() for x in l ]


def domlen_conv(l):
    return [ np.frombuffer(x, dtype='uint16')[0] for x in l ]


def idx_to_chunk_idx(idx, chunksize):
    chunk_num = idx // chunksize
    idx_in_chunk = idx % chunksize
    #print('chunk:', chunk_num, ', idx in chunk:',idx_in_chunk)
    return (chunk_num, idx_in_chunk)


def read_dbinfo(dbinfo_path):
    return json.load(open(dbinfo_path, 'r'))


def db_memmap(filename:str, shape:tuple):
    fp = np.memmap(filename, dtype='float32', mode='r', shape=shape)
    return fp


def db_iterator(embeddings, batch_size: int):
    for i0 in range(0, embeddings.shape[0], batch_size):
        yield embeddings[i0:i0 + batch_size]

def startend_memmap(filename:str, n_entries:int):
    fp = np.memmap(filename, dtype='int64', mode='r', shape=(n_entries, 2))
    return fp

def names_memmap(filename:str, n_entries:int):
    fp = np.memmap(filename, dtype='S33', mode='r', shape=n_entries)
    return fp

        
# def retrieve_names_by_idx(idx, mm, offset=33, use_sorting=True):
#     # offset is 32+1 by default; 32 for max id width, +1 for the \n in the plaintext version
#     # with open(index_names_fname, 'rb') as f:
#     #     mm = mmap.mmap(f.fileno(), 0) # 0 means mmap the whole file

#         if use_sorting:
#             # sort idx and retain original order too
#             idx = np.asarray(idx)
#             order = np.argsort(idx)

#             sorted_idx = idx[order]
#             sorted_names = list()
#             for i in sorted_idx:
#                 mm.seek( offset * i )
#                 sorted_names.append(mm.readline().decode().rstrip())

#             sorted_names = np.asarray(sorted_names)
#             return sorted_names[ np.argsort(order) ]
        
#         else:
#             sorted_names = list()
#             for i in idx:
#                 mm.seek( offset * i )
#                 sorted_names.append(mm.readline().decode().rstrip())

#             return np.asarray(sorted_names)


def retrieve_mmdata_by_idx(idx, mm, use_sorting=False, offset=np.dtype('uint16').itemsize, typeconv=None):
    # offset is used as-is; we don't assume any `\n`s.
    # typeconv must operate on a *list* of raw byte sequences.
    if use_sorting:
        # sort idx and retain original order too
        idx = np.asarray(idx)
        order = np.argsort(idx)

        sorted_idx = idx[order]
        sorted_data = list()
        for i in sorted_idx:
            mm.seek( offset * i )
            sorted_data.append(mm.read(offset))
        
        if typeconv is None:
            sorted_data = np.asarray(sorted_data)
        else:
            sorted_data = np.asarray(typeconv(sorted_data))

        return sorted_data[ np.argsort(order) ]
    
    else:
        sorted_data = list()
        for i in idx:
            mm.seek( offset * i )
            sorted_data.append(mm.read(offset))

        if typeconv is not None:
            sorted_data = typeconv(sorted_data)

        return np.asarray(sorted_data)


def retrieve_names_by_idx(idx, mm, use_sorting=False):
    return retrieve_mmdata_by_idx(idx, mm, use_sorting=use_sorting, offset=33, typeconv=name_conv)


def retrieve_names_by_idx2(idx, npmm, use_sorting=False):
    return npmm[idx]


def retrieve_domlen_by_idx(idx, mm, use_sorting=False):
    return retrieve_mmdata_by_idx(idx, mm, use_sorting=use_sorting, offset=np.dtype('uint16').itemsize, typeconv=domlen_conv)


def start_end_conv(l):
    result = []
    for x in l:
        result.append(np.frombuffer(x, dtype='int64'))

    return result


def retrieve_start_end_by_idx(idx, mm, use_sorting=False):
    return retrieve_mmdata_by_idx(idx, mm, use_sorting=use_sorting, offset=2*8, typeconv=start_end_conv)


def retrieve_bytes(start, end, mm, typeconv=None):
    mm.seek(start)
    nbytes = int(end-start)
    b = mm.read(nbytes)
    if typeconv is None:
        return b
    else:
        return typeconv(b)
    

def coord_conv(b):
    d = np.frombuffer(b, dtype='float32')
    assert len(d) % 3 == 0
    nres = len(d)//3
    return d.reshape((nres,3))


if __name__ == '__main__':

    dbinfofname = sys.argv[1]  # full path to json file

    hit_indices_f = sys.argv[2] # file with ints

    hit_indices = []

    with open(hit_indices_f, 'r') as f:
        for line in f:
            hit_indices.append(int(line.strip()))
    
    dbinfo = read_dbinfo(dbinfofname)
    db_dir = os.path.dirname(dbinfofname)
    n_db = dbinfo["DB_SIZE"]
    hit_indices.append(n_db-1)
    
    print("Read", len(hit_indices), "indices:")
    print(hit_indices)
    
    index_names_fname = os.path.join(db_dir, dbinfo['db_names_f'])
    # db_domlengths_fname = os.path.join(db_dir, dbinfo['db_domlengths_f'])
    sifname = os.path.join(db_dir, dbinfo['sif'])
    sdfname = os.path.join(db_dir, dbinfo['sdf'])
    cifname = os.path.join(db_dir, dbinfo['cif'])
    cdfname = os.path.join(db_dir, dbinfo['cdf'])
    
    # retrieve hit names (and seqlen), coords and seqs
    hit_ids = [] # list of str
    hit_seqs = [] # list of str
    hit_coords = [] # list of np.ndarray
    startends = [] # list of 2-ples
    
    with open(index_names_fname, 'rb') as idf:
        idmm = mmap.mmap(idf.fileno(), 0, access=mmap.ACCESS_READ)
        hit_ids = retrieve_names_by_idx(idx=hit_indices, mm=idmm, use_sorting=False)

    with open(sifname, 'rb') as sif, open(sdfname, 'rb') as sdf:
        simm = mmap.mmap(sif.fileno(), 0, access=mmap.ACCESS_READ)
        sdmm = mmap.mmap(sdf.fileno(), 0, access=mmap.ACCESS_READ)

        startend = retrieve_start_end_by_idx(idx=hit_indices, mm=simm)
        
        for start, end in startend:
            hit_seqs.append(retrieve_bytes(start, end, mm=sdmm, typeconv=lambda x: x.decode('ascii')))
     
    with open(cifname, 'rb') as cif, open(cdfname, 'rb') as cdf:
        cimm = mmap.mmap(cif.fileno(), 0, access=mmap.ACCESS_READ)
        cdmm = mmap.mmap(cdf.fileno(), 0, access=mmap.ACCESS_READ)

        startend = retrieve_start_end_by_idx(idx=hit_indices, mm=cimm)
        for start, end in startend:
            hit_coords.append(retrieve_bytes(start, end, mm=cdmm, typeconv=coord_conv))

    hit_lengths = list(map(len, hit_seqs))
    
    for i in range(len(hit_indices)):
        print('\t'.join(['{}']*3).format(hit_indices[i], hit_ids[i], hit_seqs[i]))
