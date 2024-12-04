# Merizo-search

Merizo-search is a method that builds on the original Merizo (Lau et al., 2023) by combining state-of-the-art domain segmentation with fast embedding-based searching. Specifically, Merizo-search makes use of an EGNN-based method called Foldclass, which embeds a structure and its sequence into a fixed size 128-length vector. This vector is then searched against a pre-encoded library of domains, and the top-K matches in terms of cosine similarity are used for confirmatory TM-align runs to validate the search. Merizo-search also supports searching larger-than-memory databases of embeddings using the Faiss library.

## Installation

#### Using conda with GPU support (Recommended):

```
cd /path/to/merizo_search
conda create -n merizo_search python=3.9
conda activate merizo_search
pip install -r merizo_search/programs/Merizo/requirements.txt
conda install -c pytorch -c nvidia faiss-gpu
```
For the CPU-only version of Faiss, replace the last step with `conda install faiss-cpu`. A GPU provides only minor speedups for searching with Faiss, but is beneficial when segmenting and/or embedding many structures.

We recommend using conda as there is no official Faiss package on PyPI the time of writing. Unofficial packages are available; use these at your own risk.


## Databases
We provide pre-built Foldclass databases for domains in CATH 4.3 and all 365 million+ domains from TED. They can be obtained from [here](https://doi.org/10.5522/04/26348605). We recommend using our convenience script in this repository (`download_dbs.sh`) to download them.


## Usage

Merizo-search supports the functionalities listed below:
```
segment         Runs standard Merizo segment on a multidomain target.
search          Runs Foldclass search on a single input PDB against a Foldclass database.
easy-search     Runs Merizo to segment a query into domains and then searches against a Foldclass database.
createdb        Creates a Foldclass database given a directory of PDB files. 
```
### `segment`

The `segment` module of Merizo can be used to segment a multidomain protein into domains and can be run using: 
```
python merizo.py segment <input.pdb> <output_prefix> <options>

# Example:
python merizo.py segment ../examples/*.pdb results --iterate
```

The `-h` flag can be used to show all options for `segment`. The input PDB can be a single PDB, or multiple, including something like `/dir/*.pdb`. The `output_prefix` will be appended with `_segment.tsv` to indicate the results of `segment`. 

The `--iterate` option can sometimes be used to generate a better segmentation result on longer models, e.g. AlphaFold models.

The `--pdb_chain` option lets you select which PDB chain will be analysed. If not provided, chain `A` is assumed. If supplying multiple structures as queries, you can supply either a single chain ID to be used for all queries, or a comma-separated list of chain IDs, e.g. `A,A,B,D,A`. 

This will print:
```
2024-03-10 19:43:00,945 | INFO | Starting merizo segment with command:

merizo_search/merizo.py segment examples/AF-Q96HM7-F1-model_v4.pdb examples/AF-Q96PD2-F1-model_v4.pdb results --iterate

2024-03-10 19:44:11,318 | INFO | Finished merizo segment in 70.37289953231812 seconds.
```

Results will be written to `results_segment.tsv`:
```
filename        nres    nres_dom        nres_ndr        ndom    pIoU    runtime result
AF-Q96PD2-F1-model_v4   775     383     392     3       0.4942  0.7174  71-189,190-290,291-453
AF-Q96HM7-F1-model_v4   432     267     165     1       0.6343  0.3958  1-267
3w5h    272     272     0       2       1.0000  0.2517  1001-1117,1118-1272
M0      31      0       31      0       0.0000  0.0225 
```

### `search`

The `search` module of Merizo will call Foldclass to search queries (as they are, without segment) against a pre-compiled database (created using `createdb`). This is useful when queries are already domains. 

The `search` module is called using:
```
python merizo.py search <input.pdb> <database_name> <output_prefix> <tmp> <options>
```
Again, the `-h` option will print all options that can be given to the program. The `database_name` argument is the prefix of a Foldclass database. A Foldclass database can be created using `createdb`.

For default Foldclass databases, database_name should be the basename of the database without `.pt` or `.index`. For example:
```
python merizo.py search ../examples/AF-Q96HM7-F1-model_v4.pdb ../examples/database/cath results tmp
```
For Faiss databases, use the basename of the `.json` file without extension:
```
python merizo.py search ../examples/AF-Q96HM7-F1-model_v4.pdb ../examples/database/ted100 results tmp
```

Results will be written to `results_search.tsv`:
```
query   topk_rank   target  cosine_similarity   q_len   t_len   len_ali seq_id  q_tm    t_tm    max_tm  rmsd
AF-Q96HM7-F1-model_v4	0	3.40.50.10540__SSG5__1_1	0.8204	432	304	169	0.1120	0.2646	0.3470	0.3470	6.27
```

Output fields are configurable using the `--format` flag which allows the section of different fields, specified as a comma-separated list. The defulat is to output all fields: `query,chopping,conf,plddt,emb_rank,target,emb_score,q_len,t_len,ali_len,seq_id,q_tm,t_tm,max_tm,rmsd,metadata`.

### `easy-search`

`easy-search` combines `segment` and `search` into a single workflow. A multidomain query is parsed using `segment`, and the resultant domains are searched against a database using `search`. This can be called using:
```
python merizo.py search <input.pdb> <database_name> <output_prefix> <tmp> <options>

# Example:
python merizo.py easy-search ../examples/AF-Q96HM7-F1-model_v4.pdb ../examples/database/cath results tmp --iterate
```

As with `search`, the `-h` option will print all options that can be given to the program. The `database_name` argument is the prefix of a Foldclass database, as above. A Foldclass database can be created using `createdb`. 

The results in the `_search.tsv` file will be different to that of `search` and will show extra information about the domain parse:
```
query_dom   chopping    conf    plddt   topk_rank   target  cosine_similarity   q_len   t_len   len_ali seq_id  q_tm    t_tm    max_tm  rmsd
AF-Q96HM7-F1-model_v4_merizo_01	1-267	1.0000	91.9215	0	3.40.50.720__SSG5__79_12	0.8583	267	178	147	0.0680	0.3811	0.5180	0.5180	4.95
```

As with `segment`, the `_segment.tsv` file will show the results of `segment`:
```
query   nres    nres_domain nres_non_domain num_domains conf    time_sec    chopping
AF-Q96HM7-F1-model_v4	432	267	165	1	0.6343	22.7448	1-267
```

Output fields are configurable using the `--format` flag which allows the section of different fields: `query, target, conf, plddt, chopping, emb_rank, emb_score, q_len, t_len, ali_len, seq_id, q_tm, t_tm, max_tm, rmsd`.

### `createdb`

`createdb` can be used to create a standard Foldclass database given a directory of PDB structures (anything with the extension `.pdb` will be read automatically). This can be run using:
```
python merizo.py createdb <directory_containing_pdbs> <output_database_prefix>

# Example:
python merizo_search/merizo.py createdb examples/database/cath_pdb examples/database/cath
```

The argument given to `output_database_prefix` will be appended with `.pt` and `.index`, with the two files constituting a Foldclass database. 

The `.pt` file is a Pytorch tensor containing the embedding representation of the PDB files.
The `.index` file contains the PDB names, CA coordinates and the sequences of the input PDBs.


### Multi-domain searching

Both `search` and `easy-search` support searching for database entries that match all domains in a query. In the case of `search`, all supplied query structures are considered as originating from a single chain and searched against the database. In the case of `easy-search`, segmentation and multi-domain search operate on a per-query-chain basis, that is, only domains segmented from a single query chain are searched as a set.

To enable multi-domain searching, add the option `--multi_domain_search` to a `search` or `easy-search` command. A few important things to note:
- In multi-domain searches, `-k` still controls the maximum number of per-domain hits retrieved using vector search. We recommend setting it to around 100.
- We only keep hits where _all_ domains in each query chain are matched at least once in a hit chain. We don't return hits containing fewer domains than the query domain set. You can, however, manually supply a subset of pre-segmented domains to the `search` command.
- The accuracy of multi-domain `easy-search` runs is dependent on the accuracy of the initial Merizo segmentation. If you're not getting many meaningful hits, we recommend checking the output from the implicit `segment` step from your run. You may wish to manually segment your query chain and then re-run multi-domain search using the `search` module. 

Multi-domain hits are categorised into one of 4 categories in the `match_category` field of the output, representing the type of multi-domain match. Each can be seen as a subset of the last:
```
0 : Unordered domain match: all query domains present in hit chain, but in different sequential order to query chain
1 : Discontiguous domain match: All query domains matched in sequential order, but hit chain may have extra domains in interstitial and/or end positions
2 : Contiguous domain match: All query domains matched in sequential order. Hit chain may have extra domains at ends, but not in interstitial positions
3 : Exact multi-domain architecture (MDA) match: query chain and hit chain correspond at domain level without domain rearrangement or indels
```

## Other outputs

The `segment` module used in `segment` and `easy-search` produces a number of different output files that can be turned on using various flags:
```
--save_domains  Save the domains as individual PDBs.
--save_pdb      Save a single PDB with the occupancy column replaced with domain IDs. (Visualise in PyMOL using the `spectrum q` command).
--save_pdf      Save a PDF output showing the domain map.
--save_fasta    Save the sequence of the input file.
```

By default, all output files will be saved alongside the original input query PDB, but they can be saved into a folder given by `--merizo_output`.
