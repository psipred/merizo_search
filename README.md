# Merizo-search

_Foldclass and Merizo-search: Scalable structural similarity search for single- and multi-domain proteins using geometric learning. [Bioinformatics](https://doi.org/10.1093/bioinformatics/btaf277)_

Merizo-search is a method that builds on the original Merizo (Lau et al., 2023) by combining state-of-the-art domain segmentation with fast embedding-based searching. Specifically, Merizo-search makes use of an EGNN-based method called Foldclass, which embeds a structure and its sequence into a fixed size 128-length vector. This vector is then searched against a pre-encoded library of domains, and the top-_k_ matches in terms of cosine similarity are used for confirmatory TM-align runs to validate the search. Merizo-search also supports searching larger-than-memory databases of embeddings using the Faiss library.

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

We recommend using conda as there is no official Faiss package on PyPI the time of writing. Unofficial packages are available; use these at your own risk (see Ansible Installation below).


## Ansible Installation

> [!NOTE]
> The Ansible installation uses `virtualenv` rather than `conda`, and therefore uses an unofficial, third-party-maintained Faiss package from PyPI. This installation method is provided for convenience and is at your own risk.

First ensure that Ansible is installed on your system, then clone this GitHub repo. 

``` bash
pip install ansible
git clone https://github.com/psipred/merizo_search.git
cd merizo_search/ansible_installer
```

Next, edit `config_vars.yml` to reflect where you would like Merizo-search and its underlying data to be installed.

You can now run ansible using

``` bash
ansible-playbook -i hosts install.yml
```

You can edit the `hosts` file to install Merizo-search on one or more machines. This ansible installation creates a python virtualenv called `merizosearch_env`, which the program needs to run. You can activate this with

``` bash 
source [app path]/merizosearch_env/bin/activate
```

If you're using a virtualenv to install Torch, you may find you need to add the paths to the virtualenv versions of `cudnn/lib/` and `nccl/lib/` to your `LD_LIBRARY_PATH`.

BY DEFAULT we do not download the Merizo-search databases as they are nearly 1TB in size. You can do this manually (see below) or open `install.yml` and uncomment the line `- dataset`


## Databases
We provide pre-built Foldclass databases for domains in CATH 4.3 and all 365 million domains from TED. They can be obtained from [here](https://doi.org/10.5522/04/26348605). We recommend using our convenience script in this repository (`download_dbs.sh`) to download them. 
If you are using a browser to access the URL above, please make sure you download the individual files in each directory, rather than download each directory as a whole.

### Metadata format
Our pre-built databases (including the ones in the `example/` directory in this repo) include metadata for each domain in the database. Metadata is organised in JSON key-value format, and the exact fields in the db are allowed to vary. 

For the CATH database, we currently include the following fields (reformatted over multiple lines for clarity and annotated):
```
{
  "cath": "2.60.120.290",   # CATH assignment up to superfamily (H) level
  "res": "3.100",           # Resolution of the PDB entry (where applicable; "NA" otherwise)
  "rr": "4-124",            # Residue range in the PDB (SEQRES numbering)
  "clen": "576"             # The length of the *chain* that this domain is from. Useful for producing domain architecture diagrams (we use this on the PSIPRED server).
}
```

For the TED database, we supply a subset of the fields available in the master TSV file. Here is the metadata available for an exampled domain:
```
{
  "ted": "AF-Q9UKA2-F1-model_v4_TED01",  # TED consensus domain ID.
  "cnsl": "high",                        # TED consensus level; this is either "high" or "medium".
  "rr": "50-229",                        # TED consensus residue range in the AFDB model, sometimes called the "chopping".
  "plddt": "93.735",                     # Average plDDT of the domain residues.
  "cath": "2.60.120.260",                # Putative CATH label. This is in formatted as C.A.T.H, or C.A.T, or "-" where a label could not be assigned.
  "cl": "H",                             # The level in the CATH hierarchy up to which the label was assigned. This is either "H", "T", or "-".
  "cm": "foldseek",                      # The method used to assign the CATH label. This is either "foldseek", "foldclass", or "-".
  "dens": "11.6",                        # The packing density for this domain.
  "rg": "0.297",                         # The radius of gyration for the domain.
  "taxid": "9606",                       # The NCBI TaxID associated with this protein.
  "taxsci": "Homo_sapiens",              # The short taxonomic name for the TaxID.
  "clen": "621"                          # The length of the *chain* that this domain is from. Useful for producing domain architecture diagrams (we use this on the PSIPRED server). 
}
```
We will soon release scripts that will allow you to add JSON-formatted metadata to a custom database created by the `createdb` module (see below).

## Usage

Merizo-search supports the functionalities listed below. The `-h` flag can be used to show all options for each mode :
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

The input PDB can be a single PDB, or multiple, including something like `/dir/*.pdb`. The `output_prefix` will be appended with `_segment.tsv` to indicate the results of `segment`. 

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

The `search` module of Merizo-search will call Foldclass to search queries (as they are, without `segment`) against a pre-compiled database (created using `createdb`). This is useful when queries are already domains. 

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


## Multi-domain searching

Both `search` and `easy-search` support searching for database entries that match all domains in a query chain. In the case of `search`, all supplied query structures are considered as domains originating from a single chain and searched against the database. In the case of `easy-search`, segmentation and multi-domain search operate on a per-query-chain basis, that is, only domains segmented from individual query chains are searched together at a time.

To enable multi-domain searching, add the option `--multi_domain_search` to a `search` or `easy-search` command. 

A few important things to note:
- In multi-domain searches, `-k` still controls the maximum number of per-domain hits retrieved using vector search. We recommend setting it to 100 or so.
- We only keep hits where _all_ domains in each query chain are matched at least once in a hit chain. We don't return hits containing fewer domains than the query domain set. You can, however, manually supply a subset of pre-segmented domains to the `search` command with `--multi_domain_search` enabled.
- The accuracy of multi-domain `easy-search` runs is dependent on the accuracy of the initial Merizo segmentation. If you're not getting many meaningful hits, we recommend checking the output from the implicit `segment` step from your run. Merizo is fairly robust, but you may wish to manually segment your query chain and then re-run multi-domain search using the `search` module. 
- If any domains in the query or hit chains are discontinuous, you will need to carry out extra checks to verify the matches. In particular, do not rely solely on the value of the `match_category` field in the output.

### Multi-domain search output
When `--multi_domain_search` is supplied, multi-domain search results are output in a file with the suffix `_search_multi_dom.tsv`. Each line of this file describes a match between a query chain and a hit chain. This is different from the outputs from `search`, in which each line describes a domain-level match.

The format of this file is not configurable (though headers can be enabled with the `--output_headers` option), and has the following format:

```
query_chain	nqd	hit_chain	nhd	match_category	match_info	hit_metadata
3w5h	2	1amoA	4	1	3w5h_merizo_01:1amoA02:0.70881,3w5h_merizo_02:1amoA04:0.71	[{"cath": "2.40.30.10", "res": "2.600"},{"cath": "3.40.50.80", "res": "2.600"}]
3w5h	2	1amoB	4	1	3w5h_merizo_01:1amoB02:0.70881,3w5h_merizo_02:1amoB04:0.71	[{"cath": "2.40.30.10", "res": "2.600"},{"cath": "3.40.50.80", "res": "2.600"}]
3w5h	2	1b2rA	2	3	3w5h_merizo_01:1b2rA01:0.73567,3w5h_merizo_02:1b2rA02:0.70819	[{"cath": "2.40.30.10", "res": "1.800"},{"cath": "3.40.50.80", "res": "1.800"}]
3w5h	2	1bjkA	2	3	3w5h_merizo_01:1bjkA01:0.7425,3w5h_merizo_02:1bjkA02:0.708	[{"cath": "2.40.30.10", "res": "2.300"},{"cath": "3.40.50.80", "res": "2.300"}]

```
The columns are:

Column name | Meaning
:---: | ---
`query_chain` | The name of the query chain. In the case of `search` mode, all supplied structures are treated as domains coming from the same chain.
`nqd` | The number of domains in the query chain. In the case of `search` mode, this is the number of supplied domains.
`hit_chain` | The name of the matched chain in the database.
`nhd` | The total number of domains in the hit chain. This is always equal to or greater than `nqd`.
`match_category` | An integer from 0 to 3 indicating the type of match (see below).
`match_info` | Domain correspondence info. A comma-separated list of length `nqd`, each element of which is formatted as `query_domain:hit_domain:tm_align_score`.
`match_metadata` | JSON array containing metadata for each hit domain, in the order that the hit domains appear in `match_info`.

Multi-domain hits are categorised into one of 4 categories in the `match_category` field of the output, representing the type of multi-domain match. Each can be seen as a subset of the last:
 `match_category` | Category name | Meaning 
 :---: | --- | --- 
 0 | Unordered domain match | All query domains present in hit chain, but in different sequential order to query chain. Domains may be inserted relative to the query chain at any position. 
 1 | Discontiguous domain match | All query domains matched in sequential order, but hit chain has at least one extra domain in an interstitial position, and possibly at one or both ends. 
 2 | Contiguous domain match | All query domains matched in sequential order. Hit chain has extra domains at one or both ends, but not in interstitial positions. 
 3 | Exact multi-domain architecture (MDA) match | Query chain and hit chain correspond at domain level without domain rearrangement or insertions. 

It is possible for the same hit chain to be listed more than once for the same query chain, as multiple query domain-hit domain mappings may be possible (e.g. in the case of repeats of domains). In such cases, Merizo-search will list all such pairings, one per line.
As stated above, if any domains in the query or hit chains are discontinuous, you will need to carry out extra checks to verify the matches. In particular, do not rely solely on the value of `match_category`, as it uses a fairly simple algorithm to assign the match category values.

## Other outputs

The `segment` module used in `segment` and `easy-search` produces a number of different output files that can be turned on using various flags:
```
--save_domains  Save the domains as individual PDBs.
--save_pdb      Save a single PDB with the occupancy column replaced with domain IDs. (Visualise in PyMOL using the `spectrum q` command).
--save_pdf      Save a PDF output showing the domain map.
--save_fasta    Save the sequence of the input file.
```

By default, all output files will be saved alongside the original input query PDB, but they can be saved into a folder given by `--merizo_output`.

# Citing

If you find Foldclass and/or Merizo-search useful, please cite our paper in _Bioinformatics_:

> _Foldclass and Merizo-search: Scalable structural similarity search for single- and multi-domain proteins using geometric learning. Bioinformatics
> https://doi.org/10.1093/bioinformatics/btaf277_

Merizo is described in the following article:

> Merizo: a rapid and accurate protein domain segmentation method using invariant point attention. _Nature Communications_ **14**, 8445
> https://doi.org/10.1038/s41467-023-43934-4
