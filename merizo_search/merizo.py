import argparse
import sys
import os
import logging
import time

SCRIPTDIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(SCRIPTDIR, 'programs'))

from programs.Merizo.predict import run_merizo as segment_pdb
from programs.Foldclass.makedb import run_createdb as createdb_from_pdb
from programs.Foldclass.dbsearch import run_dbsearch as dbsearch
from programs.utils import (
    parse_output_format, 
    write_search_results, 
    write_segment_results,
    check_for_database
)

def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# Function to handle segment mode
def segment(args):
    parser = argparse.ArgumentParser(description="Merizo segment is used to segment a multidomain protein.")
    parser.add_argument("input", type=str, nargs="+", help="Specify path to pdb file input. Can also take multiple inputs (e.g. '/path/to/file.pdb' or '/path/to/*.pdb').")
    parser.add_argument("output", type=str, help="Output file prefix to write segment results to. Results will be called _segment.tsv.")
    parser.add_argument("-d", "--device", type=str, default="cpu", help="Hardware to run on. Options: 'cpu', 'cuda', 'mps'.")
    parser.add_argument("--merizo_output", type=str, help="Designate where to save the merizo outputs to.")
    parser.add_argument("--save_pdf", action="store_true", default=False, help="Include to save the domain map as a pdf.")
    parser.add_argument("--save_pdb", action="store_true", default=False, help="Include to save the result as a pdb file. All domains will be included unless --conf_filter or --plddt_filter is used.")
    parser.add_argument("--save_domains", action="store_true", default=False, help="Include to save parsed domains as separate pdb files. Also saves the full pdb.")
    parser.add_argument("--save_fasta", action="store_true", default=False, help="Include to save a fasta file of the input pdb.")
    parser.add_argument("--output_headers", action="store_true", default=False, help="Select whether output TSV files have headers or not")
    parser.add_argument("--conf_filter", type=float, default=None, help="(float, [0-1]) If specified, only domains with a pIoU above this threshold will be saved. ")
    parser.add_argument("--plddt_filter", type=float, default=None, help="(float, [0-1]) If specified, only domain with a plDDT above this threshold will be saved. Note: if used on a non-AF structure, this will correspond to crystallographic b-factors.")
    parser.add_argument("--iterate", action="store_true", help=f"If used, domains under a length threshold (see --min_domain_size) will be re-segmented.")
    parser.add_argument("--length_conditional_iterate", action="store_true", help=f"If used, --iterate is set to True when the input sequence length is greater than 512 residues or greater")
    parser.add_argument("--max_iterations", type=int, default=3, help="(int [1, inf]) Specify the maximum number of re-segmentations that can occur.")
    parser.add_argument("--shuffle_indices", action="store_true", default=False, help="Shuffle domain indices - increases contrast between domain colours in PyMOL.")
    parser.add_argument("--return_indices", action="store_true", default=False, help="Return the domain indices for all residues.")
    parser.add_argument("--min_domain_size", type=int, default=50, help="The minimum domain size that is accepted.")
    parser.add_argument("--min_fragment_size", type=int, default=10, help="Minimum number of residues in a segment.")
    parser.add_argument("--domain_ave_size", type=int, default=200, help="[For iteration mode] Controls the size threshold to be used for further iterations.")
    parser.add_argument("--conf_threshold", type=float, default=0.5, help="[For iteration mode] Controls the minimum confidence to accept for iteration move.")
    args = parser.parse_args(args)
    
    logging.info('Starting merizo segment with command: \n\n{}\n'.format(
        " ".join([f'"{arg}"' if " " in arg else arg for arg in sys.argv])
    ))
    
    segment_output = args.output + '_segment.tsv'
    if os.path.exists(segment_output):
        logging.warning(f"Segment output file '{segment_output}' already exists. Results will be overwritten!")
    
    start_time = time.time()
    
    _, segment_results = segment_pdb(
        input_paths=args.input, 
        device=args.device, 
        max_iterations=args.max_iterations, 
        return_indices=args.return_indices, 
        length_conditional_iterate=args.length_conditional_iterate, 
        iterate=args.iterate, 
        shuffle_indices=args.shuffle_indices, 
        save_pdb=args.save_pdb,
        save_domains=args.save_domains, 
        save_fasta=args.save_fasta,
        save_pdf=args.save_pdf, 
        conf_filter=args.conf_filter, 
        plddt_filter=args.plddt_filter,
        return_domains_as_list=True,
        merizo_output=args.merizo_output,
    )
    
    elapsed_time = time.time() - start_time
    logging.info(f'Finished merizo segment in {elapsed_time} seconds.')
    
    write_segment_results(results=segment_results, output_file=segment_output, header=args.output_headers)

# Function to handle createdb mode
def createdb(args):
    parser = argparse.ArgumentParser(description="Merizo createdb calls the createdb function of Foldclass to embed a directory of pdb files into a Foldclass database.")
    parser.add_argument('input_dir', type=str, help='Directory containing pdb files. Will read all .pdb files in this directory.')
    parser.add_argument('out_db', type=str, help='Output prefix for the created Foldseek db.')
    parser.add_argument('-d', '--device', type=str, default='cuda', required=False)
    args = parser.parse_args(args)
    
    logging.info('Starting merizo createdb with command: \n\n{}\n'.format(
        " ".join([f'"{arg}"' if " " in arg else arg for arg in sys.argv])
    ))
    
    start_time = time.time()

    createdb_from_pdb(
        pdb_files=args.input_dir, 
        out_db=args.out_db, 
        device=args.device
    )
    
    elapsed_time = time.time() - start_time
    logging.info(f'Finished merizo createdb in {elapsed_time} seconds.')

# Function to handle search mode
def search(args):
    parser = argparse.ArgumentParser(description="Merizo search calls the run_search function of Foldclass and searches query PDBs against a given database.")
    parser.add_argument('input', type=str, nargs="+", help="Specify path to pdb file input.")
    parser.add_argument('db_name', type=str, help="Foldclass database to search against.")
    parser.add_argument("output", type=str, help="Output file prefix to write search results to. Results will be called _search.tsv.")
    parser.add_argument('tmp', type=str, help="Temporary directory to write things to.")
    parser.add_argument('-k', '--topk', type=int, default=1, required=False)
    parser.add_argument('-t', '--threads', type=int, default=-1, required=False)
    parser.add_argument('-s', '--mincos', type=float, default=0.5, required=False)
    parser.add_argument('-m', '--mintm', type=float, default=0.5, required=False)
    parser.add_argument('-c', '--mincov', type=float, default=0.7, required=False)
    parser.add_argument('-d', '--device', type=str, default='cpu', required=False)
    parser.add_argument('-f', '--fastmode', action='store_true', required=False)
    parser.add_argument("--format", type=str, default="query,emb_rank,target,emb_score,q_len,t_len,ali_len,seq_id,q_tm,t_tm,max_tm,rmsd", help="Comma-separated list of variable names to output. Choose from: [query, target, emb_rank, emb_score, q_len, t_len, ali_len, seq_id, q_tm, t_tm, max_tm, rmsd].")
    parser.add_argument("--output_headers", action="store_true", default=False, help="Select whether iutput TSV files have headers or not")
    args = parser.parse_args(args)
    
    logging.info('Starting merizo search with command: \n\n{}\n'.format(
        " ".join([f'"{arg}"' if " " in arg else arg for arg in sys.argv])
    ))
    
    # Check that the database is valid
    check_for_database(args.db_name)

    search_output = args.output + '_search.tsv'
    if os.path.exists(search_output):
        logging.warning(f"Search output file '{search_output}' already exists. Results will be overwritten!")

    output_fields = parse_output_format(
        format_str=args.format, 
        expected_str="query,emb_rank,target,emb_score,q_len,t_len,ali_len,seq_id,q_tm,t_tm,max_tm,rmsd"
    )
    
    start_time = time.time()

    search_results = dbsearch(
        inputs=args.input,
        db_name=args.db_name,
        tmp=args.tmp,
        device=args.device,
        topk=args.topk, 
        fastmode=args.fastmode, 
        threads=args.threads, 
        mincos=args.mincos, 
        mintm=args.mintm, 
        mincov=args.mincov,
        inputs_are_ca=False,
    )
    
    write_search_results(results=search_results, output_file=search_output, format_list=output_fields, header=args.output_headers)
    
    elapsed_time = time.time() - start_time
    logging.info(f'Finished merizo search in {elapsed_time} seconds.')

# Function to handle easy-search mode
def easy_search(args):
    parser = argparse.ArgumentParser(description="Merizo easy_search runs segment on a multidomain chain and then searches it against a database.")
    parser.add_argument("input", type=str, nargs="+", help="Specify path to pdb file input. Can also take multiple inputs (e.g. '/path/to/file.pdb' or '/path/to/*.pdb').")
    parser.add_argument("db_name", type=str, help="Foldclass database to search against.")
    parser.add_argument("output", type=str, help="Output file prefix to write segment and search results to. Results will be called _segment.tsv and _search.tsv.")
    parser.add_argument("tmp", type=str, help="Temporary directory to write things to.")
    parser.add_argument("--format", type=str, default="query,chopping,conf,plddt,emb_rank,target,emb_score,q_len,t_len,ali_len,seq_id,q_tm,t_tm,max_tm,rmsd", help="Comma-separated list of variable names to output. Choose from: [query, target, conf, plddt, chopping, emb_rank, emb_score, q_len, t_len, ali_len, seq_id, q_tm, t_tm, max_tm, rmsd].")
    
    # Foldclass options
    parser.add_argument("-d", "--device", type=str, default="cpu", help="Hardware to run on. Options: 'cpu', 'cuda', 'mps'.")
    parser.add_argument('-k', '--topk', type=int, default=1, required=False)
    parser.add_argument('-t', '--threads', type=int, default=-1, required=False)
    parser.add_argument('-s', '--mincos', type=float, default=0.5, required=False)
    parser.add_argument('-m', '--mintm', type=float, default=0.5, required=False)
    parser.add_argument('-c', '--mincov', type=float, default=0.7, required=False)
    parser.add_argument('-f', '--fastmode', action='store_true', required=False)

    # Merizo options
    parser.add_argument("--merizo_output", type=str, help="Designate where to save the merizo outputs to.")
    parser.add_argument("--save_pdf", action="store_true", default=False, help="Include to save the domain map as a pdf.")
    parser.add_argument("--save_pdb", action="store_true", default=False, help="Include to save the result as a pdb file. All domains will be included unless --conf_filter or --plddt_filter is used.")
    parser.add_argument("--save_domains", action="store_true", default=False, help="Include to save parsed domains as separate pdb files. Also saves the full pdb.")
    parser.add_argument("--output_headers", action="store_true", default=False, help="Select whether output TSV files have headers or not")
    parser.add_argument("--save_fasta", action="store_true", default=False, help="Include to save a fasta file of the input pdb.")
    parser.add_argument("--conf_filter", type=float, default=None, help="(float, [0-1]) If specified, only domains with a pIoU above this threshold will be saved. ")
    parser.add_argument("--plddt_filter", type=float, default=None, help="(float, [0-1]) If specified, only domain with a plDDT above this threshold will be saved. Note: if used on a non-AF structure, this will correspond to crystallographic b-factors.")
    parser.add_argument("--iterate", action="store_true", help=f"If used, domains under a length threshold (see --min_domain_size) will be re-segmented.")
    parser.add_argument("--length_conditional_iterate", action="store_true", help=f"If used, --iterate is set to True when the input sequence length is greater than 512 residues or greater")
    parser.add_argument("--max_iterations", type=int, default=3, help="(int [1, inf]) Specify the maximum number of re-segmentations that can occur.")
    parser.add_argument("--shuffle_indices", action="store_true", default=False, help="Shuffle domain indices - increases contrast between domain colours in PyMOL.")
    parser.add_argument("--return_indices", action="store_true", default=False, help="Return the domain indices for all residues.")
    parser.add_argument("--min_domain_size", type=int, default=50, help="The minimum domain size that is accepted.")
    parser.add_argument("--min_fragment_size", type=int, default=10, help="Minimum number of residues in a segment.")
    parser.add_argument("--domain_ave_size", type=int, default=200, help="[For iteration mode] Controls the size threshold to be used for further iterations.")
    parser.add_argument("--conf_threshold", type=float, default=0.5, help="[For iteration mode] Controls the minimum confidence to accept for iteration move.")
    args = parser.parse_args(args)
    
    logging.info('Starting merizo search with command: \n\n{}\n'.format(
        " ".join([f'"{arg}"' if " " in arg else arg for arg in sys.argv])
    ))
    
    # Check that the database is valid
    check_for_database(args.db_name)

    segment_output = args.output + '_segment.tsv'
    if os.path.exists(segment_output):
        logging.warning(f"Segment output file '{segment_output}' already exists. Results will be overwritten!")
        
    search_output = args.output + '_search.tsv'
    if os.path.exists(search_output):
        logging.warning(f"Search output file '{search_output}' already exists. Results will be overwritten!")
        
    output_fields = parse_output_format(
        format_str=args.format, 
        expected_str="query,chopping,conf,plddt,emb_rank,target,emb_score,q_len,t_len,ali_len,seq_id,q_tm,t_tm,max_tm,rmsd"
    )
    
    start_time = time.time()

    segment_domains, segment_results = segment_pdb(
        input_paths=args.input, 
        device=args.device, 
        max_iterations=args.max_iterations, 
        return_indices=args.return_indices, 
        length_conditional_iterate=args.length_conditional_iterate, 
        iterate=args.iterate, 
        shuffle_indices=args.shuffle_indices, 
        save_pdb=args.save_pdb,
        save_domains=args.save_domains, 
        save_fasta=args.save_fasta,
        save_pdf=args.save_pdf, 
        conf_filter=args.conf_filter, 
        plddt_filter=args.plddt_filter,
        return_domains_as_list=True,
        merizo_output=args.merizo_output,
    )
    
    write_segment_results(results=segment_results, output_file=segment_output, header=args.output_headers)

    search_results = dbsearch(
        inputs=segment_domains,
        db_name=args.db_name,
        tmp=args.tmp,
        device=args.device,
        topk=args.topk, 
        fastmode=args.fastmode, 
        threads=args.threads, 
        mincos=args.mincos, 
        mintm=args.mintm, 
        mincov=args.mincov,
        inputs_are_ca=True,
    )
    
    write_search_results(results=search_results, output_file=search_output, format_list=output_fields, header=args.output_headers)
    
    elapsed_time = time.time() - start_time
    logging.info(f'Finished merizo easy-search in {elapsed_time} seconds.')
    

# Main function to parse arguments and call respective functions
def main():
    setup_logging()
    
    if len(sys.argv) < 2:
        print("Usage: python merizo.py <mode> <args>")
        return

    mode = sys.argv[1]
    args = sys.argv[2:]

    if mode == "segment":
        segment(args)
    elif mode == "createdb":
        createdb(args)
    elif mode == "search":
        search(args)
    elif mode == "easy-search":
        easy_search(args)
    else:
        print("Invalid mode. Please choose one of 'segment', 'createdb', 'search', or 'easy-search'.")

if __name__ == "__main__":
    main()
