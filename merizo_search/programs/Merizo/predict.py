from __future__ import annotations

import os
import sys
import time
import argparse
import textwrap
import logging

import torch

from typing import Dict, List, Optional
from programs.Merizo.model.network import Merizo

from programs.Merizo.model.utils.features import generate_features_domain
from programs.Merizo.model.utils.utils import (
    get_device,
    format_dom_str,
    instance_matrix,
    write_pdf_predictions,
    write_pdb_predictions,
    write_domain_idx,
    write_fasta,
    clean_domains,
    clean_singletons,
    get_ids,
    remap_ids,
    shuffle_ids,
    separate_components,
)

logger = logging.getLogger(__name__)

def iterative_segmentation(network: torch.nn.Module, features: dict, max_iterations: int, domain_ave_size: int, conf_threshold: float) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Perform iterative segmentation using the provided network and features.

    Args:
        network (torch.nn.Module): The segmentation network.
        features (dict): A dictionary containing features including 's', 'z', 'r', 't', 'ri', 'domain_ids', and 'conf_res'.
        max_iterations (int): The maximum number of iterations to perform.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing the updated domain_ids and conf_res tensors.
    """
    n_iterations = 0
    iterate = True
    ignore_index = []
    
    domain_ids, conf_res = features['domain_ids'], features['conf_res']

    while iterate:
        ids, _ = get_ids(domain_ids)
        unique_ids = {}

        for d in ids:
            if d.item() not in ignore_index:
                dom_mask = domain_ids == d
                dom_conf_res = conf_res[dom_mask]
                dom_nres = len(dom_conf_res)

                assert len(dom_conf_res.unique()) == 1

                dom_conf = dom_conf_res.mean()

                cond1 = dom_nres > domain_ave_size
                # cond2 = dom_conf < conf_threshold

                if cond1:  # or cond2:
                    unique_ids[d.item()] = dom_conf
                else:
                    ignore_index.append(d.item())

        if len(unique_ids) == 0:
            iterate = False
            break

        counter = 1
        for k, dconf in unique_ids.items():
            domain_mask = domain_ids == k

            domain_ids_, conf_res_ = network(features, mask=domain_mask)

            ids, ndoms_ = get_ids(domain_ids_)

            # Get new confidence scores per new domain
            new_dom_conf = torch.zeros_like(ids).float()
            for i, d in enumerate(ids):
                new_mask = domain_ids_ == d
                new_dom_conf[i] = conf_res_[new_mask].mean()

            # If the segment is still 1 domain, skip
            if ndoms_ == 1:
                ignore_index.append(k)

            # Otherwise split and overwrite old domain ids and confidences
            else:
                # Offset new ids by at least no_classes to ensure no overlap
                # with the old ids, and then transplant back in
                dd = domain_ids_ + (counter * network.no_classes)
                dd[domain_ids_ == 0] = 0
                domain_ids[domain_mask] = dd

                # Assign all residues in the new domain with the predicted
                # domain confidence
                conf_res[domain_mask] = conf_res_
                counter += 1

        n_iterations += 1
        if n_iterations == max_iterations:
            iterate = False
            break

    return domain_ids, conf_res


def read_split_weight_files(directory: str) -> dict:
    """
    Read weights from split weight files and return them as a dictionary.

    Args:
        directory (str): Path to the directory containing the weight files.

    Returns:
        dict: Dictionary containing the weights.
    """
    weights = {}

    # Get the list of weight files in the directory
    weight_files = [file for file in os.listdir(directory) if file.endswith('.pt')]

    # Read the weights from each file
    for file in weight_files:
        file_path = os.path.join(directory, file)
        subset_weights = torch.load(file_path)

        # Add the subset weights to the overall weights dictionary
        weights.update(subset_weights)

    return weights

def segment(pdb_path: str, network: torch.nn.Module, device: str, length_conditional_iterate: bool, iterate: bool, 
            max_iterations: int, shuffle_indices: bool, min_domain_size: int = 50, min_fragment_size: int = 10,
            domain_ave_size: int = 200, conf_threshold: float = 0.5,  pdb_chain: str="A",
    ) -> dict:
    """
    Segment domains in a protein structure.

    Args:
        pdb_path (str): Path to the input PDB file.
        network (torch.nn.Module): The segmentation network.
        device (str): Device to use for computation.
        length_conditional_iterate (bool): Whether to iterate based on the length of the protein.
        iterate (bool): Whether to perform iterative segmentation.
        max_iterations (int): Maximum number of iterations for iterative segmentation.
        shuffle_indices (bool): Whether to shuffle indices after segmentation.

    Returns:
        dict: A dictionary containing segmented features.
    """
    features = generate_features_domain(pdb_path, device, pdb_chain)
    
    if length_conditional_iterate and features['nres'] > 512:
        iterate = True

    features['domain_ids'], features['conf_res'] = network(features)

    # If iterate mode, iteratively segment domains
    if iterate:
        if features['nres'] > domain_ave_size * 2:
            features['domain_ids'], features['conf_res'] = iterative_segmentation(
                network=network, 
                features=features, 
                max_iterations=max_iterations, 
                domain_ave_size=domain_ave_size, 
                conf_threshold=conf_threshold
            )
            
    features['domain_map'] = instance_matrix(features['domain_ids'])[0]
    features['domain_ids'] = separate_components(features)

    if len(torch.unique(features['domain_ids'])) > 1:
        features['domain_ids'] = clean_domains(features['domain_ids'], min_domain_size)
        features['domain_ids'] = clean_singletons(features['domain_ids'], min_fragment_size)
        
    # Recompute the domain map given the new assignment
    features['domain_map'] = instance_matrix(features['domain_ids'])[0]
    
    features['conf_global'] = features['conf_res'].mean()
    features['ndom'] = get_ids(features['domain_ids'])[1]
    
    if shuffle_indices:
        features['domain_ids'] = shuffle_ids(features['domain_ids'])
    else:
        features['domain_ids'] = remap_ids(features['domain_ids'])
        
    return features

def generate_outputs(name_dict: Dict[str, str], features: Dict[str, any], output_dir: str, conf_filter: float=None, plddt_filter: float=None, 
    save_pdb: bool=False, save_domains: bool=False, save_fasta: bool=False, save_pdf: bool=False, return_indices: bool=False,
    return_domains_as_list: bool=False) -> None:
    """
    Generate various outputs based on the provided parameters.

    Args:
        name_dict (Dict[str, str]): A dictionary containing the names of PDB files.
        features (Dict[str, any]): A dictionary containing features.
        conf_filter: The confidence filter.
        plddt_filter: The PLDDT filter.
        save_pdb (bool): Whether to save PDB files.
        save_domains (bool): Whether to save domain files.
        save_fasta (bool): Whether to save FASTA files.
        save_pdf (bool): Whether to save PDF files.
        return_indices (bool): Whether to return indices.
    """
    if save_pdb or save_domains or return_domains_as_list:
        domains = write_pdb_predictions(
            features=features,
            output_dir=output_dir,
            name_dict=name_dict, 
            save_domains=save_domains, 
            conf_filter=conf_filter, 
            plddt_filter=plddt_filter, 
            return_domains_as_list=return_domains_as_list
        )

    if save_fasta:
        write_fasta(features['pdb'], name_dict, output_dir=output_dir)

    if return_indices:
        write_domain_idx(features, name_dict, output_dir=output_dir)

    if save_pdf:
        write_pdf_predictions(features, name_dict, output_dir=output_dir)
        
    return domains

def print_summary(features: Dict[str, any], name_dict: Dict[str, str], start_time: float) -> None:
    """
    Print a summary based on the provided features and timing information.

    Args:
        features (Dict[str, any]): A dictionary containing features.
        name_dict (Dict[str, str]): A dictionary containing the names of PDB files.
        start_time (float): The starting time of the process.
    """
    dom_str = format_dom_str(features['domain_ids'], features['ri'])
    
    nres_domain = features['domain_ids'].count_nonzero()
    nres_ndomain = features['nres'] - nres_domain
    
    end_time = time.time() - start_time
    
    return {
        'name': name_dict['pdb_name'],
        'length': features['nres'],
        'nres_domain': nres_domain,
        'nres_non_domain': nres_ndomain,
        'num_domains': features['ndom'],
        'conf': features['conf_global'].item(),
        'time': end_time,
        'dom_str': dom_str,
    }

def run_merizo(input_paths: List[str], device: str = 'cpu', max_iterations: int = 3, return_indices: bool = False, 
    length_conditional_iterate: bool = False, iterate: bool = False, shuffle_indices: bool = False, 
    save_pdb: bool = False, save_domains: bool = False, save_fasta: bool = False, save_pdf: bool = False, 
    conf_filter: Optional[any] = None, plddt_filter: Optional[any] = None, min_domain_size: int = 50, min_fragment_size: int = 10,
    domain_ave_size: int = 200, conf_threshold: float = 0.5, return_domains_as_list: bool=False, merizo_output: str=None, pdb_chain: str="A",
    threads: int = 0
) -> None:
    """
    Run the Merizo algorithm on input PDB paths.

    Args:
        input_paths (List[str]): List of paths to input PDB files.
        device (str): Device to use for computation. Defaults to 'cpu'.
        max_iterations (int): Maximum number of iterations. Defaults to 10.
        return_indices (bool): Whether to return indices. Defaults to False.
        length_conditional_iterate (bool): Whether to perform length conditional iteration. Defaults to False.
        iterate (bool): Whether to iterate. Defaults to False.
        shuffle_indices (bool): Whether to shuffle indices. Defaults to False.
        save_pdb (bool): Whether to save PDB files. Defaults to False.
        save_domains (bool): Whether to save domain files. Defaults to False.
        save_fasta (bool): Whether to save FASTA files. Defaults to False.
        save_pdf (bool): Whether to save PDF files. Defaults to False.
        conf_filter: The confidence filter.
        plddt_filter: The PLDDT filter.
    """
    
    if len(input_paths) == 0:
        logging.error("No inputs were provided!")
        sys.exit(1)
    
    if threads > 0:
        torch.set_num_threads(threads)
    device = get_device(device)
    network = Merizo().to(device)

    weights_dir = os.path.join(os.path.dirname(__file__), 'weights')
    network.load_state_dict(read_split_weight_files(weights_dir), strict=True)
    network.eval()
    
    if merizo_output is not None:
        if not os.path.exists(merizo_output):
            os.mkdir(merizo_output)
            
    segment_results = []
    print_results = []
    
    pdb_chain = pdb_chain.rstrip(",")
    pdb_chains = pdb_chain.split(",")

    if len(input_paths) != len(pdb_chains):
        if len(pdb_chains) == 1:
            pdb_chains = pdb_chains * len(input_paths)
        else:
            logger.error('Number of specified chain IDs not equal to number of input PDB files.')
            sys.exit(1)

    with torch.no_grad():
        for idx, pdb_path in enumerate(input_paths):
            if os.path.exists(pdb_path):
                start_time = time.time()
                
                if merizo_output is None:
                    merizo_output = os.getcwd()
                   # merizo_output = os.path.dirname(pdb_path)

                pdb_name = os.path.basename(pdb_path)
                pdb_bn, _ = os.path.splitext(pdb_name)
                pdb_out = pdb_bn + "_merizo"
                
                name_dict = {'pdb_name': pdb_name, 'pdb_path': pdb_path, 'pdb_bn': pdb_bn, 'pdb_out': pdb_out}

                if not os.path.exists(name_dict['pdb_out']):
                    features = segment(pdb_path=pdb_path, network=network, device=device, 
                        length_conditional_iterate=length_conditional_iterate, iterate=iterate, 
                        max_iterations=max_iterations, shuffle_indices=shuffle_indices,
                        min_domain_size=min_domain_size, min_fragment_size=min_fragment_size, 
                        domain_ave_size=domain_ave_size, conf_threshold=conf_threshold, pdb_chain=pdb_chains[idx]
                        )
                    
                    domains = generate_outputs(name_dict=name_dict, features=features, conf_filter=conf_filter, 
                        plddt_filter=plddt_filter, save_pdb=save_pdb, save_domains=save_domains,
                        save_fasta=save_fasta, save_pdf=save_pdf, return_indices=return_indices,
                        return_domains_as_list=return_domains_as_list, output_dir=merizo_output,
                        )
                    
                    results = print_summary(features=features, name_dict=name_dict, start_time=start_time)
                    
                    segment_results.extend(domains)
                    print_results.append(results)
            
    return segment_results, print_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='ProgramName',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent('''\
            If you use Merizo, please cite the following paper:
                Lau, et al., 2023. Merizo: a rapid and accurate domain segmentation method using invariant point attention. bioRxiv, doi: https://doi.org/10.1101/2023.02.19.529114
            
            Example usage:
                python predict.py -d cpu -i examples/2xdqA.pdb
                python predict.py -d cpu -i examples/*.pdb --save_domains --save_pdf --save_fasta
                python predict.py -d cpu -i examples/2xdqA.pdb --save_domains --plddt_filter
                
            For AlphaFold2 models, the iterative segmentation routine may give better results on longer models:
                python predict.py -d cpu -i examples/AF-Q96PD2-F1-model_v4.pdb --iterate --plddt_filter 60 --conf_filter 0.75
         ''')
    )
    parser.add_argument("-i", "--input", type=str, nargs="+", required=True, help="Specify path to pdb file input. Can also take multiple inputs (e.g. '/path/to/file.pdb' or '/path/to/*.pdb').")
    parser.add_argument("-d", "--device", type=str, default="cuda", help="Hardware to run on. Options: 'cpu', 'cuda', 'mps'.")
    parser.add_argument("--save_pdf", dest="save_pdf", action="store_true", help="Include to save the domain map as a pdf.")
    parser.add_argument("--save_pdb", dest="save_pdb", action="store_true", help="Include to save the result as a pdb file. All domains will be included unless --conf_filter or --plddt_filter is used.")
    parser.add_argument("--save_domains", dest="save_domains", action="store_true", help="Include to save parsed domains as separate pdb files. Also saves the full pdb.")
    parser.add_argument("--save_fasta", dest="save_fasta", action="store_true", help="Include to save a fasta file of the input pdb.")
    parser.add_argument("--conf_filter", dest="conf_filter", type=float, default=None, help="(float, [0-1]) If specified, only domains with a pIoU above this threshold will be saved. ")
    parser.add_argument("--plddt_filter", dest="plddt_filter", type=float, default=None, help="(float, [0-1]) If specified, only domain with a plDDT above this threshold will be saved. Note: if used on a non-AF structure, this will correspond to crystallographic b-factors.")
    parser.add_argument("--iterate", dest="iterate", action="store_true", help=f"If used, domains under a length threshold (see --min_domain_size) will be re-segmented.")
    parser.add_argument("--length_conditional_iterate", dest="length_conditional_iterate", action="store_true", help=f"If used, --iterate is set to True when the input sequence length is greater than 512 residues or greater")
    parser.add_argument("--max_iterations", dest="max_iterations", type=int, default=3, help="(int [1, inf]) Specify the maximum number of re-segmentations that can occur.")
    parser.add_argument("--shuffle_indices", dest="shuffle_indices", action="store_true", help="Shuffle domain indices - increases contrast between domain colours in PyMOL.")
    parser.add_argument("--return_indices", dest="return_indices", action="store_true", help="Return the domain indices for all residues.")
    parser.add_argument("--min_domain_size", dest="min_domain_size", type=int, default=50, help="The minimum domain size that is accepted.")
    parser.add_argument("--min_fragment_size", dest="min_fragment_size", type=int, default=10, help="Minimum number of residues in a segment.")
    parser.add_argument("--domain_ave_size", dest="domain_ave_size", type=int, default=200, help="[For iteration mode] Controls the size threshold to be used for further iterations.")
    parser.add_argument("--conf_threshold", dest="conf_threshold", type=float, default=0.5, help="[For iteration mode] Controls the minimum confidence to accept for iteration move.")
    parser.add_argument("--pdb_chain", type=str, dest="pdb_chain", default="A", help="Select which PDB Chain you are analysing. Defaut is chain A. You can provide a comma separated list if you can provide more than one input pdb")
    parser.add_argument('-t', '--threads', type=int, default=-1, required=False, help="Number of CPU threads to use.")
    
    args = parser.parse_args()
    
    run_merizo(
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
        min_domain_size=args.min_domain_size,
        min_fragment_size=args.min_fragment_size,
        domain_ave_size=args.domain_ave_size,
        conf_threshold=args.conf_threshold,
        return_domains_as_list=False,
        pdb_chain=args.pdb_chain,
        threads=args.threads,
    )
