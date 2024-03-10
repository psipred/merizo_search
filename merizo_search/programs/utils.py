import logging 
import sys
import os

logger = logging.getLogger(__name__)

def parse_output_format(format_str: str, expected_str: str):
    wanted_fields = format_str.split(',')
    expected_fields = expected_str.split(',')
    
    if not wanted_fields:
        logger.error("No fields found in the provided format string.")
        sys.exit(1)

    for field in wanted_fields:
        if field not in expected_fields:
            logger.warning(f"Format option '{field}' is not recognized.")
            sys.exit(1)
            
    return wanted_fields

def check_for_database(db_name):
    if not os.path.exists(db_name + '.pt'):
        logging.error(f"Cannot find database file {db_name + '.pt'}")
        sys.exit(1)
        
    if not os.path.exists(db_name + '.index'):
        logging.error(f"Cannot find database file {db_name + '.index'}")
        sys.exit(1)

def write_search_results(results: list[dict], output_file: str, format_list: str):
    
    with open(output_file, 'w+') as fn:
        for res in results:
            for k, match in res.items():
                formatted_output = []

                for option in format_list:
                    if option == 'query':
                        formatted_output.append(match['query'])
                    elif option == 'target':
                        formatted_output.append(match['target'])
                    elif option == 'chopping':
                        formatted_output.append(match['dom_str'])
                    elif option == 'conf':
                        formatted_output.append("{:.4f}".format(match['dom_conf']))
                    elif option == 'plddt':
                        formatted_output.append("{:.4f}".format(match['dom_plddt']))
                    elif option == 'emb_rank':
                        formatted_output.append("{}".format(k))
                    elif option == 'emb_score':
                        formatted_output.append("{:.4f}".format(match['score']))
                    elif option == 'q_len':
                        formatted_output.append("{}".format(match['q_len']))
                    elif option == 't_len':
                        formatted_output.append("{}".format(match['t_len']))
                    elif option == 'ali_len':
                        formatted_output.append("{}".format(match['tmalign_output']['len_ali']))
                    elif option == 'seq_id':
                        formatted_output.append("{:.4f}".format(match['tmalign_output']['seq_id']))
                    elif option == 'q_tm':
                        formatted_output.append("{:.4f}".format(match['tmalign_output']['qtm']))
                    elif option == 't_tm':
                        formatted_output.append("{:.4f}".format(match['tmalign_output']['ttm']))
                    elif option == 'max_tm':
                        formatted_output.append("{:.4f}".format(max(match['tmalign_output']['qtm'], match['tmalign_output']['ttm'])))
                    elif option == 'rmsd':
                        formatted_output.append("{:.2f}".format(match['tmalign_output']['rmsd']))
                    else:
                        logger.warning(f"Format option '{option}' is not recognized.")
                        sys.exit(1)
                        
                fn.write('\t'.join(formatted_output) + '\n')
                
def write_segment_results(results: list[dict], output_file: str):
    
    with open(output_file, 'w+') as fn:
        for res in results:
            fn.write("{}\t{}\t{}\t{}\t{}\t{:.4f}\t{:.4f}\t{}\n".format(
                os.path.basename(res['name']).replace('.pdb', ''),
                int(res['length']),
                int(res['nres_domain']),
                int(res['nres_non_domain']),
                int(res['num_domains']),
                res['conf'],
                res['time'],
                res['dom_str'],
            ))