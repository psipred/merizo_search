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

    # for faiss db just check for .json as the rest is checked elsewhere
    if os.path.exists(db_name + '.json'):
        return
    else:
        if not os.path.exists(db_name + '.pt'):
            logger.error(f"Cannot find database file {db_name + '.pt'}")
            sys.exit(1)
            
        if not os.path.exists(db_name + '.index'):
            logger.error(f"Cannot find database file {db_name + '.index'}")
            sys.exit(1)

def write_search_results(results: list[dict], output_file: str, format_list: str, header: bool):

    # TODO check if it is actually as simple as :
    # with open(output_file, 'w+') as fn:
    #     if header:
    #         fn.write('\t'.join(format_list) + '\n')

    with open(output_file, 'w+') as fn:
        if header: 
            head_str=''
            for option in format_list:    
                if option == 'query':
                    head_str+='query\t'
                elif option == 'target':
                    head_str+='target\t'
                elif option == 'chopping':
                    head_str+='dom_str\t'
                elif option == 'conf':
                    head_str+='dom_conf\t'
                elif option == 'plddt':
                    head_str+='dom_plddt\t'
                elif option == 'emb_rank':
                    head_str+='emb_rank\t'
                elif option == 'emb_score':
                    head_str+='emb_score\t'
                elif option == 'q_len':
                    head_str+='q_len\t'
                elif option == 't_len':
                    head_str+='t_len\t'
                elif option == 'ali_len':
                    head_str+='ali_len\t'
                elif option == 'seq_id':
                    head_str+='seq_id\t'
                elif option == 'q_tm':
                    head_str+='q_tm\t'
                elif option == 't_tm':
                    head_str+='t_tm\t'
                elif option == 'max_tm':
                    head_str+='max_t\t'
                elif option == 'rmsd':
                    head_str+='rmsd\t'
                else:
                    logger.warning(f"Format option '{option}' is not recognized.")
                    sys.exit(1)
            fn.write(f'{head_str.rstrip()}\n')

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
                
def write_segment_results(results: list[dict], output_file: str, header: bool):
    
    with open(output_file, 'w+') as fn:
        if header:
            fn.write('filename\tnres\tnres_dom\tnres_ndr\tndom\tpIoU\truntime\tresult\n')    
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
