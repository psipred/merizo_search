import pymol

@cmd.extend
def view_matches(cat=3):

    cat = int(cat)
    if cat > 3 or cat < 0:
        print('Error: invalid cat. Valid values are 0-3.', sys.stderr)
        return None

    loaded_queries = list()
    
    with open('/home/shaunmk/software_staging/merizo_search/out_4query_cath_all_dom_search.tsv','r') as f:
        for i, line in enumerate(f):
            if i==0:
                continue
            d = line.strip().split()
            query = d[0]
            hit = d[2]
            match_cat = int(d[4])

            if match_cat == cat:
                if query not in loaded_queries:
                    cmd.fetch(query)
                    loaded_queries.append(query)

                try:
                    cmd.fetch(hit)
                    cmd.align(hit, query)
                except:
                    continue

                    
            
