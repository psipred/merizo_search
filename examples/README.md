# PDB Examples to try

## 3w5h.pdb

This PDB file should be segmented to a single domain. When search/easy-search is run with topk=10. It should report 17 significant hits against the provided ted100_9606_small test db.

This PDB file should be segmented to a two domains. When search/easy-search is run with topk=10. It should report 12 significant hits against the provided cath test db.

## AF-Q96HM7-F1-model_v4.pdb

This PDB file should be segmented to a single domain. When search/easy-search is run with topk=10. It should report 7 significant hits against the provided ted100_9606_small test db.

## AF-Q96PD2-F1-model_v4.pdb

This PDB file should be segmented to a two domains. When search/easy-search is run with topk=10. It should report 28 significant hits against the provided ted100_9606_small test db.

## M0.pdb

This PDB file should fail to segment.

# Example Databases

In `databaes/` you will find two small databases to allow you to test the functionality.

# CATH

Theses are the domains from CATH 4.3 clustered at 20% sequence identity. Names as cath-dataset-nonredundant-S20. You can use the symlinks in the directory to refer to this

# TED100

This is a small slice of the TED domains from the human genome. You can refer to this is ./ted100_9606_smallted/100_9606_small. Or you can use the symlinks in the database dir