#!/bin/bash

set -eu

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <dbname> <output_dir>" >&2
    echo "   dbname is either 'ted' or 'cath'." >&2
    exit 0
fi

wget_options="--no-clobber -c --tries=0 --retry-connrefused" # -q"

dbname=$1
output_dir=$2


# Check if output_dir is a directory and writable
if [ -d $output_dir ]; then
    if [ ! -w $output_dir ]; then
        echo "$output_dir is a directory but not writable." >&2
        exit 1
    fi
else
    echo "$output_dir is not a directory." >&2
    exit 1
fi

if [ $dbname == 'ted' ]; then
    # TED
    wget ${wget_options} -O ${output_dir}/ted_365M_ca.db  https://rdr.ucl.ac.uk/ndownloader/files/50817567
    wget ${wget_options} -O ${output_dir}/ted_365M_ca.index https://rdr.ucl.ac.uk/ndownloader/files/50813403
    wget ${wget_options} -O ${output_dir}/ted_365M.json https://rdr.ucl.ac.uk/ndownloader/files/50813226
    wget ${wget_options} -O ${output_dir}/ted_365M_metadata.db https://rdr.ucl.ac.uk/ndownloader/files/50814024
    wget ${wget_options} -O ${output_dir}/ted_365M_metadata.index https://rdr.ucl.ac.uk/ndownloader/files/50813400
    wget ${wget_options} -O ${output_dir}/ted_365M_raw_128d.index_names https://rdr.ucl.ac.uk/ndownloader/files/50813532
    wget ${wget_options} -O ${output_dir}/ted_365M_raw_128d_norm.db https://rdr.ucl.ac.uk/ndownloader/files/50814849 
    wget ${wget_options} -O ${output_dir}/ted_365M_seq.db https://rdr.ucl.ac.uk/ndownloader/files/50813850
    wget ${wget_options} -O ${output_dir}/ted_365M_seq.index https://rdr.ucl.ac.uk/ndownloader/files/50813406

elif [ $dbname == 'cath' ]; then
    # CATH
    wget ${wget_options} -O ${output_dir}/cath-4.3-foldclassdb.index https://rdr.ucl.ac.uk/ndownloader/files/50846196
    wget ${wget_options} -O ${output_dir}/cath-4.3-foldclassdb.metadata https://rdr.ucl.ac.uk/ndownloader/files/50846190
    wget ${wget_options} -O ${output_dir}/cath-4.3-foldclassdb.metadata.index https://rdr.ucl.ac.uk/ndownloader/files/50846187
    wget ${wget_options} -O ${output_dir}/cath-4.3-foldclassdb.pt https://rdr.ucl.ac.uk/ndownloader/files/50846193
else
    echo "Unrecognised db name '$dbname'; currently only 'ted' and 'cath' are supported." >&2
    exit 1
fi
