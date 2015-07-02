#!/bin/sh
#
# This is a very small script that replaces grave accents and acute accents
# with apostrophes. For example, it will convert "didn`t" to "didn't".
#
# To use it, run:
#
#   ./fix_apostrophes.sh input_file output_file

if [ -z "$2" ]
then
    echo "Usage: fix_apostrophes.sh input_file output_file"
    exit 1
fi

# This problem can be solved very quickly using the Unix utility 'sed',
# the Stream Editor, which is most often used to search and replace.
# Here, we find matches to the regular expression [`´] -- that is, characters
# that are either grave or acute accents -- and replace them with apostrophes.
#
# The grave accent has to be escaped or else it would mean something special
# in this script.

sed "s/[\`´]/'/g" "$1" > "$2"
