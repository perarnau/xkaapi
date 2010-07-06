#!/bin/sh
set -e
test $# -gt 1 ||  exit 1 
cat > .tmp

grep -B 10 -E "${2}</h" .tmp | grep "<a name="
cat .tmp | sed -r -n -e "/<h$1.*>.*$2/,/<h[${1}$((${1}-1))].*>.*/ p" | \
    sed -r  -e 's/class="menu"/class="menu" style="display:none;"/g' \
            -e '/^(Next|Previous|Up):&nbsp;/ d' \
            -e '$ d' \
            -e '1 d' \
            -e '/^<p><hr>/ d' 
rm -f .tmp

    #        -e 's,(<h.*>)[\.0-9]+ (.*),\1\2,g'
    #sed -r  -e '/<ul class="menu">/,/<\/ul>/ d' \
    #        -e '/<div class="node">/,/<\/div>/ d' \
