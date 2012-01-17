#! /bin/bash

files=$(find . -name "*omp*" -exec ls {} +)

echo "Files = $files"

# Back up the interesting gimple files
for gimpfile in $files;
do
    filename=$(basename $gimpfile)
    cp $gimpfile "gimp-$filename"
done

echo "\$1 = $1"

# Delete the rest of them
for progname in $1;
do
    echo "Progname = $progname"
    temp=$(find . -name "$progname.c.*" -exec ls {} +)
    if ! [ "$temp" == "" ];
    then	
	rm $temp
    fi
done
