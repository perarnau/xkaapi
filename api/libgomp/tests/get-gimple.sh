#! /bin/bash

echo "Generating OpenMP-related gimple files..."

files=$(find . -name "*omp*" -exec ls {} +)

# Back up the interesting gimple files
for gimpfile in $files;
do
    filename=$(basename $gimpfile)
    cp $gimpfile "gimp-$filename"
done

# Delete the rest of them
for progname in $1;
do
    temp=$(find . -name "$progname.c.*" -exec ls {} +)
    if ! [ "$temp" == "" ];
    then	
	rm $temp
    fi
done
