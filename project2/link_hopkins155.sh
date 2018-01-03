#!/bin/bash

output=$(find . -name "*Hopkins155" 2> >(grep -v 'Permission denied' >&2))
directory_MLE=$(echo $output | cut -f 1 -d $'\n')
if [ -d "$directory_MLE" ]; then
    # Will enter here if $directory_MLE exists
	max_size=$(du -sb $directory_MLE| cut -f 1 -d $'\t')
fi
for i in $(echo $output)
do
	if [ -d "$i" ]
	then
		# Will enter if the current directory in $output exists
		cur_file_size=$(du -sb $i| cut -f 1 -d $'\t')
		if [[ $cur_file_size -ge $max_size ]]
		then
			max_size=$cur_file_size
			directory_MLE=$(echo $i| cut -f 1 -d $'\t')
		fi
	fi
done
# Typical size of rqw Hopkins155 is 30Mo
if [[ $max_size -ge 30000000 ]]
then
	echo "You may have already download Hopkins155 dataset."
	echo "A symbolic will be created in data/Hopkins155 to ${directory_MLE/./$PWD}"
	ln -s ${directory_MLE/./$PWD} data/Hopkins155
else
	echo "You may not have properly downloaded Hopkins155 dataset yet. Please report to subsection Download in the README"
fi
