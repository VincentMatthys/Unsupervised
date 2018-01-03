#!/bin/bash

output=$(find . -name "*ExtendedYaleB.mat" 2> >(grep -v 'Permission denied' >&2))
file_MLE=$(echo $output | cut -f 1 -d $'\n')
if [ -f "$file_MLE" ]; then
    # Will enter here if $file_MLE exists
	max_size=$(du -sb $file_MLE| cut -f 1 -d $'\t')
fi
for i in $(echo $output)
do
	if [ -f "$i" ]
	then
		# Will enter if the current directory in $output exists
		cur_file_size=$(du -sb $i| cut -f 1 -d $'\t')
		if [[ $cur_file_size -le 3993700 ]] && [[ $max_size -ge 3991700 ]]
		then
			max_size=$cur_file_size
			file_MLE=$(echo $i| cut -f 1 -d $'\t')
		fi
	fi
done
# Typical size of rqw Hopkins155 is 30Mo
if [[ $cur_file_size -le 3993700 ]] && [[ $cur_file_size -ge 3991700 ]]
then
	echo "You may have already download ExtendedYaleB dataset."
	echo "A symbolic will be created in data/ExtendedYaleB.mat to ${file_MLE/./$PWD}"
	ln -s ${file_MLE/./$PWD} data/ExtendedYaleB.mat
else
	echo "You may not have properly downloaded ExtendedYaleB dataset yet. Please report to subsection Download in the README"
fi
