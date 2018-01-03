# Face clustering and motion segmentation

Project given as part of the following course: *Unsupervised learning*, fall 2017-2018, MSc MVA, ENS Paris-Saclay.

## Data

The data are available online, thanks to the The Vision, Dynamics and Learning Lab of John Hopkins University.

If you already have downloaded data, please skip the `download` subsection and proceed to the creation of symbolic link as suggested in `symbolic link` subsection

### Download

To download the data as required by the notebooks, you have to proceed as follow:

```
wget -O data/ExtendedYaleB.mat "http://www.vision.jhu.edu/gpca/fetchcode.php?id=210?ExtendedYaleB.mat"
```

```
wget -O data/Hopkins155 "http://www.vision.jhu.edu/data/fetchdata.php?id=1?Hopkins155.zip"
```

### Creation of symbolic links

If you have already download the Hopkins155 dataset, you can create your own symbolic link, or use the `create_link.sh` file as follow:
```
bash create_link.sh
```

output=$(find . -name "*Hopkins155" 2> >(grep -v 'Permission denied' >&2))
max_size=$(echo $output | cut -f 1 -d $'\n')
for i in $(echo $output)
do
	cur_file_size=$(du -sb $i| cut -f 1 -d $'\t')
	echo "size of $i : $cur_file_size"
done



output=$(find . -name "*Hopkins155" 2> >(grep -v 'Permission denied' >&2))
max_size=$(echo $output | cut -f 1 -d $'\n')
for i in $(echo $output)
do
	cur_file_size=$(du -b $i| cut -f 1 -d $'\t')
	if [[ $cur_file_size -ge $max_size ]]; then
		echo $cur_file_size
	fi
done
