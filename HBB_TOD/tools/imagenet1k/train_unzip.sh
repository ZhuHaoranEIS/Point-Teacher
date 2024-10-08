dir=/data/zhr/ImageNet1k/train
# for x in `ls $dir/*tar` do     
#   filename=`basename $x .tar`     
#   mkdir $dir/$filename     
#   tar -xvf $x -C $dir/$filename 
# done 
for x in $(ls $dir/*tar); do
  filename=$(basename $x .tar)
  mkdir $dir/$filename
  tar -xvf $x -C $dir/$filename
done
rm *.tar
