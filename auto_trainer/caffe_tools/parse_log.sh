#!/bin/bash
# Usage parse_log.sh caffe.log
# It creates the following two text files, each containing a table:
#     caffe.log.test (columns: '#Iters Seconds TestAccuracy TestLoss')
#     caffe.log.train (columns: '#Iters Seconds TrainingLoss LearningRate')


# get the dirname of the script
DIR="$( cd "$(dirname "$0")" ; pwd -P )"
if [ "$#" -lt 2 ]
then
echo "Usage parse_log.sh /path/to/your.log /output/path/"
exit
fi
OUTDIR="$2"
sed -n '/Iteration .* Testing net/,/Iteration *. loss/p' $1 > $OUTDIR/aux.txt
sed -i '/Waiting for data/d' $OUTDIR/aux.txt
sed -i '/prefetch queue empty/d' $OUTDIR/aux.txt
sed -i '/Iteration .* loss/d' $OUTDIR/aux.txt
sed -i '/Iteration .* lr/d' $OUTDIR/aux.txt
sed -i '/Train net/d' $OUTDIR/aux.txt
grep 'Iteration ' $OUTDIR/aux.txt | sed  's/.*Iteration \([[:digit:]]*\).*/\1/g' > $OUTDIR/aux0.txt
grep 'Test net output #0' $OUTDIR/aux.txt | awk '{print $11}' > $OUTDIR/aux1.txt
grep 'Test net output #1' $OUTDIR/aux.txt | awk '{print $11}' > $OUTDIR/aux2.txt

# Extracting elapsed seconds
# For extraction of time since this line contains the start time
grep '] Solving ' $1 > $OUTDIR/aux3.txt
grep 'Testing net' $1 >> $OUTDIR/aux3.txt
python $DIR/extract_seconds.py $OUTDIR/aux3.txt $OUTDIR/aux4.txt

# Generating
echo 'writing to $OUTDIR/parsed_caffe_log.test'
echo 'Iters Seconds TestAccuracy TestLoss'> $OUTDIR/parsed_caffe_log.test
paste $OUTDIR/aux0.txt $OUTDIR/aux4.txt $OUTDIR/aux1.txt $OUTDIR/aux2.txt | column -t >> $OUTDIR/parsed_caffe_log.test
rm $OUTDIR/aux.txt $OUTDIR/aux0.txt $OUTDIR/aux1.txt $OUTDIR/aux2.txt $OUTDIR/aux3.txt $OUTDIR/aux4.txt

# For extraction of time since this line contains the start time
grep '] Solving ' $1 > $OUTDIR/aux.txt
grep ', loss = ' $1 >> $OUTDIR/aux.txt
grep 'Iteration ' $OUTDIR/aux.txt | sed  's/.*Iteration \([[:digit:]]*\).*/\1/g' > $OUTDIR/aux0.txt
grep ', loss = ' $1 | awk '{print $9}' > $OUTDIR/aux1.txt
grep ', lr = ' $1 | awk '{print $9}' > $OUTDIR/aux2.txt

# Extracting elapsed seconds
python $DIR/extract_seconds.py $OUTDIR/aux.txt $OUTDIR/aux3.txt

# Generating
echo 'writing to $OUTDIR/parsed_caffe_log.train'
echo 'Iters Seconds TrainingLoss LearningRate'> $OUTDIR/parsed_caffe_log.train
paste $OUTDIR/aux0.txt $OUTDIR/aux3.txt $OUTDIR/aux1.txt $OUTDIR/aux2.txt | column -t >> $OUTDIR/parsed_caffe_log.train
rm $OUTDIR/aux.txt $OUTDIR/aux0.txt $OUTDIR/aux1.txt $OUTDIR/aux2.txt  $OUTDIR/aux3.txt
