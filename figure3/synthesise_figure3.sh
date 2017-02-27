# Individual scales
samples=(1 2 3);

for sample in ${samples[@]}; do
  python3 ../synthesise.py -t pebbles.jpg -s 256 -f pebbles-sample-$sample.jpg
  python3 ../synthesise.py -t trees.jpg -s 256 -f trees-sample-$sample.jpg
  python3 ../synthesise.py -t brick_wall.jpg -s 256 -f brick-sample-$sample.jpg
done
