# Individual scales
scales=(3 7 11 23 37 55);

for scale in ${scales[@]}; do
  python3 ../synthesise.py -t pebbles.jpg -s 256 -n 1 -f pebbles-single-$scale.jpg -c 1024 --scales $scale
  python3 ../synthesise.py -t trees.jpg -s 256 -n 1 -f trees-single-$scale.jpg -c 1024 --scales $scale
  python3 ../synthesise.py -t brick_wall.jpg -s 256 -n 1 -f brick-single-$scale.jpg -c 1024 --scales $scale
done


# Linear multi-scale
python3 ../synthesise.py -t pebbles.jpg -s 256 -n 1 -f pebbles-multiscale.jpg
python3 ../synthesise.py -t trees.jpg -s 256 -n 1 -f trees-multiscale.jpg
python3 ../synthesise.py -t brick_wall.jpg -s 256 -n 1 -f brick-multiscale.jpg

# Linear multi-scale
python3 ../synthesise.py -t pebbles.jpg -s 256 -n 1 -f pebbles-multi-linear.jpg --linear
python3 ../synthesise.py -t trees.jpg -s 256 -n 1 -f trees-multi-linear.jpg --linear
python3 ../synthesise.py -t brick_wall.jpg -s 256 -n 1 -f brick-multi-linear.jpg --linear
