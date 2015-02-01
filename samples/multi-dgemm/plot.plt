set terminal png
set output "plot.png"
set xlabel "batch size"
set ylabel "GFLOP/s"
set autoscale fix
set mxtics 2
set mytics 2

plot "plot.txt" using 2:3 notitle smooth sbezier, \
     "" using 2:3 notitle with linespoints pt 7
