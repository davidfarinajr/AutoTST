%nprocshared=20
%mem=5GB
#p m062x/6-311+g(2df,2p) opt=(ts,calcfc,noeigentest,ModRedun) 

Gaussian input prepared by ASE

0 2
O                -2.6943000000       -0.0049000000        0.1335000000
O                -3.4761000000       -1.0578000000        0.0243000000
C                 0.3373000000        0.5562000000       -0.9851000000
C                 0.9880000000       -0.1118000000       -0.0102000000
C                -0.8219000000        1.4737000000       -0.7225000000
C                 0.5888000000        0.0224000000        1.4402000000
C                 2.1386000000       -1.0073000000       -0.3899000000
H                -2.9758000000       -1.7857000000        0.4721000000
H                 0.6556000000        0.4338000000       -2.0175000000
H                -1.8047000000        0.8090000000       -0.1650000000
H                -1.1838000000        1.9010000000       -1.6811000000
H                -0.5085000000        2.3076000000       -0.0602000000
H                 0.6792000000        1.0811000000        1.7621000000
H                -0.4591000000       -0.3171000000        1.5781000000
H                 1.2367000000       -0.5919000000        2.0994000000
H                 2.9280000000       -0.4105000000       -0.8941000000
H                 2.5877000000       -1.4978000000        0.4986000000
H                 1.7844000000       -1.7998000000       -1.0827000000

2 3 F
2 4 F
2 6 F
2 7 F
2 8 F
2 9 F
2 11 F
2 12 F
2 13 F
2 14 F
2 15 F
2 16 F
2 17 F
2 18 F
3 4 F
3 6 F
3 7 F
3 8 F
3 9 F
3 11 F
3 12 F
3 13 F
3 14 F
3 15 F
3 16 F
3 17 F
3 18 F
4 6 F
4 7 F
4 8 F
4 9 F
4 11 F
4 12 F
4 13 F
4 14 F
4 15 F
4 16 F
4 17 F
4 18 F
6 7 F
6 8 F
6 9 F
6 11 F
6 12 F
6 13 F
6 14 F
6 15 F
6 16 F
6 17 F
6 18 F
7 8 F
7 9 F
7 11 F
7 12 F
7 13 F
7 14 F
7 15 F
7 16 F
7 17 F
7 18 F
8 9 F
8 11 F
8 12 F
8 13 F
8 14 F
8 15 F
8 16 F
8 17 F
8 18 F
9 11 F
9 12 F
9 13 F
9 14 F
9 15 F
9 16 F
9 17 F
9 18 F
11 12 F
11 13 F
11 14 F
11 15 F
11 16 F
11 17 F
11 18 F
12 13 F
12 14 F
12 15 F
12 16 F
12 17 F
12 18 F
13 14 F
13 15 F
13 16 F
13 17 F
13 18 F
14 15 F
14 16 F
14 17 F
14 18 F
15 16 F
15 17 F
15 18 F
16 17 F
16 18 F
17 18 F

