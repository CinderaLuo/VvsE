data=/home/xxling/Luo/datasets/rmat-n1m5_
srcSuffix=.gt
destSuffix=.bin
dataname=rmat_n1m5_
generateCMD=/home/xxling/Luo/VvsE/src/data/./mod

:<<!
#generate data
for i in 1 2 3 4 5 6 7 8
do
	echo "${generateCMD} ${data}${i}${srcSuffix}  ./result-rmat/data/${dataname}${i}${destSuffix}"
	${generateCMD} ${data}${i}${srcSuffix} 	./result-rmat/data/${dataname}${i}${destSuffix}
done
!

#Run
for i in 1 2 3 4 5 6 7 8
do
    echo  ".././exp ./result-rmat/data/${dataname}${i}${destSuffix} >> ./result-rmat/${dataname}${i}.txt"
  	.././exp ./result-rmat/data/${dataname}${i}${destSuffix} >> ./result-rmat/${dataname}${i}.txt
done

