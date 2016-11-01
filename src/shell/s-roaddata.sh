path=/home/xxling/Luo/TKDE-0323/add-data/
for i in roadCA.bin roadPA.bin roadTX.bin
do
    echo  ".././exp ${path}${i} >> ./result-road/${i}.txt"
  	.././exp ${path}${i} >> ./result-road/${i}.txt
done

