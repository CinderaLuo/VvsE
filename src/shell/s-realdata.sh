path=../data/
for i in amazon.bin dblp.bin wiki.bin socLive.bin
do
    echo  ".././exp ${path}${i} >> ./result-real/${i}.txt"
  	.././exp ${path}${i} >> ./result-real/${i}.txt
done

