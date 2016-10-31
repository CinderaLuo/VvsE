path=../data/
for i in amazon.bin dblp.bin wiki.bin
do
    echo  ".././exp ${path}${i} >> ./result/${i}.txt"
  	.././exp ${path}${i} >> ./result/${i}.txt
done

