wget -O train/example_dataset/enwiki-latest-abstract10.xml.gz https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-abstract10.xml.gz
gzip -d train/example_dataset/enwiki-latest-abstract10.xml.gz
mv train/example_dataset/enwiki-latest-abstract10.xml train/example_dataset/enwiki-latest-abstract.xml

