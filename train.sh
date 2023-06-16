#!bash

for length in 256 512 768 1024 1024 1536; do
    python vits_japanese.py --sample_length ${length}  >  "results_fit-${length}.log"
done
