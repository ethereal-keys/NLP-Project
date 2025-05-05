# PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

# run each command one by one to perform each steps.

# Localize
python agentless/fl/localize.py --file_level \
                                --output_folder results/swe-bench-lite/file_level \
                                --num_threads 1\
                                --target_id astropy__astropy-14995

python agentless/fl/localize.py --file_level \
                                --irrelevant \
                                --output_folder results/swe-bench-lite/file_level_irrelevant \
                                --num_threads 1 \
                                --target_id astropy__astropy-14995

python agentless/fl/retrieve.py --index_type simple --filter_type given_files \
                                --filter_file results/swe-bench-lite/file_level_irrelevant/loc_outputs.jsonl \
                                --output_folder results/swe-bench-lite/retrievel_embedding \
                                --persist_dir embedding/swe-bench_simple \
                                --num_threads 1 \
                                --target_id astropy__astropy-14995

python agentless/fl/combine.py  --retrieval_loc_file results/swe-bench-lite/retrievel_embedding/retrieve_locs.jsonl \
                                --model_loc_file results/swe-bench-lite/file_level/loc_outputs.jsonl \
                                --top_n 3 \
                                --output_folder results/swe-bench-lite/file_level_combined 

python agentless/fl/localize.py --related_level \
                                --output_folder results/swe-bench-lite/related_elements \
                                --top_n 3 \
                                --compress_assign \
                                --compress \
                                --start_file results/swe-bench-lite/file_level_combined/combined_locs.jsonl \
                                --num_threads 1 \
                                --skip_existing \
                                --target_id astropy__astropy-14995