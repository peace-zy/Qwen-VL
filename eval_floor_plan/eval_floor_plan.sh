echo "\033[32mword\033[0m"
python eval_floor_plan.py --dataset huxinggaizao --standardization --cut_mode word --save_path eval_out_zero3_cosine --submit --res_file Qwen-VL-Chat-huxinggaizao_zero3_cosine/res.json
echo "\033[32mphrase\033[0m"
python eval_floor_plan.py --dataset huxinggaizao --standardization --cut_mode phrase --save_path eval_out_zero3_cosine --submit --res_file Qwen-VL-Chat-huxinggaizao_zero3_cosine/res.json
