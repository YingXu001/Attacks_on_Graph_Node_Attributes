python main.py --dataset_type graph --dataset_name Cora --lr 0.001 --patience 50 --epochs 500 --model GAT
python main.py --dataset_type graph --dataset_name Cora --lr 0.001 --patience 50 --epochs 500 --apply_attack --attack_type decision_time --norm_type L1
python main.py --dataset_type text --dataset_name hellaswag --file_path data/combined_hellaswag.json --lr 0.001 --patience 50 --epochs 500 --model GAT
python main.py --dataset_type text --dataset_name hellaswag --file_path data/combined_hellaswag.json --lr 0.001 --patience 50 --epochs 500 --apply_attack --attack_type decision_time --norm_type L1
python main.py --dataset_type text --dataset_name hellaswag --file_path data/combined_hellaswag.json --lr 0.001 --patience 50 --epochs 500 --apply_attack --attack_type decision_time_K --norm_type Linf
python main.py --dataset_type text --dataset_name hellaswag --file_path data/combined_hellaswag.json --lr 0.001 --patience 50 --epochs 500 --apply_attack --attack_type poisoning