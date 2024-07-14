# python main.py --name edge-nodownscale-fra --dataset_name 2020NHHAZE --sample_interval 300 --model net --h_size 512 --w_size 512 --epochs 1000 --lambda_edge 0
# python main.py --name siglePretrainedBranch --dataset_name 2018OHAZE --test_only --model siglePretrainedBranch

python main.py \
--name 2020NHHAZE_Split_Aug \
--model sym \
--comment RNet_DNet_best\
