# 在wqmix基础上改
# 观察其他的baseline的训练次数，尽量差不多
# 显存大（12g之类），cpu、内存大
# 前面的CUDA_VISIBLE_DEVICES=3 先注释
# python src/main.py --config=ow_qmix --env-config=sc2 with env_args.map_name=3s_vs_5z w=0.5 epsilon_anneal_time=100000 t_max=3005000
# sleep 3
python src/main.py --config=qmix_wgan_nqlearner_new --env-config=sc2 with env_args.map_name=3m  epsilon_anneal_time=100000 t_max=10050000

python src/main.py --config=qmix_bak --env-config=sc2 with env_args.map_name=3m  epsilon_anneal_time=100000 t_max=1005000

python src/main.py --config=qmix_wgan --env-config=sc2 with env_args.map_name=3s_vs_5z  epsilon_anneal_time=100000 t_max=10050000


python src/main.py --config=qmix_wgan --env-config=sc2 with env_args.map_name=3m  epsilon_anneal_time=100000 t_max=10050000
sleep 3

python src/main.py --config=qmix_wgan_v2 --env-config=sc2 with env_args.map_name=3m  epsilon_anneal_time=100000 t_max=1005000
sleep 3

python src/main.py --config=qmix_wgan_v3 --env-config=sc2 with env_args.map_name=3m  epsilon_anneal_time=100000 t_max=10050000
sleep 3


python src/main.py --config=ow_qmix_wgan --env-config=sc2 with env_args.map_name=3s_vs_5z w=0.5 epsilon_anneal_time=100000 t_max=4005000
sleep 3

python src/main.py --config=ow_qmix_wgan --env-config=sc2 with env_args.map_name=5m_vs_6m w=0.5 epsilon_anneal_time=100000 t_max=4005000
sleep 3

python src/main.py --config=ow_qmix_wgan --env-config=sc2 with env_args.map_name=MMM2 w=0.5 epsilon_anneal_time=100000 t_max=2005000

sleep 3

python src/main.py --config=ow_qmix_wgan --env-config=sc2 with env_args.map_name=corridor w=0.5 epsilon_anneal_time=100000 t_max=5005000
sleep 3

python src/main.py --config=ow_qmix_wgan --env-config=sc2 with env_args.map_name=6h_vs_8z w=0.5 epsilon_anneal_time=500000 t_max=5005000
sleep 3






