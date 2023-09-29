gpu=0
######## airfoil
CUDA_VISIBLE_DEVICES=$gpu \
python main.py --dataset Airfoil --mixtype kde --name your_name --kde_bandwidth 1.75 --use_manifold 1 --store_model 1 --read_best_model 0 --seed 0
CUDA_VISIBLE_DEVICES=$gpu \
python main.py --dataset Airfoil --mixtype kde --name your_name --kde_bandwidth 1.75 --use_manifold 1 --store_model 1 --read_best_model 0 --seed 1
CUDA_VISIBLE_DEVICES=$gpu \
python main.py --dataset Airfoil --mixtype kde --name your_name --kde_bandwidth 1.75 --use_manifold 1 --store_model 1 --read_best_model 0 --seed 2

## no2
CUDA_VISIBLE_DEVICES=$gpu \
python main.py --dataset NO2 --mixtype kde --name your_name --kde_bandwidth 1.2 --use_manifold 0 --store_model 1 --read_best_model 0 --seed 0
CUDA_VISIBLE_DEVICES=$gpu \
python main.py --dataset NO2 --mixtype kde --name your_name --kde_bandwidth 1.2 --use_manifold 0 --store_model 1 --read_best_model 0 --seed 1
CUDA_VISIBLE_DEVICES=$gpu \
python main.py --dataset NO2 --mixtype kde --name your_name --kde_bandwidth 1.2 --use_manifold 0 --store_model 1 --read_best_model 0 --seed 2
#
## exchange_rate
CUDA_VISIBLE_DEVICES=$gpu \
python main.py --dataset TimeSeries --name your_name --data_dir data/exchange_rate/exchange_rate.txt --ts_name exchange_rate --mixtype kde --name your_name --kde_bandwidth 5e-2 --use_manifold 1 --store_model 1 --read_best_model 0 --seed 0
CUDA_VISIBLE_DEVICES=$gpu \
python main.py --dataset TimeSeries --name your_name --data_dir data/exchange_rate/exchange_rate.txt --ts_name exchange_rate --mixtype kde --name your_name --kde_bandwidth 5e-2 --use_manifold 1 --store_model 1 --read_best_model 0 --seed 1
CUDA_VISIBLE_DEVICES=$gpu \
python main.py --dataset TimeSeries --name your_name --data_dir data/exchange_rate/exchange_rate.txt --ts_name exchange_rate --mixtype kde --name your_name --kde_bandwidth 5e-2 --use_manifold 1 --store_model 1 --read_best_model 0 --seed 2

#### rcf-mnist
CUDA_VISIBLE_DEVICES=$gpu \
python main.py --dataset RCF_MNIST --data_dir data/RCF_MNIST --mixtype random --name your_name --batch_type 1 --kde_bandwidth 0.2 --use_manifold 1 --store_model 1 --read_best_model 0 --seed 0
CUDA_VISIBLE_DEVICES=$gpu \
python main.py --dataset RCF_MNIST --data_dir data/RCF_MNIST --mixtype random --name your_name --batch_type 1 --kde_bandwidth 0.2 --use_manifold 1 --store_model 1 --read_best_model 0 --seed 1

CUDA_VISIBLE_DEVICES=$gpu \
python main.py --dataset RCF_MNIST --data_dir data/RCF_MNIST --mixtype random --name your_name --batch_type 1 --kde_bandwidth 0.2 --use_manifold 1 --store_model 1 --read_best_model 0 --seed 2

##### crime
CUDA_VISIBLE_DEVICES=$gpu \
python main.py --dataset CommunitiesAndCrime --mixtype kde --name your_name --kde_bandwidth 4.0 --use_manifold 1 --store_model 1 --read_best_model 0 --seed 0
CUDA_VISIBLE_DEVICES=$gpu \
python main.py --dataset CommunitiesAndCrime --mixtype kde --name your_name --kde_bandwidth 4.0 --use_manifold 1 --store_model 1 --read_best_model 0 --seed 1
CUDA_VISIBLE_DEVICES=$gpu \
python main.py --dataset CommunitiesAndCrime --mixtype kde --name your_name --kde_bandwidth 4.0 --use_manifold 1 --store_model 1 --read_best_model 0 --seed 2
#
#### skillcraft
CUDA_VISIBLE_DEVICES=$gpu \
python main.py --dataset SkillCraft --mixtype kde --name your_name --kde_bandwidth 1.0 --use_manifold 0 --store_model 1 --read_best_model 0 --seed 0
CUDA_VISIBLE_DEVICES=$gpu \
python main.py --dataset SkillCraft --mixtype kde --name your_name --kde_bandwidth 1.0 --use_manifold 0 --store_model 1 --read_best_model 0 --seed 1
CUDA_VISIBLE_DEVICES=$gpu \
python main.py --dataset SkillCraft --mixtype kde --name your_name --kde_bandwidth 1.0 --use_manifold 0 --store_model 1 --read_best_model 0 --seed 2
####
### dti_dg
CUDA_VISIBLE_DEVICES=$gpu \
python main.py --dataset Dti_dg --data_dir data/dti --mixtype kde --name your_name --kde_bandwidth 10.0 --use_manifold 1 --store_model 1 --read_best_model 0 --seed 0
CUDA_VISIBLE_DEVICES=$gpu \
python main.py --dataset Dti_dg --data_dir data/dti --mixtype kde --name your_name --kde_bandwidth 10.0 --use_manifold 1 --store_model 1 --read_best_model 0 --seed 1
CUDA_VISIBLE_DEVICES=$gpu \
python main.py --dataset Dti_dg --data_dir data/dti --mixtype kde --name your_name --kde_bandwidth 10.0 --use_manifold 1 --store_model 1 --read_best_model 0 --seed 2