set -x
base_dir=/home/zfgao/work/mlp-mixer-pytorch
check_point_dir=/home/zfgao/checkpoint/mlp-mixer-pytorch/
data_path=/home/zfgao/data
gpu_num=$1
function get_gpu_count() {
  str=$1
  array=(${str//,/})
  echo ${#array}
}


function run_cifar(){
    export CUDA_VISIBLE_DEVICES=$1
    COMMON_ARGS="--run_name=$2 --dataset=$3 --data_path=${data_path} --save_model_pth=$check_point_dir/$2 --batch_size=$4 --log_file=Experiments/cifar_logs/$2_$(date "+%Y%m%d-%H%M%S").log $5"
    nohup python -u $base_dir/main.py \
    ${COMMON_ARGS} > Experiments/cifar_logs/error.log 2>&1 &
    }

# run_cifar 3 Vanilla_Mixer CIFAR10 150 --epoch=100\ --model=MLPMixer\ --patch_size=2\ --dim=512\ --depth=12\ 
run_cifar 0 Vanilla_Mixer_patch4_dim512_depth12_tokendim128_channeldim1024 CIFAR10 300 --epoch=100\ --model=MLPMixer\ --patch_size=4\ --dim=512\ --depth=12\ --token_dim=128\ --channel_dim=1024\  




