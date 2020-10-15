#!/bin/bash

set -x
set -e

num_gpu=1

aimet_release_path="/prj/neo_lv/scratch/kwanan/mltt/release/aimet-1.5.0_build-0.90.0.87"
docker_name=aimet-prod_0.90.0.87

#node_name=morph-lsf14-gpulv
# node_name=morph-lsf09-gpulv
# node_name=morph-lsf16-gpulv
# node_name=morph-lsf11-gpulv
# node_name=morph-lsf17-gpulv
# node_name=morph-lsf15-gpulv
node_name=crd-lsf-gpu08
export CONTAINERIZE_SCRIPT="/prj/neoci/tools/ssit-tools/containerize_job.sh"

LOG_DIR="/prj/neo_lv/scratch/kwanan/personal/PACT/pytorch_resnet_cifar10"

bsub -q QcDev -P morpheus -app ${num_gpu}gpuEP -m "$node_name" -R "select[gpu && ubuntu16 && nvTeslaV100]" -eo $LOG_DIR/quant_error.log -oo $LOG_DIR/quant_std.log $CONTAINERIZE_SCRIPT -f $aimet_release_path/lib/config/aimet.conf -a $docker_name -c /prj/neo_lv/scratch/kwanan/personal/PACT/pytorch_resnet_cifar10/run.sh -w /prj/neo_lv/scratch/kwanan/personal/PACT/pytorch_resnet_cifar10

