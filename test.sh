CUDA_VISIBLE_DEVICES="0" \
python test.py --checkpoint "trainpth3/Net_epoch_best_106.pth" --test_image_path "Dataset/TestDataset/CAMO/Imgs/" --test_gt_path "Dataset/TestDataset/CAMO/GT/" --save_path "results/CAMO/"
CUDA_VISIBLE_DEVICES="0" \
python test.py --checkpoint "trainpth3/Net_epoch_best_106.pth" --test_image_path "Dataset/TestDataset/CHAMELEON/Imgs/" --test_gt_path "Dataset/TestDataset/CHAMELEON/GT/" --save_path "results/CHAMELEON/"
CUDA_VISIBLE_DEVICES="0" \
python test.py --checkpoint "trainpth3/Net_epoch_best_106.pth" --test_image_path "Dataset/TestDataset/COD10K/Imgs/" --test_gt_path "Dataset/TestDataset/COD10K/GT/" --save_path "results/COD10K/"
CUDA_VISIBLE_DEVICES="0" \
python test.py --checkpoint "trainpth3/Net_epoch_best_106.pth" --test_image_path "Dataset/TestDataset/NC4K/Imgs/" --test_gt_path "Dataset/TestDataset/NC4K/GT/" --save_path "results/NC4K/"

#CUDA_VISIBLE_DEVICES="0" \
#python test.py --checkpoint "Net_epoch_best.pth" --test_image_path "Dataset/TestDataset/CAMO/Imgs/" --test_gt_path "Dataset/TestDataset/CAMO/GT/" --save_path "results/CAMO/"
#
