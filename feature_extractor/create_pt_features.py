import torch
import torch.nn as nn
from math import floor
import os
from torchvision import transforms
import time
from clam_base_utils.clam_datasets.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP, Whole_Slide_Bag_FP_Patch_Dir
from torch.utils.data import DataLoader
import argparse
from clam_base_utils.utils.utils import collate_features
from clam_base_utils.utils.file_utils import save_hdf5
import h5py
from clam_base_utils.utils.model_utils import get_transforms, get_backbone, feature_extractor_adapter
import openslide
import pandas as pd
import csv
import tempfile


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def valid_feature_outputs(h5_path, pt_path):
	try:
		if not os.path.exists(h5_path) or not os.path.exists(pt_path):
			return False
		with h5py.File(h5_path, "r") as handle:
			if "features" not in handle or "coords" not in handle:
				return False
			features = handle["features"]
			coords = handle["coords"]
			if features.ndim != 2 or coords.ndim != 2:
				return False
			if features.shape[0] == 0 or features.shape[0] != coords.shape[0]:
				return False
			h5_shape = tuple(features.shape)
		loaded = torch.load(pt_path, map_location="cpu", weights_only=True)
		return isinstance(loaded, torch.Tensor) and tuple(loaded.shape) == h5_shape
	except (OSError, RuntimeError, ValueError, KeyError, EOFError):
		return False


def record_failure(path, slide_id, slide_path, error):
	if not path:
		return
	os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
	write_header = not os.path.exists(path)
	with open(path, "a", newline="", encoding="utf-8") as handle:
		writer = csv.writer(handle)
		if write_header:
			writer.writerow(["slide_id", "wsi_path", "error"])
		writer.writerow([slide_id, slide_path, str(error)])


def compute_w_loader(args,file_path, patch_img_save_dir,output_path, wsi, model,
 	batch_size = 8, verbose = 0, print_every=20, pretrained=True, 
	custom_downsample=1, target_patch_size=-1, num_workers = 4):
	"""
	args:
		file_path: directory of bag (.h5 file)
		output_path: directory to save computed features (.h5 file)
		model: pytorch model
		batch_size: batch_size for computing features in batches
		verbose: level of feedback
		pretrained: use weights pretrained on imagenet
		custom_downsample: custom defined downscale factor of image patches
		target_patch_size: custom defined, rescaled image size before embedding
	"""

	custom_transforms = get_transforms(args.backbone)
	if patch_img_save_dir != None:
		dataset = Whole_Slide_Bag_FP_Patch_Dir(file_path=file_path, wsi=wsi, pretrained=pretrained, 
		custom_downsample=custom_downsample,custom_transforms=custom_transforms,target_patch_size=target_patch_size,patch_img_save_dir=patch_img_save_dir)
	else:
		dataset = Whole_Slide_Bag_FP(file_path=file_path, wsi=wsi, pretrained=pretrained, 
			custom_downsample=custom_downsample,custom_transforms=custom_transforms,target_patch_size=target_patch_size)
	kwargs = {'num_workers': num_workers, 'pin_memory': True} if device.type == "cuda" else {}
	loader = DataLoader(dataset=dataset, batch_size=batch_size, **kwargs, collate_fn=collate_features)

	if verbose > 0:
		print('processing {}: total of {} batches'.format(file_path,len(loader)))

	mode = 'w'
	for count, (batch, coords) in enumerate(loader):
		
		with torch.no_grad():	
			if count % print_every == 0:
				print('batch {}/{}, {} files processed'.format(count, len(loader), count * batch_size))
			batch = batch.to(device, non_blocking=True)
			# print(batch.shape)
			features = feature_extractor_adapter(model, batch,args.backbone)
			features = features.cpu().numpy()

			asset_dict = {'features': features, 'coords': coords}
			save_hdf5(output_path, asset_dict, attr_dict= None, mode=mode)
			mode = 'a'
	
	return output_path


def main(args):

	print('initializing dataset')
	process_wsi_paths_csv = args.process_wsi_paths_csv
	if process_wsi_paths_csv != None:
		if os.path.exists(process_wsi_paths_csv):
			process_csv_path = process_wsi_paths_csv
			head = 'wsi_path'
	else:
		process_csv_path = os.path.join(args.data_h5_dir, 'process_list_autogen.csv')
		head = 'slide_id'
		if not os.path.exists(process_csv_path):
			raise NotImplementedError


	custom_downsample = 1
	bags_dataset = Dataset_All_Bags(process_csv_path,head)
	os.makedirs(args.feat_dir, exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'pt_files'), exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)
	print('loading model checkpoint')
	model = get_backbone(args.backbone, device, args.pretrained_weights_dir)
	if torch.cuda.device_count() > 1:
		model = nn.DataParallel(model)
	model.eval()
	total = len(bags_dataset)
	for bag_candidate_idx in range(total):
		if bag_candidate_idx % args.num_shards != args.shard_index:
			continue
		now_slide_ext = '.'+bags_dataset[bag_candidate_idx].split('.')[-1]
		slide_file_path = bags_dataset[bag_candidate_idx]
		slide_id = os.path.basename(bags_dataset[bag_candidate_idx]).split(now_slide_ext)[0]
		bag_name = slide_id+'.h5'
		h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)
		if not os.path.exists(h5_file_path):
			continue
		patch_img_save_dir = None
		if args.use_patch_img:
			patch_img_save_dir = os.path.join(args.data_h5_dir,'patch_imgs',slide_id)
			print(patch_img_save_dir)
			assert os.path.exists(patch_img_save_dir)
		
		print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
		print(slide_id)

		output_path = os.path.join(args.feat_dir, 'h5_files', bag_name)
		pt_path = os.path.join(args.feat_dir, 'pt_files', slide_id + '.pt')
		if not args.no_auto_skip and valid_feature_outputs(output_path, pt_path):
			print('skipped {}'.format(slide_id))
			continue 

		for stale_path in (output_path, pt_path):
			if os.path.exists(stale_path):
				os.remove(stale_path)
		fd, temp_h5_path = tempfile.mkstemp(
			prefix=f".{slide_id}.", suffix=".h5.tmp",
			dir=os.path.dirname(output_path)
		)
		os.close(fd)
		os.remove(temp_h5_path)
		temp_pt_path = pt_path + f".tmp.{os.getpid()}"
		time_start = time.time()
		if slide_file_path.endswith('.sdpc'):
			try:
				import opensdpc
			except:
				raise ImportError('opensdpc has not been installed, please run pip install opensdpc (https://github.com/WonderLandxD/opensdpc)')
			try:
				wsi = opensdpc.OpenSdpc(slide_file_path)
			except:
				print(f'{slide_file_path} can not open')
				continue
		else:
			try:
				wsi = openslide.open_slide(slide_file_path)
			except:
				print(f'{slide_file_path} can not open')
				continue
		try:
			output_file_path = compute_w_loader(args,h5_file_path, patch_img_save_dir, temp_h5_path, wsi,
		model = model, batch_size = args.batch_size, verbose = 1, print_every = 20, 
		custom_downsample=custom_downsample, target_patch_size=args.target_patch_size,num_workers= args.num_workers)
		except Exception as exc:
			print(f'feature extraction failed for {slide_id}: {exc}')
			record_failure(args.failure_log, slide_id, slide_file_path, exc)
			if os.path.exists(temp_h5_path):
				os.remove(temp_h5_path)
			continue

		time_elapsed = time.time() - time_start
		print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))
		with h5py.File(output_file_path, "r") as file:
			features = file['features'][:]
			coords_shape = file['coords'].shape
		print('features size: ', features.shape)
		print('coordinates size: ', coords_shape)
		features = torch.from_numpy(features)
		torch.save(features, temp_pt_path)
		os.replace(temp_h5_path, output_path)
		os.replace(temp_pt_path, pt_path)


def build_parser():
	parser = argparse.ArgumentParser(description='Feature Extraction')
	parser.add_argument('--data_h5_dir', default='', type=str)
	parser.add_argument('--process_wsi_paths_csv', default=None, type=str, help='prior process, head -> wsi_path, need when use_patch_img is False')
	parser.add_argument('--use_patch_img', default=False, action='store_true',
						help='read saved patch JPEGs instead of sampling the WSI')
	parser.add_argument('--feat_dir', type=str, default='')
	parser.add_argument('--batch_size', type=int, default=256)
	parser.add_argument('--num_workers', type=int, default=4)
	parser.add_argument('--no_auto_skip', default=False, action='store_true')
	parser.add_argument('--target_patch_size', type=int, default=224)
	parser.add_argument('--backbone', default='gigapath', type=str, choices=['vit_s_imagenet','resnet50_imagenet','plip','conch','uni','ctranspath','gigapath','virchow','virchow_v2','conch_v1_5','uni_v2','hoptimus_v0','hoptimus_v1','midnight'], help='backbone model')
	parser.add_argument('--pretrained_weights_dir', type=str, default='/mnt/net_sda/lxt/Other_Pathology_Model/GigaPath_weights', help='dir to the pretrained backbone')
	parser.add_argument('--num_shards', type=int, default=1)
	parser.add_argument('--shard_index', type=int, default=0)
	parser.add_argument('--failure_log', type=str, default=None)
	return parser


if __name__ == '__main__':
	parser = build_parser()
	args = parser.parse_args()
	if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
		parser.error('--shard_index must be in [0, --num_shards)')
	main(args)
