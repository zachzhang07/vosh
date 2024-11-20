import os
import sys
import argparse


def do_system(arg):
    print(f"==== running: {arg}")
    err = os.system(arg)
    if err:
        print("FATAL: command failed")
        sys.exit(err)


def parse_args():
    parser = argparse.ArgumentParser(description="fully evaluate the mipnerf-360 dataset")
    parser.add_argument('path', type=str, help='path to the mipnerf-360 dataset directory')
    parser.add_argument('--workspace', type=str, default='./output', help='path to the workspace directory')

    return parser.parse_args()


def collect_result(path):
    path = os.path.join(path, 'log_vosh.txt')
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('PSNR'):
                psnr = float(line.split('=')[1].strip())
            elif line.startswith('SSIM'):
                ssim = float(line.split('=')[1].strip())
            elif line.startswith('LPIPS'):
                lpips = float(line.split('=')[1].strip())
    return psnr, ssim, lpips


if __name__ == "__main__":
    args = parse_args()

    light_psnr_list, base_psnr_list, = [], []
    light_ssim_list, base_ssim_list = [], []
    light_lpips_list, base_lpips_list = [], []
    # run all outdoor scenes
    for scene in ['bicycle', 'garden', 'stump']:
        data_path = os.path.join(args.path, scene)
        vol_path = os.path.join(args.workspace, scene)
        mesh_path = os.path.join(args.workspace, scene + "_mesh")
        light_path = os.path.join(args.workspace, scene + "_light")
        base_path = os.path.join(args.workspace, scene + "_base")

        do_system(f'python main_vol.py {data_path} --workspace {vol_path}')
        do_system(f'python main_mesh.py {data_path} --vol_path {vol_path} --workspace {mesh_path}')
        do_system(f'python main_vosh.py {data_path} --vol_path {mesh_path} --workspace {light_path} '
                  f'--lambda_mesh_weight 0.01 --mesh_select 1.0 --keep_center 0.25 --lambda_bg_weight 0.01 '
                  f'--use_mesh_occ_grid --mesh_check_ratio 8 --no_baking')
        do_system(f'python main_vosh.py {data_path} --vol_path {mesh_path} --workspace {base_path} '
                  f'--lambda_mesh_weight 0.001 --mesh_select 0.9 --keep_center 0.25 --lambda_bg_weight 0.01 --no_baking')

        psnr, ssim, lpips = collect_result(light_path)
        light_psnr_list.append(psnr)
        light_ssim_list.append(ssim)
        light_lpips_list.append(lpips)
        print(f"light model for {light_path}:\npsnr: {psnr}, ssim: {ssim}, lpips: {lpips}")

        psnr, ssim, lpips = collect_result(base_path)
        base_psnr_list.append(psnr)
        base_ssim_list.append(ssim)
        base_lpips_list.append(lpips)
        print(f"base model for {base_path}:\npsnr: {psnr}, ssim: {ssim}, lpips: {lpips}")

    outdoor_result = f"\n\033[1mVosh-light in outdoor scenes: PSNR={sum(light_psnr_list) / len(light_psnr_list):.2f}, " \
                     f"SSIM={sum(light_ssim_list) / len(light_ssim_list):.3f}, " \
                     f"LPIPS={sum(light_lpips_list) / len(light_lpips_list):.3f} \n" \
                     f"Vosh-base in outdoor scenes: PSNR={sum(base_psnr_list) / len(base_psnr_list):.2f}, " \
                     f"SSIM={sum(base_ssim_list) / len(base_ssim_list):.3f}, " \
                     f"LPIPS={sum(base_lpips_list) / len(base_lpips_list):.3f}\033[0m"

    light_psnr_list, base_psnr_list, = [], []
    light_ssim_list, base_ssim_list = [], []
    light_lpips_list, base_lpips_list = [], []
    # run all indoor scenes
    for scene in ['room', 'counter', 'kitchen', 'bonsai']:
        data_path = os.path.join(args.path, scene)
        vol_path = os.path.join(args.workspace, scene)
        mesh_path = os.path.join(args.workspace, scene + "_mesh")
        light_path = os.path.join(args.workspace, scene + "_light")
        base_path = os.path.join(args.workspace, scene + "_base")

        do_system(f'python main_vol.py {data_path} --workspace {vol_path} --bound 16 '
                  f'--max_edge_len 0.3 --clean_min_f 16')
        do_system(f'python main_mesh.py {data_path} --vol_path {vol_path} --workspace {mesh_path} --bound 16 '
                  f'--max_edge_len 0.3 --clean_min_f 16')
        do_system(f'python main_vosh.py {data_path} --vol_path {mesh_path} --workspace {light_path} --no_baking '
                  f'--lambda_mesh_weight 0.01 --mesh_select 1.0 --keep_center 0.75 --lambda_bg_weight 0.01 '
                  f'--use_mesh_occ_grid --mesh_check_ratio 8 --bound 16')
        do_system(f'python main_vosh.py {data_path} --vol_path {mesh_path} --workspace {base_path} --no_baking '
                  f'--lambda_mesh_weight 0.001 --mesh_select 0.9 --keep_center 0.75 --lambda_bg_weight 0.01 --bound 16')

        psnr, ssim, lpips = collect_result(light_path)
        light_psnr_list.append(psnr)
        light_ssim_list.append(ssim)
        light_lpips_list.append(lpips)
        print(f"light model for {light_path}:\npsnr: {psnr}, ssim: {ssim}, lpips: {lpips}")

        psnr, ssim, lpips = collect_result(base_path)
        base_psnr_list.append(psnr)
        base_ssim_list.append(ssim)
        base_lpips_list.append(lpips)
        print(f"base model for {base_path}:\npsnr: {psnr}, ssim: {ssim}, lpips: {lpips}")

    indoor_result = f"\n\033[1mVosh-light in indoor scenes: PSNR={sum(light_psnr_list) / len(light_psnr_list):.2f}, " \
                    f"SSIM={sum(light_ssim_list) / len(light_ssim_list):.3f}, " \
                    f"LPIPS={sum(light_lpips_list) / len(light_lpips_list):.3f} \n" \
                    f"Vosh-base in indoor scenes: PSNR={sum(base_psnr_list) / len(base_psnr_list):.2f}, " \
                    f"SSIM={sum(base_ssim_list) / len(base_ssim_list):.3f}, " \
                    f"LPIPS={sum(base_lpips_list) / len(base_lpips_list):.3f}\033[0m"

    print(outdoor_result)
    print(indoor_result)

    # write results to workspace
    with open(os.path.join(args.workspace, 'full_eval_360.txt'), 'w') as f:
        f.write(outdoor_result)
        f.write(indoor_result)