import argparse
from nerf.utils import *
# from nerf.gui import NeRFGUI
from nerf import math
import time

if __name__ == '__main__':
    start_t = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--ckpt', type=str, default='latest')
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    parser.add_argument('--fast_baking', action='store_true',
                        help="faster baking at the cost of maybe missing blocks at background")

    ### new options
    parser.add_argument('--render', type=str, default='grid', choices=['grid'])
    parser.add_argument('--criterion', type=str, default='MSE', choices=['L1', 'MSE'])
    parser.add_argument('--max_edge_len', type=float, default=0.5)
    parser.add_argument('--min_iso_size', type=int, default=100)
    parser.add_argument('--alpha_thres', type=float, default=0.005,
                        help="initial learning rate for vert optimization")
    parser.add_argument('--cascade_list', type=float, nargs='*', default=[0.5, 1.0, 1.5, 2.0])
    parser.add_argument('--decimate_target', type=int, nargs='*', default=[5e5, 5e5, 2e6, 2e6],
                        help="decimate target for number of triangles, <=0 to disable")

    ### mesh options
    parser.add_argument('--visibility_mask_dilation', type=int, default=50, help="visibility dilation")
    parser.add_argument('--clean_min_f', type=int, default=25, help="mesh clean: min face count for isolated mesh")
    parser.add_argument('--clean_min_d', type=int, default=10, help="mesh clean: min diameter for isolated mesh")
    parser.add_argument('--ssaa', type=int, default=2, help="super sampling anti-aliasing ratio")
    parser.add_argument('--texture_size', type=int, default=4096, help="exported texture resolution")

    ### model options
    # parser.add_argument('--backbone', type=str, default='merf_new', choices=['merf_new'], help="backbone type")
    parser.add_argument('--grid_resolution', type=int, default=1024)
    parser.add_argument('--triplane_resolution', type=int, default=-1)

    ### testing options
    parser.add_argument('--save_cnt', type=int, default=1,
                        help="save checkpoints for $ times during training")
    parser.add_argument('--eval_cnt', type=int, default=1,
                        help="perform validation for $ times during training")

    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--test_no_video', action='store_true', help="test mode: do not save video")
    parser.add_argument('--test_no_baking', action='store_true', help="test mode: do not save baking")
    parser.add_argument('--test_no_mesh', action='store_true', help="test mode: do not save mesh")
    parser.add_argument('--camera_traj', type=str, default='path',
                        help="interp for interpolation, circle for circular camera")

    ### dataset options
    parser.add_argument('--data_format', type=str, default='colmap', choices=['nerf', 'colmap', 'dtu'])
    parser.add_argument('--train_split', type=str, default='train', choices=['train', 'all'])
    parser.add_argument('--test_split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--preload', action='store_true',
                        help="preload all data into GPU, accelerate training but use more GPU memory")
    parser.add_argument('--random_image_batch', action='store_true',
                        help="randomly sample rays from all images per step in training")
    parser.add_argument('--downscale', type=int, default=4, help="downscale training images")
    parser.add_argument('--bound', type=float, default=128,
                        help="assume the scene is bounded in box[-bound, bound]^3, "
                             "if > 1, will invoke adaptive ray marching.")
    parser.add_argument('--scale', type=float, default=-1,
                        help="scale camera location into box[-bound, bound]^3, "
                             "-1 means automatically determine based on camera poses..")
    parser.add_argument('--offset', type=float, nargs='*', default=[0, 0, 0], help="offset of camera location")
    parser.add_argument('--enable_cam_near_far', action='store_true',
                        help="colmap mode: use the sparse points to estimate camera near far per view.")
    parser.add_argument('--enable_cam_center', action='store_true',
                        help="use camera center instead of sparse point center (colmap dataset only)")
    parser.add_argument('--min_near', type=float, default=0.2, help="minimum near distance for camera")
    parser.add_argument('--T_thresh', type=float, default=2e-4,
                        help="minimum transmittance to continue ray marching")

    ### optimization options
    parser.add_argument('--lr_init', type=float, default=1e-2, help="The initial learning rate")
    parser.add_argument('--lr_final', type=float, default=1e-3, help="The final learning rate")
    parser.add_argument('--lr_delay_steps', type=int, default=100,
                        help="The number of 'warmup' learning steps")
    parser.add_argument('--lr_delay_mult', type=float, default=0.01,
                        help="How much sever the 'warmup' should be")
    # grad_max_norm: float = 0.001  # Gradient clipping magnitude, disabled if == 0.
    # grad_max_val: float = 0.0  # Gradient clipping value, disabled if == 0.

    ### training options
    parser.add_argument('--iters', type=int, default=25000, help="training iters")
    # parser.add_argument('--lr', type=float, default=1e-3, help="initial learning rate")
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    # parser.add_argument('--max_steps', type=int, default=1024,
    #                     help="max num steps sampled per ray (only valid when using --cuda_ray)")
    parser.add_argument('--num_steps', type=int, nargs='*', default=[128, 64, 32],
                        help="num steps sampled per ray for each proposal level (only valid when NOT using --cuda_ray)")
    parser.add_argument('--contract', action='store_true',
                        help="apply spatial contraction as in MERF, only work for bound > 1, will override bound to 2.")
    parser.add_argument('--enable_dense_depth', action='store_true', help="dense depth supervision")
    parser.add_argument('--background', type=str, default='random',
                        choices=['white', 'random', 'last_sample'], help="training background mode")

    # parser.add_argument('--update_extra_interval', type=int, default=16,
    #                     help="iter interval to update extra status (only valid when using --cuda_ray)")
    parser.add_argument('--max_ray_batch', type=int, default=4096 * 4,
                        help="batch size of rays at inference to avoid OOM (only valid when NOT using --cuda_ray)")
    # parser.add_argument('--grid_size', type=int, default=128, help="density grid resolution")
    parser.add_argument('--mark_untrained', action='store_true', help="mark_untrained grid")
    parser.add_argument('--dt_gamma', type=float, default=1 / 256,
                        help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, "
                             ">0 to accelerate rendering (but usually with worse quality)")
    parser.add_argument('--density_thresh', type=float, default=10,
                        help="threshold for density grid to be occupied")
    parser.add_argument('--diffuse_step', type=int, default=0,
                        help="training iters that only trains diffuse color for better initialization")

    # batch size related
    parser.add_argument('--num_rays', type=int, default=4096,
                        help="num rays sampled per image for each training step")
    parser.add_argument('--adaptive_num_rays', action='store_true',
                        help="adaptive num rays for more efficient training")
    parser.add_argument('--num_points', type=int, default=2 ** 18,
                        help="target num points for each training step, only work with adaptive num_rays")

    # regularizations
    parser.add_argument('--lambda_entropy', type=float, default=0, help="loss scale")
    parser.add_argument('--lambda_tv', type=float, default=0, help="loss scale")
    parser.add_argument('--lambda_proposal', type=float, default=1,
                        help="loss scale (only for non-cuda-ray mode)")
    parser.add_argument('--lambda_distort', type=float, default=0.01,
                        help="loss scale (only for non-cuda-ray mode)")
    parser.add_argument('--lambda_specular', type=float, default=1e-5, help="loss scale")
    parser.add_argument('--lambda_depth', type=float, default=0, help="loss scale")
    parser.add_argument('--lambda_sparsity', type=float, default=0.01, help="loss scale")
    parser.add_argument('--lambda_mask', type=float, default=0.1, help="loss scale")

    ### GUI options
    parser.add_argument('--vis_pose', action='store_true', help="visualize the poses")
    parser.add_argument('--gui', action='store_true', help="start a GUI")
    parser.add_argument('--W', type=int, default=1000, help="GUI width")
    parser.add_argument('--H', type=int, default=1000, help="GUI height")
    parser.add_argument('--radius', type=float, default=1, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=50, help="default GUI camera fovy")
    parser.add_argument('--max_spp', type=int, default=1, help="GUI rendering max sample per pixel")

    opt = parser.parse_args()

    opt.fp16 = True
    opt.preload = True
    opt.contract = True
    opt.adaptive_num_rays = True

    opt.enable_cam_center = True
    opt.enable_cam_near_far = True

    # assert opt.alpha_thres == 0.005
    opt.random_image_batch = True

    if opt.contract:
        # mark untrained is not correct in contraction mode...
        opt.mark_untrained = False

    if opt.data_format == 'colmap':
        from nerf.colmap_provider import ColmapDataset as NeRFDataset
    elif opt.data_format == 'dtu':
        from nerf.dtu_provider import NeRFDataset
    else:  # nerf
        from nerf.provider import NeRFDataset

    #todo: change contract by opt.data_format
    if opt.data_format == 'nerf':
        # opt.bound = 1
        opt.min_near = 0.05
        # opt.diffuse_step = 0
        # opt.cascade_list = [0.5, 1.0, 2.0]
        # opt.cascade_list = [0.5, 1.0]
        opt.contract = False
        #opt.alpha_thres = 0.001
        opt.decimate_target = [1e5, 1e5, 1e5]
        opt.random_image_batch = True
        opt.clean_min_f = 16
        opt.clean_min_d = 10
        opt.visibility_mask_dilation = 5

    #todo end

    seed_everything(opt.seed)

    # if opt.backbone == 'merf_new':
    #     from nerf.network_vosh import NeRFNetwork
    # else:
    #     raise NotImplementedError
    from nerf.network_vosh import NeRFNetwork

    if 'stump' in opt.path:
        opt.diffuse_step = 5000

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = NeRFNetwork(opt)

    if opt.criterion == 'L1':
        criterion = torch.nn.SmoothL1Loss(reduction='none')
    elif opt.criterion == 'MSE':
        criterion = torch.nn.MSELoss(reduction='none')
    else:
        raise NotImplementedError

    if opt.test:
        trainer = Trainer('vol', opt, model, device=device, workspace=opt.workspace, criterion=criterion,
                          fp16=opt.fp16, use_checkpoint=opt.ckpt)

        if not opt.test_no_video:
            test_loader = NeRFDataset(opt, device=device, type=opt.test_split).dataloader()

            if test_loader.has_gt:
                trainer.metrics = [PSNRMeter(), SSIMMeter(), LPIPSMeter(device=device)]  # set up metrics
                trainer.evaluate(test_loader, name='test')  # blender has gt, so evaluate it.

            trainer.test(test_loader, write_video=True)  # test and save video

    else:
        optimizer = lambda model: torch.optim.Adam(model.get_params(1.0), eps=1e-15)

        train_loader = NeRFDataset(opt, device=device, type=opt.train_split).dataloader()

        max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32)
        save_interval = max(1, max_epoch // max(1, opt.save_cnt))
        eval_interval = max(1, max_epoch // max(1, opt.eval_cnt))
        print(f'[INFO] max_epoch {max_epoch}, eval every {eval_interval}.')

        # colmap can estimate a more compact AABB
        # if not opt.contract and opt.data_format == 'colmap':
        #     model.update_aabb(train_loader._data.pts_aabb)

        # scheduler = lambda optimizer: torch.optim.lr_scheduler.LambdaLR(optimizer,
        #                                                                 lambda iter: 0.1 ** (iter / opt.iters))
        scheduler = lambda optimizer: (
            torch.optim.lr_scheduler.LambdaLR(optimizer, lambda iter:
            math.learning_rate_decay(iter + 1, opt.lr_init, opt.lr_final, opt.iters,
                                     opt.lr_delay_steps, opt.lr_delay_mult)))

        trainer = Trainer('vol', opt, model, device=device, workspace=opt.workspace, optimizer=optimizer,
                          criterion=criterion, ema_decay=None, fp16=opt.fp16, lr_scheduler=scheduler,
                          scheduler_update_every_step=True, use_checkpoint=opt.ckpt, eval_interval=eval_interval,
                          save_interval=save_interval)

        if opt.gui:
            # gui = NeRFGUI(opt, trainer, train_loader)
            # gui.render()
            pass

        else:
            valid_loader = NeRFDataset(opt, device=device, type='val').dataloader()

            trainer.metrics = [PSNRMeter(), ]

            trainer.train(train_loader, valid_loader, max_epoch)

            # last validation
            trainer.metrics = [PSNRMeter(), SSIMMeter(), LPIPSMeter(device=device)]
            trainer.evaluate(valid_loader)

            # also test
            test_loader = NeRFDataset(opt, device=device, type=opt.test_split).dataloader()

            if test_loader.has_gt:
                trainer.evaluate(test_loader, name='test')  # blender has gt, so evaluate it.

            # trainer.test(test_loader, write_video=True) # test and save video

            all_loader = NeRFDataset(opt, device=device, type='train_all')
            all_loader.training = False  # load full image from train split
            all_loader = all_loader.dataloader()

            occ_grid_list = sorted(glob.glob(f'{opt.workspace}/merf_occ_grid_vol.pt'))
            if occ_grid_list:
                print(f'[INFO] Load occ_grid from {occ_grid_list[-1]}')
                occ_grid = torch.load(occ_grid_list[-1])
                print(f'[INFO] Occupancy rate: '
                      f'{occ_grid.sum().item() / occ_grid.numel() * 100.:.2f}% '
                      f'with resolution {occ_grid.shape[0]}')
            else:
                occ_grid = trainer.cal_occ_grid(all_loader)
                torch.save(occ_grid, os.path.join(opt.workspace, 'merf_occ_grid_vol.pt'))

            # trainer.save_baking(loader=all_loader, occupancy_grid=occ_grid)

            trainer.voxel_to_mesh(occ_grid.cuda(), resolution=opt.grid_resolution, loader=all_loader)

            voxel_error_grid_list = sorted(glob.glob(f'{opt.workspace}/voxel_error_grid*.pt'))
            if voxel_error_grid_list:
                print(f'[INFO] Load voxel_error_grid from {voxel_error_grid_list[-1]}')
                voxel_error_grid = torch.load(voxel_error_grid_list[-1]).to(device)
            else:
                voxel_error_grid = trainer.run_voxel_error_epoch(all_loader, resolution=512).half()
                torch.save(voxel_error_grid, os.path.join(opt.workspace, 'voxel_error_grid.pt'))

            end_t = time.time()
            print(f"[INFO] volume stage takes {(end_t - start_t) / 60:.6f} minutes.")
